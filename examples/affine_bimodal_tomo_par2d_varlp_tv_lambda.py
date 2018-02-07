"""Bimodal tomography with variable Lp TV regularizer."""

import matplotlib.pyplot as plt
import numpy as np
import imageio
import odl
from odl.contrib import fom
import variable_lp


# --- Reconstruction space and phantom --- #

# Read image and transform from 'ij' storage to 'xy'
image = np.rot90(imageio.imread('affine_phantom.png'), k=-1)

reco_space = odl.uniform_discr([-10, -10], [10, 10], image.shape,
                               dtype='float32')
phantom = reco_space.element(image) / np.max(image)


# --- Set up the forward operator --- #

# Make a fan beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 3/4 * pi (limited angle)
angle_partition = odl.uniform_partition(0, 3 * np.pi / 4, 270)
# Detector: uniformly sampled, n = 300, min = -15, max = 15
detector_partition = odl.uniform_partition(-15, 15, 300)
# Parallel 2d geometry, so we have a reconstruction kernel
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cpu')

# Generate data with predictable randomness to make them reproducible
data = ray_trafo(phantom)
with odl.util.NumpyRandomSeed(123):
    good_data = (data +
                 0.01 * np.max(data) * odl.phantom.white_noise(data.space))
    bad_data = (data +
                0.1 * np.max(data) * odl.phantom.white_noise(data.space))


# --- Compute the exponent --- #


# The exponent is computed from the good data using feature reconstruction of
# FBP type with feature operator being the spatial gradient
def gaussian(x, **kwargs):
    s = kwargs.pop('s', 0.5)
    scaled = [xi / (np.sqrt(2) * s) for xi in x]
    return np.exp(-sum(xi ** 2 for xi in scaled))


def gaussian_det(x, **kwargs):
    s = kwargs.pop('s', 0.5)
    scaled = [xi / (np.sqrt(2) * s) for xi in x[1:]]
    return np.exp(-sum(xi ** 2 for xi in scaled))


def lambda_ft(x):
    det_var = x[1]
    return abs(det_var)


def deriv0_ft(x):
    angle = x[0]
    det_var = x[1]
    return np.sin(angle) * det_var


def deriv1_ft(x):
    angle = x[0]
    det_var = x[1]
    return -np.cos(angle) * det_var


# Fourier transform on the range of the FT along the detector axis, including
# padding
ran_shp = (ray_trafo.range.shape[0],
           ray_trafo.range.shape[1] * 2 - 1)
resizing = odl.ResizingOperator(ray_trafo.range, ran_shp=ran_shp,
                                pad_mode='order0')

fourier = odl.trafos.FourierTransform(resizing.range, axes=1, impl='pyfftw')
fourier = fourier * resizing

# Ramp filter part
alen = ray_trafo.geometry.motion_params.length
ramp_filter = 1 / (2 * alen) * fourier.range.element(lambda x: abs(x[1]))

# Smoothing filter (mollifier) for the reconstruction kernel (Fourier space)
gaussian_ker = fourier.domain.element(gaussian_det, s=0.15)
exp_filter = fourier(gaussian_ker)

# Filters for lambda reconstruction
lambda_filter_ft = fourier.range.element(lambda_ft)
lambda_kernel_ft = exp_filter * ramp_filter * lambda_filter_ft

# Filters for the feature operators (partial derivatives)
# feature_filter_0_ft = fourier.range.element(deriv0_ft)
# feature_kernel_0_ft = exp_filter * ramp_filter * feature_filter_0_ft
# feature_filter_1_ft = fourier.range.element(deriv1_ft)
# feature_kernel_1_ft = exp_filter * ramp_filter * feature_filter_1_ft

fbp_lambda = odl.tomo.fbp_op(ray_trafo, padding=True, filter=lambda_kernel_ft)
# fbp0 = odl.tomo.fbp_op(ray_trafo, padding=True, filter=feature_kernel_0_ft)
# fbp1 = odl.tomo.fbp_op(ray_trafo, padding=True, filter=feature_kernel_1_ft)

lambda_reco = fbp_lambda(good_data)
# good_reco0 = fbp0(good_data)
# good_reco1 = fbp1(good_data)
# combined = np.hypot(good_reco0, good_reco1)

# Remove jumps at the boundary, they're artificial
avg = np.mean(lambda_reco)
lambda_reco[:3, :] = avg
lambda_reco[-3:, :] = avg
lambda_reco[:, :3] = avg
lambda_reco[:, -3:] = avg


# We use the following procedure to generate the exponent from the reco g:
# - Compute a moderately smoothed version of the Laplacian L(g)
# - Take its absolute value and smooth it more aggressively
# - Multiply by 2 / max(L(g)), then clip at value 1.
#   This is to make the regions with high values broader.
# - Use 2 minus the result as exponent
def exp_kernel(x, s=0.5):
    scaled = [xi / (np.sqrt(2) * s) for xi in x]
    return np.exp(-sum(xi ** 2 for xi in scaled))


# Post-smoothing
fourier_reco_space = odl.trafos.FourierTransform(reco_space)
post_kernel = reco_space.element(gaussian, s=0.15)
post_kernel_ft = fourier_reco_space(post_kernel) * (2 * np.pi)
post_conv = fourier_reco_space.inverse * post_kernel_ft * fourier_reco_space

abs_lambda_reco = np.maximum(post_conv(np.abs(lambda_reco)), 0)
abs_lambda_reco -= np.min(abs_lambda_reco)
abs_lambda_reco *= 4 / np.max(abs_lambda_reco)
abs_lambda_reco[:] = np.minimum(abs_lambda_reco, 1)
exponent = 2.0 - abs_lambda_reco
exponent.show()


# --- Set up the inverse problem for the bad data --- #


# Assemble operators and functionals for the solver
gradient = (2e2 / reco_space.cell_volume *
            odl.Gradient(reco_space, pad_mode='order1'))
lin_ops = [ray_trafo, gradient]
data_matching = odl.solvers.L2NormSquared(ray_trafo.range).translated(bad_data)
varlp_func = variable_lp.VariableLpModular(gradient.range, exponent,
                                           impl='numba_parallel')
# Left-multiplication version
reg_param = 1e-4
# regularizer = reg_param * varlp_func
# Right-multiplication version
# reg_param = 8e-2
regularizer = varlp_func * reg_param

g = [data_matching, regularizer]
f = odl.solvers.IndicatorBox(reco_space, 0, 1)

# Use standard strategy to set solver parameters (see doc of
# `douglas_rachford_pd` for more on the convergence criterion)
ray_trafo_norm = odl.power_method_opnorm(ray_trafo, maxiter=10)
grad_norm = odl.power_method_opnorm(gradient,
                                    xstart=ray_trafo.adjoint(good_data),
                                    maxiter=20)
opnorms = [ray_trafo_norm, grad_norm]
num_ops = len(lin_ops)
tau = 0.1
sigmas = [3 / (tau * num_ops * opnorm ** 2) for opnorm in opnorms]

# Uncomment the combined callback to also display iterates
callback = (odl.solvers.CallbackShow('iterate', step=10, clim=[0, 1]) &
            odl.solvers.CallbackPrintIteration())
# callback = odl.solvers.CallbackPrintIteration()

# Solve with initial guess x = data.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = reco_space.zero()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=tau, sigma=sigmas, lam=1.5,
                                niter=200, callback=callback)


# --- Compute FOMs --- #

with open('affine_bimodal_tomo_par2d_varlp_tv_lambda_fom.txt', 'w+') as f:
    psnr = fom.psnr(x, phantom)
    print('PSNR:', psnr, file=f)
    ssim = fom.ssim(x, phantom)
    print('SSIM:', ssim, file=f)
    haarpsi = fom.haarpsi(x, phantom)
    print('HaarPSI:', haarpsi, file=f)

# --- Display images --- #


# phantom.show(title='Phantom', clim=[0, 1])
# data.show(title='Data')
x.show(title='TV-p Reconstruction', clim=[0, 1])
exponent.show(title='Exponent function', clim=[1, 2])
# Display horizontal profile
# fig = phantom.show(coords=[None, -4.25])
# x.show(coords=[None, -4.25], fig=fig, force_show=True)

# Create horizontal profile through the "tip"
phantom_slice = phantom[:, 35]
reco_slice = x[:, 35]
x_vals = reco_space.grid.coord_vectors[0]
plt.figure()
axes = plt.gca()
axes.set_ylim([0.3, 1.1])
plt.plot(x_vals, phantom_slice, label='Phantom')
plt.plot(x_vals, reco_slice, label='TV-p reconstruction')
plt.legend()
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_varlp_lambda_tv_bootstrap_profile.png')


# Display full images
plt.figure()
plt.imshow(np.rot90(x), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_varlp_lambda_tv_limang_reco.png')

plt.figure()
plt.imshow(np.rot90(exponent), cmap='bone', clim=[1, 2])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_varlp_lambda_tv_limang_exp.png')

plt.figure()
plt.imshow(np.rot90(phantom), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_phantom.png')

plt.figure()
plt.imshow(np.rot90(good_data), cmap='bone')
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_par2d_good_data.png')

plt.figure()
plt.imshow(np.rot90(bad_data), cmap='bone')
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_par2d_bad_data.png')
