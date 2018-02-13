"""Bimodal tomography with variable Lp TV regularizer (disturbed exponent)."""

import matplotlib.pyplot as plt
import numpy as np
import odl
import imageio
import variable_lp
from odl.contrib import fom


# --- Reconstruction space and phantom --- #

# Read image and transform from 'ij' storage to 'xy'
image = np.rot90(imageio.imread('affine_phantom.png'), k=-1)

reco_space = odl.uniform_discr([-1, -1], [1, 1], image.shape, dtype='float32')
phantom = reco_space.element(image) / np.max(image)


# --- Set up the forward operator --- #

# Make a fan beam geometry with flat detector
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
detector_partition = odl.uniform_partition(-4, 4, 400)
geometry = odl.tomo.FanFlatGeometry(
    angle_partition, detector_partition, src_radius=40, det_radius=40)

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


# The exponent is computed from an FBP reconstruction using the good data
fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)
good_reco = fbp(good_data)


# We use the following procedure to generate the exponent from the reco g:
# - Compute a moderately smoothed version of the Laplacian L(g)
# - Take its absolute value and smooth it more aggressively
# - Multiply by 3 / max(L(g)), then clip at value 1.
#   This is to make the regions with high values broader.
# - Use 2 minus the result as exponent
def exp_kernel(x, s=0.5):
    scaled = [xi / (np.sqrt(2) * s) for xi in x]
    return np.exp(-sum(xi ** 2 for xi in scaled))


# Pre-smoothing convolution
fourier = odl.trafos.FourierTransform(reco_space)
pre_kernel = reco_space.element(exp_kernel, s=0.005)
pre_kernel_ft = fourier(pre_kernel) * (2 * np.pi)
pre_conv = fourier.inverse * pre_kernel_ft * fourier
smoothed_lapl = odl.Laplacian(reco_space, pad_mode='symmetric') * pre_conv
# Smoothed Laplacian of the data
abs_lapl = np.abs(smoothed_lapl(good_reco))
# Remove jumps at the boundary, they're artificial
avg = np.mean(abs_lapl)
abs_lapl[:5, :] = avg
abs_lapl[-5:, :] = avg
abs_lapl[:, :5] = avg
abs_lapl[:, -5:] = avg
# Post-smoothing
post_kernel = reco_space.element(exp_kernel, s=0.02)
post_kernel_ft = fourier(post_kernel) * (2 * np.pi)
post_conv = fourier.inverse * post_kernel_ft * fourier
conv_abs_lapl = np.maximum(post_conv(abs_lapl), 0)
conv_abs_lapl -= np.min(conv_abs_lapl)
conv_abs_lapl *= 3 / np.max(conv_abs_lapl)
conv_abs_lapl[:] = np.minimum(conv_abs_lapl, 1)
exponent = 2.0 - conv_abs_lapl
exponent[10:50, 59:69] = 1
exponent.show()


# Assemble operators and functionals for the solver
grad = odl.Gradient(reco_space, pad_mode='order1')
L = [ray_trafo, grad]
data_matching = odl.solvers.L2NormSquared(ray_trafo.range).translated(bad_data)
varlp_func = variable_lp.VariableLpModular(grad.range, exponent,
                                           impl='numba_parallel')
# Left-multiplication version
reg_param = 5e-2
regularizer = reg_param * varlp_func
# Right-multiplication version
#reg_param = 5e-3
#regularizer = varlp_func * reg_param

g = [data_matching, regularizer]
f = odl.solvers.IndicatorBox(reco_space, 0, 1)

# Compute sigma parameters for the Douglas-Rachford solver, using a custom
# choice for tau and the norms of the operators in L
ray_trafo_norm = 1.2 * odl.power_method_opnorm(ray_trafo, maxiter=20)
tau, sigma = odl.solvers.douglas_rachford_pd_stepsize([ray_trafo_norm, grad])

# Uncomment the combined callback to also display iterates
callback = (odl.solvers.CallbackShow('iterate', step=5, clim=[0, 1]) &
            odl.solvers.CallbackPrintIteration())
# callback = odl.solvers.CallbackPrintIteration()

# Solve with initial guess x = data.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = reco_space.zero()
odl.solvers.douglas_rachford_pd(x, f, g, L, tau=tau, sigma=sigma, lam=1.5,
                                niter=100, callback=callback)


# --- Compute FOMs --- #

with open('affine_bimodal_tomo_varlp_tv_disturbed_fom.txt', 'w+') as f:
    psnr = fom.psnr(x, phantom)
    print('PSNR:', psnr, file=f)
    ssim = fom.ssim(x, phantom)
    print('SSIM:', ssim, file=f)
    haarpsi = fom.haarpsi(x, phantom)
    print('HaarPSI:', haarpsi, file=f)

# --- Display images --- #


x.show(title='TV-p Reconstruction', clim=[0, 1])
exponent.show(title='Exponent function', clim=[1, 2])

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
plt.savefig('affine_bimodal_tomo_varlp_tv_disturbed_profile.png')


# Display full images
plt.figure()
plt.imshow(np.rot90(x), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_varlp_tv_disturbed_reco.png')

plt.figure()
plt.imshow(np.rot90(exponent), cmap='bone', clim=[1, 2])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_varlp_tv_disturbed_exp.png')
