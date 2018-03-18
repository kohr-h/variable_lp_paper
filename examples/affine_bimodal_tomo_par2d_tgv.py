"""Bimodal tomography with TGV regularizer."""

import imageio
import matplotlib.pyplot as plt
import numpy as np

import odl
from odl.contrib import fom

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


# --- Set up the inverse problem --- #


# Initialize gradient and 2nd order derivative operator
gradient = odl.Gradient(reco_space, pad_mode='order1')
eps = odl.DiagonalOperator(gradient, reco_space.ndim)
domain = odl.ProductSpace(gradient.domain, eps.domain)

# Assemble operators and functionals for the solver

# The linear operators are
# 1. ray transform on the first component for the data matching
# 2. gradient of component 1 - component 2 for the auxiliary functional
# 3. eps on the second component
# 4. projection onto the first component

lin_ops = [
    ray_trafo * odl.ComponentProjection(domain, 0),
    odl.ReductionOperator(gradient, odl.ScalingOperator(gradient.range, -1)),
    eps * odl.ComponentProjection(domain, 1),
    odl.ComponentProjection(domain, 0)
    ]

# The functionals are
# 1. L2 data matching
# 2. regularization parameter 1 times L1 norm on the range of the gradient
# 3. regularization parameter 2 times L1 norm on the range of eps
# 4. box indicator on the reconstruction space

data_matching = odl.solvers.L2NormSquared(ray_trafo.range).translated(bad_data)
reg_param1 = 3e-1
regularizer1 = reg_param1 * odl.solvers.L1Norm(gradient.range)
reg_param2 = 3e-1
regularizer2 = reg_param2 * odl.solvers.L1Norm(eps.range)
box_constr = odl.solvers.IndicatorBox(reco_space, 0, 1)

g = [data_matching, regularizer1, regularizer2, box_constr]

# Don't use f
f = odl.solvers.ZeroFunctional(domain)

# Use standard strategy to set solver parameters (see doc of
# `douglas_rachford_pd` for more on the convergence criterion).
# When computing the operator norms, we need to be a bit careful with the
# start value for differential operators (no constants).
linop_0_norm = odl.power_method_opnorm(lin_ops[0], maxiter=10)
xstart = ray_trafo.adjoint(good_data)
dom_xstart = domain.element([xstart, [xstart, xstart]])
linop_1_norm = odl.power_method_opnorm(lin_ops[1], xstart=dom_xstart,
                                       maxiter=20)
linop_2_norm = odl.power_method_opnorm(lin_ops[2], xstart=dom_xstart,
                                       maxiter=20)
# No need for the fourth one since it's used in an indicator function
# where the step size doesn't matter
opnorms = [linop_0_norm, linop_1_norm, linop_2_norm]
num_ops = len(opnorms)
tau = 0.1
sigmas = [3 / (tau * num_ops * opnorm ** 2) for opnorm in opnorms]
sigmas.append(1)


# Create callback that prints the iteration number and shows partial results
callback_fig = None


def show_first(x):
    global callback_fig
    callback_fig = x[0].show('iterate', clim=[0, 1], fig=callback_fig)


# Uncomment the combined callback to also display iterates
callback = (odl.solvers.CallbackApply(show_first, step=10) &
            odl.solvers.CallbackPrintIteration())
# callback = odl.solvers.CallbackPrintIteration()

# Solve with initial guess x = 0.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = domain.zero()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=tau, sigma=sigmas, lam=1.5,
                                niter=200, callback=callback)


# --- Compute FOMs --- #

with open('affine_bimodal_tomo_par2d_tgv_fom.txt', 'w+') as f:
    psnr = fom.psnr(x[0], phantom)
    print('PSNR:', psnr, file=f)
    ssim = fom.ssim(x[0], phantom)
    print('SSIM:', ssim, file=f)
    haarpsi = fom.haarpsi(x[0], phantom)
    print('HaarPSI:', haarpsi, file=f)

# --- Display images --- #


# phantom.show(title='Phantom', clim=[0, 1])
# data.show(title='Data')
x[0].show(title='TGV Reconstruction', clim=[0, 1])
# Display horizontal profile
# fig = phantom.show(coords=[None, -4.25])
# x.show(coords=[None, -4.25], fig=fig, force_show=True)

# Create horizontal profile through the "tip"
phantom_slice = phantom[:, 35]
reco_slice = x[0][:, 35]
x_vals = reco_space.grid.coord_vectors[0]
plt.figure()
axes = plt.gca()
axes.set_ylim([0.3, 1.1])
plt.plot(x_vals, phantom_slice, label='Phantom')
plt.plot(x_vals, reco_slice, label='TGV reconstruction')
plt.legend()
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_tgv_par2d_profile.png')


# Display full image
plt.figure()
plt.imshow(np.rot90(x[0]), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_tgv_par2d_limang_reco.png')
