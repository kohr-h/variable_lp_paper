"""Bimodal tomography with TGV regularizer."""

import matplotlib.pyplot as plt
import numpy as np
import imageio
import odl
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


# --- Set up the inverse problem for the bad data --- #


# Initialize gradient and 2nd order derivative operator
grad = odl.Gradient(reco_space, pad_mode='order1')
eps = odl.DiagonalOperator(grad, reco_space.ndim)
domain = odl.ProductSpace(grad.domain, eps.domain)

# Assemble operators and functionals for the solver

# The linear operators are
# 1. ray transform on the first component for the data matching
# 2. gradient of component 1 - component 2 for the auxiliary functional
# 3. eps on the second component
# 4. projection onto the first component

L = [
    ray_trafo * odl.ComponentProjection(domain, 0),
    odl.ReductionOperator(grad, odl.ScalingOperator(grad.range, -1)),
    eps * odl.ComponentProjection(domain, 1),
    odl.ComponentProjection(domain, 0),
    ]

# The functionals are
# 1. L2 data matching
# 2. regularization parameter 1 times L1 norm on the range of the gradient
# 3. regularization parameter 2 times L1 norm on the range of eps
# 4. box indicator on the reconstruction space

data_matching = odl.solvers.L2NormSquared(ray_trafo.range).translated(bad_data)
reg_param1 = 5e-2
regularizer1 = reg_param1 * odl.solvers.L1Norm(grad.range)
reg_param2 = 5e-2
regularizer2 = reg_param2 * odl.solvers.L1Norm(eps.range)
box_constr = odl.solvers.IndicatorBox(reco_space, 0, 1)

g = [data_matching, regularizer1, regularizer2, box_constr]

# Don't use f
f = odl.solvers.ZeroFunctional(domain)

# Create callback that prints the iteration number and shows partial results
callback_fig = None


def show_first(x):
    global callback_fig
    callback_fig = x[0].show('iterate', clim=[0, 1], fig=callback_fig)


# Uncomment the combined callback to also display iterates
callback = (odl.solvers.CallbackApply(show_first, step=10) &
            odl.solvers.CallbackPrintIteration())
# callback = odl.solvers.CallbackPrintIteration()

# Compute sigma parameters for the Douglas-Rachford solver, using a custom
# choice for tau and the norms of the operators in L
L_norms = [
    1.2 * odl.power_method_opnorm(L[0], maxiter=20),
    1.2 * odl.power_method_opnorm(L[1]),
    1.2 * odl.power_method_opnorm(L[2]),
    ]

tau = 0.1
tau, sigma = odl.solvers.douglas_rachford_pd_stepsize(L_norms, tau)
sigma = list(sigma)
sigma.append(1.0)

# Solve with initial guess x = 0
x = domain.zero()
odl.solvers.douglas_rachford_pd(x, f, g, L, tau=tau, sigma=sigma, lam=1.5,
                                niter=300, callback=callback)


# --- Compute FOMs --- #

with open('affine_bimodal_tomo_tgv_fom.txt', 'w+') as f:
    psnr = fom.psnr(x[0], phantom)
    print('PSNR:', psnr, file=f)
    ssim = fom.ssim(x[0], phantom)
    print('SSIM:', ssim, file=f)
    haarpsi = fom.haarpsi(x[0], phantom)
    print('HaarPSI:', haarpsi, file=f)

# --- Display images --- #


x[0].show(title='TGV Reconstruction', clim=[0, 1])

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
plt.savefig('affine_bimodal_tomo_tgv_profile.png')


# Display full image
plt.figure()
plt.imshow(np.rot90(x[0]), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_tgv_reco.png')
