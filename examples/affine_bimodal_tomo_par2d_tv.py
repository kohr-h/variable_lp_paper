"""Bimodal tomography with TV regularizer (using only the bad data)."""

import matplotlib.pyplot as plt
import numpy as np
import imageio
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


# --- Set up the inverse problem for the bad data --- #


# Assemble operators and functionals for the solver
gradient = odl.Gradient(reco_space, pad_mode='order1')
lin_ops = [ray_trafo, gradient]
data_matching = odl.solvers.L2NormSquared(ray_trafo.range).translated(bad_data)
l1_func = odl.solvers.L1Norm(gradient.range)
reg_param = 3e-1
regularizer = reg_param * l1_func

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

with open('affine_bimodal_tomo_par2d_tv_fom.txt', 'w+') as f:
    psnr = fom.psnr(x, phantom)
    print('PSNR:', psnr, file=f)
    ssim = fom.ssim(x, phantom)
    print('SSIM:', ssim, file=f)
    haarpsi = fom.haarpsi(x, phantom)
    print('HaarPSI:', haarpsi, file=f)

# --- Display images --- #


# phantom.show(title='Phantom', clim=[0, 1])
# data.show(title='Data')
x.show(title='TV Reconstruction', clim=[0, 1])
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
plt.plot(x_vals, reco_slice, label='TV reconstruction')
plt.legend()
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_tv_profile.png')


# Display full image
plt.figure()
plt.imshow(np.rot90(x), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_tv_par2d_reco.png')
