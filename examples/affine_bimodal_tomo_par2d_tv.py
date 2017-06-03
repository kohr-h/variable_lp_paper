"""Bimodal tomography with TV regularizer."""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import odl


# --- Reconstruction space and phantom --- #

# Read image and transform from 'ij' storage to 'xy'
# NOTE: this requires the "pillow" package
image = np.rot90(scipy.misc.imread('affine_phantom.png'), k=-1)

reco_space = odl.uniform_discr([-10, -10], [10, 10], image.shape,
                               dtype='float32')
phantom = reco_space.element(image)


# --- Set up the forward operator --- #

# Make a fan beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi
angle_partition = odl.uniform_partition(0, 3 * np.pi / 4, 270)
# Detector: uniformly sampled, n = 300, min = -15, max = 15
detector_partition = odl.uniform_partition(-15, 15, 300)
# Parallel 2d geometry, so we have a reconstruction kernel
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Read data
# data = ray_trafo(phantom)
# good_data = data + 0.01 * np.max(data) * odl.phantom.white_noise(data.space)
# bad_data = data + 0.1 * np.max(data) * odl.phantom.white_noise(data.space)
bad_data = ray_trafo.range.element(
    np.load('affine_tomo_par2d_limang_bad_data.npy'))


# --- Set up the inverse problem for the bad data --- #


# Assemble operators and functionals for the solver
gradient = odl.Gradient(reco_space, pad_mode='order1')
lin_ops = [ray_trafo, gradient]
data_matching = odl.solvers.L2NormSquared(ray_trafo.range).translated(bad_data)
l1_func = odl.solvers.L1Norm(gradient.range)
# Left-multiplication version
reg_param = 1e2
regularizer = reg_param * l1_func

g = [data_matching, regularizer]
f = odl.solvers.IndicatorBox(reco_space, 0, 255)

# Uncomment the combined callback to also display iterates
callback = (odl.solvers.CallbackShow('iterate', step=20, clim=[0, 255]) &
            odl.solvers.CallbackPrintIteration())
# callback = odl.solvers.CallbackPrintIteration()

# Solve with initial guess x = data.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = reco_space.zero()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.1, sigma=[0.1, 0.02], lam=1.5,
                                niter=200, callback=callback)


# --- Display images --- #


# phantom.show(title='Phantom', clim=[-5, 5])
# data.show(title='Data')
x.show(title='TV Reconstruction', clim=[0, 255])
# Display horizontal profile
# fig = phantom.show(coords=[None, -4.25])
# x.show(coords=[None, -4.25], fig=fig, force_show=True)

# Create horizontal profile through the "tip"
phantom_slice = phantom[:, 35]
reco_slice = x[:, 35]
x_vals = reco_space.grid.coord_vectors[0]
plt.figure()
axes = plt.gca()
axes.set_ylim([80, 270])
plt.plot(x_vals, phantom_slice, label='Phantom')
plt.plot(x_vals, reco_slice, label='TV reconstruction')
plt.legend()
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_tv_profile.png')


# Display full image
plt.figure()
plt.imshow(np.rot90(x), cmap='bone', clim=[0, 255])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('affine_bimodal_tomo_tv_par2d_reco.png')
