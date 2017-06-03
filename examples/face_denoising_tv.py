"""Reference TV denoising of the affine example."""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy
import odl


# --- Reconstruction space, phantom and data --- #

# Convert "face" image to greyscale
image = Image.fromarray(scipy.misc.face()).convert('L')
# Transform from 'ij' storage to 'xy'
image = np.rot90(image, k=-1) / np.max(image)

reco_space = odl.uniform_discr([-10, -7.5], [10, 7.5], image.shape,
                               dtype='float32')
phantom = reco_space.element(image)
# data = phantom + odl.phantom.white_noise(reco_space) * 0.15 * np.max(phantom)
data = reco_space.element(np.load('face.npy'))


# --- Set up the inverse problem --- #


# Assemble operators and functionals for the solver
gradient = odl.Gradient(reco_space)
lin_ops = [odl.IdentityOperator(reco_space), gradient]
data_matching = odl.solvers.L2NormSquared(reco_space).translated(data)
reg_param = 5e-3
regularizer = reg_param * odl.solvers.L1Norm(gradient.range)
g = [data_matching, regularizer]
f = odl.solvers.IndicatorBox(reco_space, 0, 1.0)

# Uncomment the combined callback to also display iterates
callback = (odl.solvers.CallbackShow('iterate', step=20, clim=[0, 1.0]) &
            odl.solvers.CallbackPrintIteration())
# callback = odl.solvers.CallbackPrintIteration()

# Solve with initial guess x = data.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = data.copy()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.1, sigma=[0.2, 0.001], lam=1.5,
                                niter=100, callback=callback)


# --- Display images --- #


# phantom.show(title='Phantom', clim=[-5, 5])
# data.show(title='Data')
x.show(title='TV Reconstruction', clim=[0, 1.0])
# Display horizontal profile through the "tip"
fig = phantom.show(coords=[None, -1.4])
x.show(coords=[None, -1.4], fig=fig, force_show=True)

# Create horizontal profile through the nose
phantom_slice = phantom[:, 310]
reco_slice = x[:, 310]
x_vals = reco_space.grid.coord_vectors[0]
plt.figure()
axes = plt.gca()
axes.set_ylim([0, 0.9])
plt.plot(x_vals, phantom_slice, label='Phantom')
plt.plot(x_vals, reco_slice, label='TV reconstruction')
plt.legend()
plt.savefig('face_denoising_tv_profile.png')


# Display full images
plt.figure()
plt.imshow(np.rot90(x), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.savefig('face_denoising_tv_reco.png')


# Make detail images
plt.figure()
detail = x[500:800, 250:550]
plt.imshow(np.rot90(detail), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.savefig('face_denoising_tv_reco_detail.png')
