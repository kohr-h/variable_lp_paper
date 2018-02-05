"""Reference TGV denoising of the affine example."""


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.misc
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


# Initialize gradient and 2nd order derivative operator
gradient = odl.Gradient(reco_space)
eps = odl.DiagonalOperator(gradient, reco_space.ndim)
domain = odl.ProductSpace(gradient.domain, eps.domain)

# Assemble operators and functionals for the solver

# The linear operators are
# 1. identity on the first component for the data matching
# 2. gradient of component 1 - component 2 for the auxiliary functional
# 3. eps on the second component
# 4. projection onto the first component

lin_ops = [
    odl.ComponentProjection(domain, 0),
    odl.ReductionOperator(gradient, odl.ScalingOperator(gradient.range, -1)),
    eps * odl.ComponentProjection(domain, 1),
    odl.ComponentProjection(domain, 0)
    ]

# The functionals are
# 1. L2 data matching
# 2. regularization parameter 1 times L1 norm on the range of the gradient
# 3. regularization parameter 2 times L1 norm on the range of eps
# 4. box indicator on the reconstruction space

data_matching = odl.solvers.L2NormSquared(reco_space).translated(data)
reg_param1 = 5e-3
regularizer1 = reg_param1 * odl.solvers.L1Norm(gradient.range)
reg_param2 = 5e-6
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
callback = (odl.solvers.CallbackApply(show_first, step=5) &
            odl.solvers.CallbackPrintIteration())
# callback = odl.solvers.CallbackPrintIteration()

# Solve with initial guess x = data.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = domain.element([data.copy(), eps.domain.zero()])
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.1, sigma=[0.2, 0.001, 0.001, 1], lam=1.5,
                                niter=100, callback=callback)


# --- Display images --- #


# phantom.show(title='Phantom', clim=[-5, 5])
# data.show(title='Data')
x[0].show(title='TGV Reconstruction', clim=[0, 1.0])
# Display horizontal profile through the "tip"
fig = phantom.show(coords=[None, -1.4])
x[0].show(coords=[None, -1.4], fig=fig, force_show=True)

# Create horizontal profile through the nose
phantom_slice = phantom[:, 310]
reco_slice = x[0][:, 310]
x_vals = reco_space.grid.coord_vectors[0]
plt.figure()
axes = plt.gca()
axes.set_ylim([0, 0.9])
plt.plot(x_vals, phantom_slice, label='Phantom')
plt.plot(x_vals, reco_slice, label='TGV reconstruction')
plt.legend()
plt.savefig('face_denoising_tgv_profile.png')


# Display full images
plt.figure()
plt.imshow(np.rot90(x[0]), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.savefig('face_denoising_tgv_reco.png')


# Make detail images
plt.figure()
detail = x[0][500:800, 250:550]
plt.imshow(np.rot90(detail), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.savefig('face_denoising_tgv_reco_detail.png')
