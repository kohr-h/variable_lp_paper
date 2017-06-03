"""Reference TGV denoising of the square example."""

import numpy as np
import odl
import matplotlib.pyplot as plt


# --- Reconstruction space, phantom and data --- #

reco_space = odl.uniform_discr([-10, -10], [10, 10], (300, 300),
                               dtype='float32')


def square(x):
    return x[0] * ((np.abs(x[0]) <= 5) & (np.abs(x[1]) <= 5))


phantom = reco_space.element(square)
data = reco_space.element(np.load('square.npy'))


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
reg_param1 = 2e-1
regularizer1 = reg_param1 * odl.solvers.L1Norm(gradient.range)
reg_param2 = 5e-2
regularizer2 = reg_param2 * odl.solvers.L1Norm(eps.range)
box_constr = odl.solvers.IndicatorBox(reco_space, -5, 5)

g = [data_matching, regularizer1, regularizer2, box_constr]

# Don't use f
f = odl.solvers.ZeroFunctional(domain)

# Create callback that prints the iteration number and shows partial results
callback_fig = None


def show_first(x):
    global callback_fig
    callback_fig = x[0].show('iterate', clim=[-5, 5], fig=callback_fig)


# Uncomment the combined callback to also display iterates
# callback = (odl.solvers.CallbackApply(show_first, step=10) &
#             odl.solvers.CallbackPrintIteration())
callback = odl.solvers.CallbackPrintIteration()

# Solve with initial guess x = data.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = domain.element([data.copy(), eps.domain.zero()])
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.1, sigma=[0.1, 0.02, 0.001, 1], lam=1.5,
                                niter=200, callback=callback)


# --- Display images --- #


# phantom.show(title='Phantom', clim=[-5, 5])
# data.show(title='Data')
x[0].show(title='TGV Reconstruction', clim=[-5, 5])
# Display horizontal profile
# fig = phantom.show(coords=[None, 0], label='')
# x.show(coords=[None, 0], fig=fig, force_show=True)

# Create horizontal profile plot at the middle
phantom_slice = phantom[:, 150]
reco_slice = x[0][:, 150]
x_vals = reco_space.grid.coord_vectors[0]
plt.figure()
axes = plt.gca()
axes.set_ylim([-5.1, 5.1])
plt.plot(x_vals, phantom_slice, label='Phantom')
plt.plot(x_vals, reco_slice, label='TGV reconstruction')
plt.legend()
plt.tight_layout()
plt.savefig('square_denoising_tgv_profile.png')


# Display full image
plt.figure()
plt.imshow(np.rot90(x[0]), cmap='bone', clim=[-5, 5])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('square_denoising_tgv_reco.png')
