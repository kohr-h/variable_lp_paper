"""Reference TGV denoising of the affine example."""

import imageio
import numpy as np

import odl

# --- Reconstruction space, phantom and data --- #

# Transform from 'ij' storage to 'xy'
image = np.rot90(imageio.imread('affine_phantom.png'), k=-1)

reco_space = odl.uniform_discr([-10, -10], [10, 10], image.shape,
                               dtype='float32')
phantom = reco_space.element(image)
data = reco_space.element(np.load('affine_data.npy'))


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
reg_param1 = 4e0
regularizer1 = reg_param1 * odl.solvers.L1Norm(gradient.range)
reg_param2 = 1e0
regularizer2 = reg_param2 * odl.solvers.L1Norm(eps.range)
box_constr = odl.solvers.IndicatorBox(reco_space, 0, 255)

g = [data_matching, regularizer1, regularizer2, box_constr]

# Don't use f
f = odl.solvers.ZeroFunctional(domain)

# Create callback that prints the iteration number and shows partial results
callback_fig = None


def show_first(x):
    global callback_fig
    callback_fig = x[0].show('iterate', clim=[0, 255], fig=callback_fig)


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


# Display images
# phantom.show(title='Phantom')
# data.show(title='Data')
x[0].show(title='TGV-2 Reconstruction', clim=[0, 255])
# Display horizontal profile through the "tip"
fig = phantom.show(coords=[None, -4.25])
x[0].show(coords=[None, -4.25], fig=fig, force_show=True)
