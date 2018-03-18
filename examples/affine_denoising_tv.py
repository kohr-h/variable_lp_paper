"""Reference TV denoising of the affine example."""

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


# Assemble operators and functionals for the solver
gradient = odl.Gradient(reco_space)
lin_ops = [odl.IdentityOperator(reco_space), gradient]
data_matching = odl.solvers.L2NormSquared(reco_space).translated(data)
reg_param = 6e0
regularizer = reg_param * odl.solvers.L1Norm(gradient.range)
g = [data_matching, regularizer]
f = odl.solvers.IndicatorBox(reco_space, 0, 255)

# Uncomment the combined callback to also display iterates
callback = (odl.solvers.CallbackShow('iterate', step=20, clim=[0, 255]) &
            odl.solvers.CallbackPrintIteration())
# callback = odl.solvers.CallbackPrintIteration()

# Solve with initial guess x = data.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = data.copy()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.1, sigma=[0.1, 0.02], lam=1.5,
                                niter=100, callback=callback)


# Display images
# phantom.show(title='Phantom', clim=[-5, 5])
# data.show(title='Data')
x.show(title='TV-1 Reconstruction', clim=[0, 255])
# Display horizontal profile through the "tip"
fig = phantom.show(coords=[None, -4.25])
x.show(coords=[None, -4.25], fig=fig, force_show=True)
