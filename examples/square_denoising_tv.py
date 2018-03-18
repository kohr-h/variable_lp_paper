"""Reference TV denoising of the square example."""

import matplotlib.pyplot as plt
import numpy as np

import odl

# --- Reconstruction space, phantom and data --- #

reco_space = odl.uniform_discr([-10, -10], [10, 10], (300, 300),
                               dtype='float32')


def square(x):
    return x[0] * ((np.abs(x[0]) <= 5) & (np.abs(x[1]) <= 5))


phantom = reco_space.element(square)
data = reco_space.element(np.load('square.npy'))


# --- Set up the inverse problem --- #


# Assemble operators and functionals for the solver
gradient = odl.Gradient(reco_space)
lin_ops = [odl.IdentityOperator(reco_space), gradient]
data_matching = odl.solvers.L2NormSquared(reco_space).translated(data)
reg_param = 2e-1
regularizer = reg_param * odl.solvers.L1Norm(gradient.range)
g = [data_matching, regularizer]
f = odl.solvers.IndicatorBox(reco_space, -5, 5)

# Uncomment the combined callback to also display iterates
# callback = (odl.solvers.CallbackShow('iterate', step=20, clim=[-5, 5]) &
#             odl.solvers.CallbackPrintIteration())
callback = odl.solvers.CallbackPrintIteration()

# Solve with initial guess x = data.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = data.copy()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.1, sigma=[0.1, 0.02], lam=1.5,
                                niter=100, callback=callback)


# --- Display images --- #


# phantom.show(title='Phantom', clim=[-5, 5])
# data.show(title='Data')
x.show(title='TV Reconstruction', clim=[-5, 5])
# Display horizontal profile
# fig = phantom.show(coords=[None, 0], label='')
# x.show(coords=[None, 0], fig=fig, force_show=True)

# Create horizontal profile plot at the middle
phantom_slice = phantom[:, 150]
reco_slice = x[:, 150]
x_vals = reco_space.grid.coord_vectors[0]
plt.figure()
axes = plt.gca()
axes.set_ylim([-5.1, 5.1])
plt.plot(x_vals, phantom_slice, label='Phantom')
plt.plot(x_vals, reco_slice, label='TV reconstruction')
plt.legend()
plt.tight_layout()
plt.savefig('square_denoising_tv_profile.png')


# Display full image
plt.figure()
plt.imshow(np.rot90(x), cmap='bone', clim=[-5, 5])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('square_denoising_tv_reco.png')
