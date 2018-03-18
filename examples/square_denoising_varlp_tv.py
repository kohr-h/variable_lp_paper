"""Denoising with variable Lp TV regularizer."""

import matplotlib.pyplot as plt
import numpy as np

import odl
import variable_lp

# --- Reconstruction space, phantom and data --- #

reco_space = odl.uniform_discr([-10, -10], [10, 10], (300, 300),
                               dtype='float32')


def square(x):
    return x[0] * ((np.abs(x[0]) <= 5) & (np.abs(x[1]) <= 5))


phantom = reco_space.element(square)
data = reco_space.element(np.load('square.npy'))


# --- Compute the exponent --- #


# We use the following procedure to generate the exponent from the data:
# - Compute a moderately smoothed version of the Laplacian L(g)
# - Take its absolute value and smooth it more aggressively
# - Multiply by 2 / max(L(g)), then clip at value 1.
#   This is to make the regions with high values broader.
# - Use 2 minus the result as exponent
def exp_kernel(x, **kwargs):
    s = kwargs.pop('s', 0.5)
    scaled = [xi / (np.sqrt(2) * s) for xi in x]
    return np.exp(-sum(xi ** 2 for xi in scaled))


# Pre-smoothing convolution
fourier = odl.trafos.FourierTransform(reco_space)
pre_kernel = reco_space.element(exp_kernel, s=0.2)
pre_kernel_ft = fourier(pre_kernel) * (2 * np.pi)
pre_conv = fourier.inverse * pre_kernel_ft * fourier
smoothed_lapl = odl.Laplacian(reco_space, pad_mode='symmetric') * pre_conv
# Smoothed Laplacian of the data
abs_lapl = np.abs(smoothed_lapl(data))
# Post-smoothing
post_kernel = reco_space.element(exp_kernel, s=0.2)
post_kernel_ft = fourier(post_kernel) * (2 * np.pi)
post_conv = fourier.inverse * post_kernel_ft * fourier
conv_abs_lapl = np.maximum(post_conv(abs_lapl), 0)
conv_abs_lapl -= np.min(conv_abs_lapl)
conv_abs_lapl *= 2 / np.max(conv_abs_lapl)
conv_abs_lapl[:] = np.minimum(conv_abs_lapl, 1)
exponent = 2.0 - conv_abs_lapl


# Artifically set the values inside the square to new values
# exponent[100:200, 100:150] = 1.05
# exponent[100:200, 150:200] = 2.0

# --- Set up the inverse problem --- #


# Assemble operators and functionals for the solver
gradient = odl.Gradient(reco_space)
lin_ops = [odl.IdentityOperator(reco_space), gradient]
data_matching = odl.solvers.L2NormSquared(reco_space).translated(data)
varlp_func = variable_lp.VariableLpModular(gradient.range, exponent,
                                           impl='cython')
# Left-multiplication version
reg_param = 2e-1
regularizer = reg_param * varlp_func
# Right-multiplication version
# reg_param = 8e-2
# regularizer = varlp_func * reg_param

g = [data_matching, regularizer]
f = odl.solvers.IndicatorBox(reco_space, -5, 5)

# Uncomment the combined callback to also display iterates
callback = (odl.solvers.CallbackShow('iterate', step=20, clim=[-5, 5]) &
            odl.solvers.CallbackPrintIteration())
# callback = odl.solvers.CallbackPrintIteration()

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
x.show(title='TV-p Reconstruction', clim=[-5, 5])
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
plt.plot(x_vals, reco_slice, label='TV-p reconstruction')
plt.legend()
plt.tight_layout()
plt.savefig('square_denoising_varlp_tv_profile.png')


# Display full images
plt.figure()
plt.imshow(np.rot90(x), cmap='bone', clim=[-5, 5])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('square_denoising_varlp_tv_reco.png')

plt.figure()
plt.imshow(np.rot90(exponent), cmap='bone', clim=[1, 2])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('square_denoising_varlp_tv_exp.png')

plt.figure()
plt.imshow(np.rot90(phantom), cmap='bone', clim=[-5, 5])
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('square_denoising_phantom.png')

plt.figure()
plt.imshow(np.rot90(data), cmap='bone')
axes = plt.gca()
axes.axis('off')
plt.tight_layout()
plt.savefig('square_denoising_data.png')
