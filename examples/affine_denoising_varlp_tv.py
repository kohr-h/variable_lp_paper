"""Denoising with variable Lp TV regularizer."""

import numpy as np
import odl
import scipy
import variable_lp


# --- Reconstruction space, phantom and data --- #

# Transform from 'ij' storage to 'xy'
image = np.rot90(scipy.misc.imread('affine_phantom.png'), k=-1)

reco_space = odl.uniform_discr([-10, -10], [10, 10], image.shape,
                               dtype='float32')
phantom = reco_space.element(image)
data = reco_space.element(np.load('affine_data.npy'))


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
pre_kernel = reco_space.element(exp_kernel, s=0.05)
pre_kernel_ft = fourier(pre_kernel) * (2 * np.pi)
pre_conv = fourier.inverse * pre_kernel_ft * fourier
smoothed_lapl = odl.Laplacian(reco_space, pad_mode='symmetric') * pre_conv
# Smoothed Laplacian of the data
abs_lapl = np.abs(smoothed_lapl(data))
# Remove jumps at the boundary, they're artificial
avg = np.mean(abs_lapl)
abs_lapl[:5, :] = avg
abs_lapl[-5:, :] = avg
abs_lapl[:, :5] = avg
abs_lapl[:, -5:] = avg
# Post-smoothing
post_kernel = reco_space.element(exp_kernel, s=0.4)
post_kernel_ft = fourier(post_kernel) * (2 * np.pi)
post_conv = fourier.inverse * post_kernel_ft * fourier
conv_abs_lapl = np.maximum(post_conv(abs_lapl), 0)
conv_abs_lapl -= np.min(conv_abs_lapl)
conv_abs_lapl *= 2 / np.max(conv_abs_lapl)
conv_abs_lapl[:] = np.minimum(conv_abs_lapl, 1)
exponent = 2.0 - conv_abs_lapl
exponent.show(force_show=True)


# Disturb the exponent function
# var_exponent_arr[:, 100:150] = 2.0


# --- Set up the inverse problem --- #


# Assemble operators and functionals for the solver
gradient = odl.Gradient(reco_space, pad_mode='order1')
lin_ops = [odl.IdentityOperator(reco_space), gradient]
data_matching = odl.solvers.L2NormSquared(reco_space).translated(data)
varlp_func = variable_lp.VariableLpModular(gradient.range, exponent,
                                           impl='cython')
# Left-multiplication version
reg_param = 2e0
regularizer = reg_param * varlp_func
# Right-multiplication version
# reg_param = 8e-2
# regularizer = varlp_func * reg_param

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
# phantom.show(title='Phantom')
# data.show(title='Data')
x.show(title='TV-p Reconstruction')
exponent.show('Exponent function')
# Display horizontal profile through the "tip"
fig = phantom.show(coords=[None, -4.25])
x.show(coords=[None, -4.25], fig=fig, force_show=True)
