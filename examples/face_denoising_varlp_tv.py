"""Denoising with variable Lp TV regularizer."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from PIL import Image

import odl
import variable_lp

# --- Reconstruction space, phantom and data --- #

# Convert "face" image to greyscale
image = Image.fromarray(scipy.misc.face()).convert('L')
# Transform from 'ij' storage to 'xy'
image = np.rot90(image, k=-1) / np.max(image)

reco_space = odl.uniform_discr([-10, -7.5], [10, 7.5], image.shape,
                               dtype='float32')
phantom = reco_space.element(image)
data = reco_space.element(np.load('face.npy'))


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
post_kernel = reco_space.element(exp_kernel, s=0.1)
post_kernel_ft = fourier(post_kernel) * (2 * np.pi)
post_conv = fourier.inverse * post_kernel_ft * fourier
conv_abs_lapl = np.maximum(post_conv(abs_lapl), 0)
conv_abs_lapl -= np.min(conv_abs_lapl)
conv_abs_lapl *= 2 / np.max(conv_abs_lapl)
conv_abs_lapl[:] = np.minimum(conv_abs_lapl, 1)
exponent = 2.0 - conv_abs_lapl
exponent.show(force_show=True)


# --- Set up the inverse problem --- #


# Assemble operators and functionals for the solver
gradient = odl.Gradient(reco_space, pad_mode='order1')
lin_ops = [odl.IdentityOperator(reco_space), gradient]
data_matching = odl.solvers.L2NormSquared(reco_space).translated(data)
varlp_func = variable_lp.VariableLpModular(gradient.range, exponent,
                                           impl='numba_parallel')
reg_param = 4.5e-3
regularizer = reg_param * varlp_func

g = [data_matching, regularizer]
f = odl.solvers.IndicatorBox(reco_space, 0, 1.0)

# Uncomment the combined callback to also display iterates
callback = (odl.solvers.CallbackShow('iterate', step=10, clim=[0, 1.0]) &
            odl.solvers.CallbackPrintIteration())
# callback = odl.solvers.CallbackPrintIteration()

# Solve with initial guess x = data.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = data.copy()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.1, sigma=[0.2, 0.001], lam=1.5,
                                niter=30, callback=callback)


# --- Display images --- #


# phantom.show(title='Phantom', clim=[-5, 5])
# data.show(title='Data')
x.show(title='TV-p Reconstruction', clim=[0, 1.0])
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
plt.plot(x_vals, reco_slice, label='TV-p reconstruction')
plt.legend()
plt.savefig('face_denoising_varlp_tv_profile.png')


# Full images
plt.figure()
plt.imshow(np.rot90(x), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.savefig('face_denoising_varlp_tv_reco.png')

plt.figure()
plt.imshow(np.rot90(exponent), cmap='bone', clim=[1, 2])
axes = plt.gca()
axes.axis('off')
plt.savefig('face_denoising_varlp_tv_exp.png')

plt.figure()
plt.imshow(np.rot90(phantom), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.savefig('face_denoising_phantom.png')

plt.figure()
plt.imshow(np.rot90(data), cmap='bone')
axes = plt.gca()
axes.axis('off')
plt.savefig('face_denoising_data.png')

# Detail images
plt.figure()
detail = phantom[500:800, 250:550]
plt.imshow(np.rot90(detail), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.savefig('face_denoising_phantom_detail.png')

plt.figure()
detail = x[500:800, 250:550]
plt.imshow(np.rot90(detail), cmap='bone', clim=[0, 1])
axes = plt.gca()
axes.axis('off')
plt.savefig('face_denoising_varlp_tv_reco_detail.png')

plt.figure()
detail = exponent[500:800, 250:550]
plt.imshow(np.rot90(detail), cmap='bone', clim=[1, 2])
axes = plt.gca()
axes.axis('off')
plt.savefig('face_denoising_varlp_tv_exp_detail.png')
