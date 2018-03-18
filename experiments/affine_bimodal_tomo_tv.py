"""Bimodal tomography with TV regularizer (bad data only)."""

import imageio
import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment

import odl
from odl.contrib import fom

# Experiment configuration
ex = Experiment('Bimodal tomography with TV regularizer')


@ex.config
def config():
    # Geometry: 'cone2d', 'parallel2d', 'parallel2d_lim_ang'
    geometry = 'parallel2d_lim_ang'
    # Regularization parameter, around 1e-2 is reasonable
    reg_param = 5e-3
    # Number of iterations of the optimizer, 150 should be enough
    num_iter = 150


@ex.automain
def run(geometry, reg_param, num_iter):

    file_prefix = 'affine_bimodal_tomo_{}_tv'.format(geometry)

    # --- Reconstruction space and phantom --- #

    # Read image and transform from 'ij' storage to 'xy'
    image = np.rot90(imageio.imread('affine_phantom.png'), k=-1)

    reco_space = odl.uniform_discr([-1, -1], [1, 1], image.shape,
                                   dtype='float32')
    phantom = reco_space.element(image) / np.max(image)

    # --- Set up the forward operator --- #

    # Make a 2D geometry with flat detector
    if geometry == 'cone2d':
        angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
        detector_partition = odl.uniform_partition(-4, 4, 400)
        geometry = odl.tomo.FanFlatGeometry(
            angle_partition, detector_partition, src_radius=40, det_radius=40)

    elif geometry == 'parallel2d':
        angle_partition = odl.uniform_partition(0, np.pi, 180)
        detector_partition = odl.uniform_partition(-2, 2, 200)
        geometry = odl.tomo.Parallel2dGeometry(
            angle_partition, detector_partition)

    elif geometry == 'parallel2d_lim_ang':
        angle_partition = odl.uniform_partition(0, 3 * np.pi / 4, 135)
        detector_partition = odl.uniform_partition(-2, 2, 400)
        geometry = odl.tomo.Parallel2dGeometry(
            angle_partition, detector_partition)

    else:
        raise ValueError('geometry {!r} not understood'.format(geometry))

    # Ray transform (= forward projection).
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cpu')

    # Generate data with predictable randomness to make them reproducible
    data = ray_trafo(phantom)
    good_data = (data +
                 0.01 * np.max(data) * odl.phantom.white_noise(data.space))
    bad_data = (data +
                0.1 * np.max(data) * odl.phantom.white_noise(data.space))

    # --- Set up the inverse problem for the bad data --- #

    # Gradient for the TV term
    grad = odl.Gradient(reco_space, pad_mode='order1')

    # Balance operator norms and rescale data
    ray_trafo_norm = 1.2 * odl.power_method_opnorm(ray_trafo, maxiter=20)

    # Create operators and functionals for the solver
    L = [ray_trafo, grad]
    data_matching = odl.solvers.L2NormSquared(ray_trafo.range)
    data_matching = data_matching.translated(bad_data)
    l1_func = odl.solvers.L1Norm(grad.range)
    regularizer = reg_param * l1_func

    g = [data_matching, regularizer]
    f = odl.solvers.IndicatorBox(reco_space, 0, 1)

    # Show iteration counter
    callback = odl.solvers.CallbackPrintIteration(step=10)

    # Use default tau and sigma parameters for the Douglas-Rachford solver
    tau, sigma = odl.solvers.douglas_rachford_pd_stepsize(
        [ray_trafo_norm, grad])

    # Solve with initial guess x = 0
    x = reco_space.zero()
    odl.solvers.douglas_rachford_pd(x, f, g, L, tau=tau, sigma=sigma, lam=1.5,
                                    niter=num_iter, callback=callback)

    np.save(file_prefix + '_reco', x)
    ex.add_artifact(file_prefix + '_reco.npy')

    # --- Compute FOMs --- #

    with open(file_prefix + '_fom.txt', 'w+') as f:
        psnr = fom.psnr(x, phantom)
        print('PSNR:', psnr, file=f)
        ssim = fom.ssim(x, phantom)
        print('SSIM:', ssim, file=f)
        haarpsi = fom.haarpsi(x, phantom)
        print('HaarPSI:', haarpsi, file=f)

    ex.add_artifact(file_prefix + '_fom.txt')

    # --- Display images --- #

    # Create horizontal profile through the "tip"
    phantom_slice = phantom[:, 35]
    reco_slice = x[:, 35]
    x_vals = reco_space.grid.coord_vectors[0]
    plt.figure()
    axes = plt.gca()
    axes.set_ylim([0.3, 1.1])
    plt.plot(x_vals, phantom_slice, label='Phantom')
    plt.plot(x_vals, reco_slice, label='TV reconstruction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_prefix + '_profile.png')
    ex.add_artifact(file_prefix + '_profile.png')

    # Display full image
    plt.figure()
    plt.imshow(np.rot90(x), cmap='bone', clim=[0, 1])
    axes = plt.gca()
    axes.axis('off')
    plt.tight_layout()
    plt.savefig(file_prefix + '_reco.png')
    ex.add_artifact(file_prefix + '_reco.png')
