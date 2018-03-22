"""Bimodal tomography with TV regularizer (bad data only)."""

import imageio
import numpy as np
import os
import simplejson as json

import odl
from odl.contrib import fom
from odl.util import run_from_ipython
from variable_lp.util import (
    log_sampler, const_sampler, run_many_examples, plot_foms_1, plot_foms_2)

# --- Experiment configuration --- #

# Number of iterations of the optimizer, 150 should be enough
num_iter = 150


def run_example(geometry, reg_param):
    """Run the example for given parameters, producing some result files."""
    reg_param_1, reg_param_2 = reg_param

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
    with odl.util.NumpyRandomSeed(123):
        # Create even if unused, so as to have the same random state when
        # creating `bad_data` (as in other examples)
        good_data = (data +
                     0.01 * np.max(data) * odl.phantom.white_noise(data.space))
        bad_data = (data +
                    0.1 * np.max(data) * odl.phantom.white_noise(data.space))

    # --- Set up the inverse problem for the bad data --- #

    # Initialize gradient and 2nd order derivative operator
    grad = odl.Gradient(reco_space, pad_mode='order1')
    eps = odl.DiagonalOperator(grad, reco_space.ndim)
    domain = odl.ProductSpace(grad.domain, eps.domain)

    # Assemble operators and functionals for the solver

    # The linear operators are
    # 1. ray transform on the first component for the data matching
    # 2. gradient of component 1 - component 2 for the auxiliary functional
    # 3. eps on the second component
    # 4. projection onto the first component

    L = [
        ray_trafo * odl.ComponentProjection(domain, 0),
        odl.ReductionOperator(grad, odl.ScalingOperator(grad.range, -1)),
        eps * odl.ComponentProjection(domain, 1),
        odl.ComponentProjection(domain, 0),
    ]

    # The functionals are
    # 1. L2 data matching
    # 2. regularization parameter 1 times L1 norm on the range of the gradient
    # 3. regularization parameter 2 times L1 norm on the range of eps
    # 4. box indicator on the reconstruction space
    data_matching = odl.solvers.L2NormSquared(ray_trafo.range)
    data_matching = data_matching.translated(bad_data)
    regularizer1 = reg_param_1 * odl.solvers.L1Norm(grad.range)
    regularizer2 = reg_param_2 * odl.solvers.L1Norm(eps.range)
    box_constr = odl.solvers.IndicatorBox(reco_space, 0, 1)

    g = [data_matching, regularizer1, regularizer2, box_constr]

    # Don't use f
    f = odl.solvers.ZeroFunctional(domain)

    # Display iteration counter
    callback = odl.solvers.CallbackPrintIteration(step=10)

    # Compute sigma parameters for the Douglas-Rachford solver, using a custom
    # choice for tau and the norms of the operators in L
    L_norms = [
        1.2 * odl.power_method_opnorm(L[0], maxiter=20),
        1.2 * odl.power_method_opnorm(L[1]),
        1.2 * odl.power_method_opnorm(L[2]),
    ]

    tau = 0.1
    tau, sigma = odl.solvers.douglas_rachford_pd_stepsize(L_norms, tau)
    sigma = list(sigma)
    sigma.append(1.0)

    # Solve with initial guess x = 0
    x = domain.zero()
    odl.solvers.douglas_rachford_pd(x, f, g, L, tau=tau, sigma=sigma, lam=1.5,
                                    niter=300, callback=callback)

    # --- Compute FOMs --- #

    foms = {}
    foms['psnr'] = fom.psnr(x[0], phantom)
    foms['ssim'] = fom.ssim(x[0], phantom)
    foms['hpsi'] = fom.haarpsi(x[0], phantom)

    return foms


# %% Global parameters

# Set random seed for reproducibility
np.random.seed(123)
foms = ['psnr', 'ssim', 'hpsi']
# Geometry: 'cone2d', 'parallel2d', 'parallel2d_lim_ang'
geometry = 'parallel2d'

# Make overall structure
if run_from_ipython():
    here = ''  # current working directory
else:
    try:
        here = os.path.abspath(__file__)
    except NameError:
        here = ''

results_root = os.path.join(here, 'results', 'bimodal_tomo', geometry, 'tgv')
os.makedirs(results_root, exist_ok=True)

# %% Find optimal reg_param: cycle 1 - coarse sampling

# Sample the regularization parameter (log-uniformly)
num_samples = 50
min_val = [1e-6, 1e-6]
max_val = [1e4, 1e4]
reg_param_sampler = log_sampler(min_val, max_val)

arg_sampler = zip(const_sampler(geometry), reg_param_sampler)

results_cycle_1 = run_many_examples(run_example, arg_sampler, num_samples)
xys_cycle_1 = np.array([res[0][1] for res in results_cycle_1])
fom_values = [res[1] for res in results_cycle_1]

# Record meta info
meta_cycle_1 = {
    'cycle': 1,
    'sampling': 'log',
    'min_val': min_val,
    'max_val': max_val,
    'num_samples': num_samples,
    'columns': ['reg_param_1', 'reg_param_2'] + foms,
}

# Rearrange data for plotting and saving
fom_arrays_cycle_1 = [np.array([res[fom] for res in fom_values])
                      for fom in foms]
fom_array = np.hstack([arr[:, None] for arr in fom_arrays_cycle_1])
plot_foms_2(xys_cycle_1, fom_array, meta=meta_cycle_1)

# Save data
data_cycle_1 = np.hstack([xys_cycle_1, fom_array])
with open(os.path.join(results_root, 'meta_cycle_1.json'), 'w') as fp:
    json.dump(meta_cycle_1, fp, indent='    ')

np.save(os.path.join(results_root, 'data_cycle_1'), data_cycle_1)

# %% Find optimal reg_param: cycle 2 - fine sampling

# Sample the regularization parameter (log-uniformly)
num_samples = 20
rand_state_cycle_2 = np.random.get_state()  # for multiple runs

if geometry == 'parallel2d':
    min_val = 5e-4
    max_val = 1e-1
elif geometry == 'parallel2d_lim_ang':
    min_val = 1e-4
    max_val = 1e-1
else:
    min_val = 5e-4
    max_val = 1e-1

reg_param_sampler = log_sampler(min_val, max_val)

arg_sampler = zip(const_sampler(geometry), reg_param_sampler)

results_cycle_2 = run_many_examples(run_example, arg_sampler, num_samples)
xs_cycle_2 = [res[0][1] for res in results_cycle_2]
fom_values = [res[1] for res in results_cycle_2]
plot_foms_1(xs_cycle_2, fom_values, foms, x_label='Regularization parameter')

# Record meta info
meta_cycle_2 = {
    'cycle': 2,
    'sampling': 'log',
    'min_val': min_val,
    'max_val': max_val,
    'num_samples': num_samples,
    'columns': ['x'] + foms,
}

# Rearrange data for saving
fom_arrays_cycle_2 = [np.array([res[fom] for res in fom_values])
                      for fom in foms]
data_cycle_2 = np.empty((num_samples, 1 + len(foms)), dtype=float)
data_cycle_2[:, 0] = xs_cycle_2
for i, arr in enumerate(fom_arrays_cycle_2):
    data_cycle_2[:, i + 1] = arr

# Save data
with open(os.path.join(results_root, 'meta_cycle_2.json'), 'w') as fp:
    json.dump(meta_cycle_2, fp, indent='    ')

np.save(os.path.join(results_root, 'data_cycle_2'), data_cycle_2)

# %% Find optimal reg_param: cycle 3 - final grid search

# Sample the regularization parameter (uniformly)
rand_state_cycle_3 = np.random.get_state()  # for multiple runs

if geometry == 'parallel2d':
    num_samples = 16
    min_val = 5e-4
    max_val = 1.45e-2
elif geometry == 'parallel2d_lim_ang':
    num_samples = 16
    min_val = 5e-4
    max_val = 1.45e-2
else:
    num_samples = 31
    min_val = 2e-3
    max_val = 3.2e-2

reg_param_sampler = iter(np.linspace(min_val, max_val, num_samples))

arg_sampler = zip(const_sampler(geometry), reg_param_sampler)

results_cycle_3 = run_many_examples(run_example, arg_sampler, num_samples)
xs_cycle_3 = [res[0][1] for res in results_cycle_3]
fom_values = [res[1] for res in results_cycle_3]
plot_foms_1(xs_cycle_3, fom_values, foms, x_label='Regularization parameter',
            log_x=False)

# Record meta info
meta_cycle_3 = {
    'cycle': 3,
    'sampling': 'grid',
    'min_val': min_val,
    'max_val': max_val,
    'num_samples': num_samples,
    'columns': ['x'] + foms,
}

# Rearrange data for saving
fom_arrays_cycle_3 = [np.array([res[fom] for res in fom_values])
                      for fom in foms]
data_cycle_3 = np.empty((num_samples, 1 + len(foms)), dtype=float)
data_cycle_3[:, 0] = xs_cycle_3
for i, arr in enumerate(fom_arrays_cycle_3):
    data_cycle_3[:, i + 1] = arr

# Save data
with open(os.path.join(results_root, 'meta_cycle_3.json'), 'w') as fp:
    json.dump(meta_cycle_3, fp, indent='    ')

np.save(os.path.join(results_root, 'data_cycle_3'), data_cycle_3)

# %% For the records

if geometry == 'parallel2d':
    optimal_param = 4e-3
elif geometry == 'parallel2d_lim_ang':
    optimal_param = 3e-3
else:
    optimal_param = 6e-3
