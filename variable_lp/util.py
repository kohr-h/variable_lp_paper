# Copyright 2017, 2018 Holger Kohr
#
# This file is part of variable_lp_paper.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Some utility functions."""

from __future__ import absolute_import, division, print_function

import numpy as np

__all__ = ('log_sampler', 'const_sampler', 'run_many_examples',
           'plot_foms_1')


def log_sampler(minval, maxval):
    r"""Generator that yields log-uniform samples in ``[minval, maxval]``.

    The samples are computed by drawing uniform exponents

    .. math::
        X \sim U(\log_{10} a, \log_{10} b)

    with :math:`a =` ``minval``, :math:`b =` ``maxval``, and then
    transforming the result as

    .. math::
        Y = 10^{X}.

    Parameters
    ----------
    minval, maxval : positive float or array-like
        Boundaries of the (multi-dimensional) interval in which the samlples
        should be log-uniform. This requires ``minval < maxval``.

    Yields
    ------
    sample : positive float or `numpy.ndarray`
        A sample drawn from the above specified distribution.
    """
    minval, minval_in = (np.array(minval, dtype=float, copy=True, ndmin=1),
                         minval)
    maxval, maxval_in = (np.array(maxval, dtype=float, copy=True, ndmin=1),
                         maxval)

    if minval.shape != maxval.shape:
        raise ValueError('shapes of `minval` and `maxval` must be equal, '
                         'but {} != {}'.format(minval.shape, maxval.shape))

    if np.any(minval <= 0):
        raise ValueError('`minval` must be positive, got {}'
                         ''.format(minval_in))

    if np.any(maxval <= 0):
        raise ValueError('`maxval` must be positive, got {}'
                         ''.format(maxval_in))

    if not np.all(minval < maxval):
        raise ValueError('`minval` must be smaller than `maxval`, but '
                         '{} !< {}'.format(minval_in, maxval_in))

    while True:
        exp10 = np.random.uniform(np.log10(minval), np.log10(maxval))
        yield np.exp(exp10 * np.log(10))


def const_sampler(value):
    """Generator that simply yields a constant value."""
    while True:
        yield value


def run_many_examples(example, arg_sampler, num_samples):
    """Run examples with sampled parameters and collect the results."""
    results = []
    for run in range(num_samples):
        print('========  Run {}  ========'.format(run))
        args = next(arg_sampler)
        print('args:', args)
        result = example(*args)
        print('Result:', result)
        results.append((args, result))
    return tuple(results)


def plot_foms_1(xs, results, foms, x_label='x', log_x=True):
    """Make scatter plots for all FOMs, potentially with log x scale.

    Parameters
    ----------
    xs : array-like
        x values in the plots, shared among all plots.
    results : dict
        Dictionary containing the results to plot. Each value should be
        an array of the same length as ``xs``.
    foms : sequence of str
        Names of the FOMs to be plotted. They must be the keys of the
        ``results`` dictionary.
    x_label : str, optional
        Label for the x axis of the plot.
    log_x : bool, optional
        If ``True``, use a log scale for the x axis.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure to be shown.
    """
    import matplotlib.pyplot as plt

    fom_vals = [[res[fom] for res in results] for fom in foms]
    fig, axs = plt.subplots(nrows=len(foms), sharex=True)
    for ax, ys, fom_name in zip(axs[:-1], fom_vals[:-1], foms[:-1]):
        ax.scatter(xs, ys)
        ax.set_ylabel(fom_name.upper())

    ax = axs[-1]
    ys = fom_vals[-1]
    fom_name = foms[-1]
    ax.scatter(xs, ys)
    ax.set_ylabel(fom_name.upper())

    min_x = np.min(xs)
    max_x = np.max(xs)
    if log_x:
        ax.set_xscale('log')
        ax.set_xlim(10 ** int(np.floor(np.log10(min_x))),
                    10 ** int(np.floor(np.log10(max_x))))
    else:
        ax.set_xlim(min_x, max_x)

    ax.set_xlabel(x_label)
    fig.canvas.set_window_title('FOMs')
    fig.suptitle('Figures of Merit: {}'
                 ''.format(', '.join(fom.upper() for fom in foms)))

    return fig
