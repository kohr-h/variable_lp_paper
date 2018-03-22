# Copyright 2017, 2018 Holger Kohr
#
# This file is part of variable_lp_paper.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Some utility functions."""

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import simplejson as json

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


def plot_foms_1(xs, foms, meta=None, log_x=None):
    """Make scatter plots for all FOMs, potentially with log x scale.

    Parameters
    ----------
    xs : array-like
        x values in the plots, shared among all plots.
    foms : array-like
        Array containing the results to plot. Each column should be
        an array of the same length as ``xs``.
    meta : dict, optional
        Metadata for extra information on how to show the data. Entries
        used for plotting (all optional) are

        - ``'cycle'``: ``str``; Used in the title; Default: not used
        - ``'columns'``: sequence of str; Used for axis labeling;
          Default: ``'x', 'y_1', 'y_2', ...``
        - ``'min_val', 'max_val'``: float; Used for x axis limits;
          Default: data limits.
        - ``'sampling'``: str; If ``'log'``, a log scale is used for the
          x axis (if not overridden by the ``log_x`` parameter);
          Default: linear scale.

    log_x : bool, optional
        If ``True``, use a log scale for the x axis.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure to be shown.
    """
    import matplotlib.pyplot as plt

    xs = np.array(xs, copy=False, ndmin=1)
    foms = np.array(foms, copy=False, ndmin=2)
    assert xs.ndim == 1, 'xs must be 1-dimensional'
    assert foms.ndim == 2, 'forms must be 2-dimensional'
    assert len(xs) == len(foms), 'len(xs) must match len(foms)'

    if meta is None:
        cycle = columns = min_val = max_val = sampling = None
    else:
        cycle = meta.get('cycle', None)
        columns = meta.get('columns', None)
        min_val = meta.get('min_val', None)
        max_val = meta.get('max_val', None)
        sampling = meta.get('sampling', None)

    fig, axs = plt.subplots(nrows=foms.shape[1], sharex=True)
    if columns is not None:
        assert len(columns) == 1 + foms.shape[1], 'wrong number of columns'
    for i, ax in enumerate(axs[:-1]):
        ax.scatter(xs, foms[:, i])
        if columns is None:
            ax.set_ylabel('y_{}'.format(i + 1))
        else:
            ax.set_ylabel(columns[i + 1].upper())

    ax = axs[-1]
    ax.scatter(xs, foms[:, -1])
    if columns is None:
        ax.set_xlabel('x')
        ax.set_ylabel('y_{}'.format(foms.shape[1]))
    else:
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[-1].upper())

    if min_val is None:
        min_x = np.min(xs)
    else:
        min_x = float(min_val)

    if max_val is None:
        max_x = np.max(xs)
    else:
        max_x = float(max_val)

    if ((log_x is None and sampling is not None and sampling == 'log') or
            log_x is not None and log_x):
        ax.set_xscale('log')
        ax.set_xlim(10 ** int(np.floor(np.log10(min_x))),
                    10 ** int(np.ceil(np.log10(max_x))))
    else:
        ax.set_xlim(min_x, max_x)

    if cycle is None:
        fig.canvas.set_window_title('FOMs')
        if columns is None:
            fig.suptitle('Figures of Merit')
        else:
            fig.suptitle(
                'Figures of Merit: {}'
                ''.format(', '.join(fom.upper() for fom in columns[1:])))
    else:
        fig.canvas.set_window_title('FOMs cycle {}'.format(cycle))
        if columns is None:
            fig.suptitle(
                'Figures of Merit (cycle {})'
                ''.format(', '.join(fom.upper() for fom in foms)))
        else:
            fig.suptitle(
                'Figures of Merit (cycle {}): {}'
                ''.format(cycle,
                          ', '.join(fom.upper() for fom in columns[1:])))

    return fig


def plot_foms_2(xys, foms, meta=None, log_x=None, log_y=None):
    """Make scatter plots for all FOMs, potentially with log x scale.

    Parameters
    ----------
    xys : array-like
        x-y value pairs in the plots, shared among all plots.
    foms : array-like
        Array containing the results to plot. Each column should be
        an array of the same length as ``xys``.
    meta : dict, optional
        Metadata for extra information on how to show the data. Entries
        used for plotting (all optional) are

        - ``'cycle'``: ``str``; Used in the title; Default: not used
        - ``'columns'``: sequence of str; Used for axis labeling;
          Default: ``'x', 'y', 'z_1', 'z_2', ...``
        - ``'min_val', 'max_val'``: 2-tuple of float (each); Used for
          x and y axis limits; Default: data limits.
        - ``'sampling'``: str; If ``'log'``, a log scale is used for the
          x and y axes (if not overridden by the ``log_x`` or ``log_y``
          parameters); Default: linear scale.

    log_x, log_y : bool, optional
        If ``True``, use a log scale for the x/y axis.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure to be shown.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xys = np.array(xys, copy=False, ndmin=2)
    foms = np.array(foms, copy=False, ndmin=2)
    assert xys.ndim == 2, 'xys must be 2-dimensional'
    assert xys.shape[1] == 2, 'xys must have 2 columns'
    assert foms.ndim == 2, 'forms must be 2-dimensional'
    assert len(xys) == len(foms), 'len(xys) must match len(foms)'

    if meta is None:
        cycle = columns = min_val = max_val = sampling = None
    else:
        cycle = meta.get('cycle', None)
        columns = meta.get('columns', None)
        min_val = meta.get('min_val', None)
        max_val = meta.get('max_val', None)
        sampling = meta.get('sampling', None)

    if min_val is None:
        min_xy = np.min(xys, axis=0)
    else:
        min_xy = np.array(min_val).reshape([2])

    if max_val is None:
        max_xy = np.max(xys, axis=0)
    else:
        max_xy = np.array(max_val).reshape([2])

    min_exp = np.floor(np.log10(min_xy)).astype(int)
    max_exp = np.ceil(np.log10(max_xy)).astype(int)

    if log_x is None and sampling is not None and sampling == 'log':
        log_x = True
    if log_y is None and sampling is not None and sampling == 'log':
        log_y = True

    if log_x:
        xs = np.log10(xys[:, 0])
        xticklabels = ['1e{:+}'.format(e)
                       for e in range(min_exp[0], max_exp[0])]
        min_x = np.log10(min_xy[0])
        max_x = np.log10(max_xy[0])
    else:
        xs = xys[:, 0]
        min_x = min_xy[0]
        min_x = max_xy[0]

    if log_y:
        ys = np.log10(xys[:, 1])
        yticklabels = ['1e{:+}'.format(e)
                       for e in range(min_exp[1], max_exp[1])]
        min_y = np.log10(min_xy[1])
        max_y = np.log10(max_xy[1])
    else:
        ys = xys[:, 1]
        min_y = min_xy[1]
        max_x = max_xy[1]

    if columns is not None:
        assert len(columns) == 2 + foms.shape[1], 'wrong number of columns'

    fig = plt.figure()
    for i in range(foms.shape[1]):
        # Arrange plots in one column
        ax = fig.add_subplot(foms.shape[1], 1, i + 1, projection='3d')

        ax.scatter3D(xs, ys, foms[:, i])
        if columns is None:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z_{}'.format(i + 1))
        else:
            ax.set_xlabel(columns[0])
            ax.set_ylabel(columns[1])
            ax.set_zlabel(columns[i + 2].upper())

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        if log_x:
            ax.set_xticklabels(xticklabels)
        if log_y:
            ax.set_yticklabels(yticklabels)

    if cycle is None:
        fig.canvas.set_window_title('FOMs')
        if columns is None:
            fig.suptitle('Figures of Merit')
        else:
            fig.suptitle(
                'Figures of Merit: {}'
                ''.format(', '.join(fom.upper() for fom in columns[2:])))
    else:
        fig.canvas.set_window_title('FOMs cycle {}'.format(cycle))
        if columns is None:
            fig.suptitle(
                'Figures of Merit (cycle {})'
                ''.format(', '.join(fom.upper() for fom in foms)))
        else:
            fig.suptitle(
                'Figures of Merit (cycle {}): {}'
                ''.format(cycle,
                          ', '.join(fom.upper() for fom in columns[2:])))

    return fig


def read_results(path):
    """Read results from files on a given path.

    The results are expected to be given in ``meta_cycle_[num].json``
    files with for metadata and ``data_cycle_[num].npy`` for numeric data.
    """
    path = os.path.normpath(path)
    file_names = os.listdir(path)
    meta_file_names = [n for n in file_names
                       if n.startswith('meta_cycle_') and
                       n.endswith('.json')]
    metadata = [json.load(open(os.path.join(path, fname), 'r'))
                for fname in sorted(meta_file_names)]
    data_file_names = [n for n in file_names
                       if n.startswith('data_cycle_') and
                       n.endswith('.npy')]
    data = [np.load(os.path.join(path, fname))
            for fname in sorted(data_file_names)]

    return tuple(zip(metadata, data))
