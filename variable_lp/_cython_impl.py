# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Helpers for Cython implementation of variable Lp functionality."""


import numpy as np
try:
    from . import _cython_impl_f32
    from . import _cython_impl_f64
except ImportError:
    CYTHON_EXTENSION_BUILT = False
else:
    CYTHON_EXTENSION_BUILT = True
    from ._cython_impl_f32 import (
        varlp_prox_factor_f32_c, varlp_cc_prox_factor_f32_c,
        varlp_cc_integrand_f32_c, varlp_moreau_integrand_f32_c)
    from ._cython_impl_f64 import (
        varlp_prox_factor_f64_c, varlp_cc_prox_factor_f64_c,
        varlp_cc_integrand_f64_c, varlp_moreau_integrand_f64_c)


__all__ = ('varlp_prox_factor_cython', 'varlp_cc_prox_factor_cython',
           'varlp_cc_integrand_cython', 'varlp_moreau_integrand_cython')


# --- Proximal operator --- #


def varlp_prox_factor_cython(abs_f, p, sigma, num_newton_iter, out=None):
    """Multiplicative factor for the variable Lp prox, Cython version.

    Parameters
    ----------
    abs_f : `numpy.ndarray`
        Magnitude of the input function (scalar or vectorial) to the
        proximal.
    p : `numpy.ndarray`
        Spatially varying exponent of the Lp modular. It must have the
        same ``shape`` and ``dtype`` as ``abs_f``.
    sigma : positive float
        Step-size-like parameter of the proximal.
    num_newton_iter : positive int
        Number of Newton iterations to perform for the places where
        ``1 < p < 2``.
    out : `numpy.ndarray`, optional
        Array to which the result is written. It must be contiguous and
        have the same ``dtype`` and ``shape`` as ``abs_f``.

    Returns
    -------
    out : `numpy.ndarray`
        Factor of the proximal operator. Has the same shape as ``abs_f``.
        If ``out`` was provided, the returned object is a reference to it.

    Examples
    --------
    When ``abs_f == 0``, the returned value is always 0.
    Otherwise, exponent ``p = 1`` gives ``max(1.0 - sigma / abs_f, 0.0)``:

    >>> abs_f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> sigma = 1.0
    >>> result = varlp_prox_factor_cython(abs_f, p1, sigma,
    ...                                   num_newton_iter=1)
    >>> np.allclose(result, [0, 0, 0, 1.0 / 3.0, 0.5])
    True
    >>> sigma = 0.5
    >>> result = varlp_prox_factor_cython(abs_f, p1, sigma,
    ...                                   num_newton_iter=1)
    >>> np.allclose(result, [0, 0, 0.5, 2.0 / 3.0, 0.75])
    True

    With ``p = 2`` one gets ``1 / (1 + 2 * sigma)``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> sigma = 0.5
    >>> result = varlp_prox_factor_cython(abs_f, p2, sigma,
    ...                                   num_newton_iter=1)
    >>> np.allclose(result, [0] + [0.5] * 4)
    True

    For other ``p`` values, the result times ``abs_f`` solves the
    equation ``v + sigma * p * v**(p-1) = abs_f``:

    >>> p15 = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    >>> sigma = 1.0
    >>> result = varlp_prox_factor_cython(abs_f, p15, sigma,
    ...                                   num_newton_iter=10)
    >>> v = result * abs_f
    >>> lhs = v + sigma * p15 * v ** (p15 - 1)
    >>> np.allclose(lhs, abs_f)
    True
    """
    if out is None:
        out = np.empty_like(abs_f)

    if all(arr.dtype == np.dtype('float32') for arr in (abs_f, p, out)):
        func = varlp_prox_factor_f32_c
    elif all(arr.dtype == np.dtype('float64') for arr in (abs_f, p, out)):
        func = varlp_prox_factor_f64_c
    else:
        raise ValueError("all arrays must either have 'float32' or "
                         "'float64' dtype")

    func(abs_f.ravel(), p.ravel(), sigma, num_newton_iter, out.ravel())
    return out


def varlp_cc_prox_factor_cython(abs_f, p, sigma, num_newton_iter, out=None):
    """Multiplicative factor for the variable Lp cconj prox, Cython version.

    Parameters
    ----------
    abs_f : `numpy.ndarray`
        Magnitude of the input function (scalar or vectorial) to the
        proximal.
    p : `numpy.ndarray`
        Spatially varying exponent of the Lp modular. It must have the
        same ``shape`` and ``dtype`` as ``abs_f``.
    sigma : positive float
        Step-size-like parameter of the proximal.
    num_newton_iter : positive int
        Number of Newton iterations to perform for the places where
        ``1 < p < 2``.
    out : `numpy.ndarray`, optional
        Array to which the result is written. It must be contiguous and
        have the same ``dtype`` and ``shape`` as ``abs_f``.

    Returns
    -------
    out : `numpy.ndarray`
        Factor of the proximal operator of the convex conjugate. Has the
        same shape as ``abs_f``.
        If ``out`` was provided, the returned object is a reference to it.

    Examples
    --------
    When ``abs_f == 0``, the returned value is always 0.
    Otherwise, exponent ``p = 1`` gives ``min(1, 1 / abs_f)``:

    >>> abs_f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> sigma = 1.0
    >>> result = varlp_cc_prox_factor_cython(abs_f, p1, sigma,
    ...                                      num_newton_iter=1)
    >>> np.allclose(result, [0, 1, 1, 2.0 / 3.0, 0.5])
    True

    With ``p = 2`` one gets ``2 / (2 + sigma)``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> sigma = 0.5
    >>> result = varlp_cc_prox_factor_cython(abs_f, p2, sigma,
    ...                                      num_newton_iter=1)
    >>> np.allclose(result, [0] + [0.8] * 4)
    True

    For other ``p`` values, the result is ``1 - v / abs_f``, where ``v``
    satisfies the equation ``v + sigma**(1-p) * p * v**(p-1) = abs_f``:

    >>> p15 = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    >>> sigma = 1.0
    >>> result = varlp_cc_prox_factor_cython(abs_f, p15, sigma,
    ...                                      num_newton_iter=5)
    >>> v = (1 - result) * abs_f
    >>> lhs = v + sigma ** (1 - p15) * p15 * v ** (p15 - 1)
    >>> np.allclose(lhs, abs_f)
    True
    """
    if out is None:
        out = np.empty_like(abs_f)

    if all(arr.dtype == np.dtype('float32') for arr in (abs_f, p, out)):
        func = varlp_cc_prox_factor_f32_c
    elif all(arr.dtype == np.dtype('float64') for arr in (abs_f, p, out)):
        func = varlp_cc_prox_factor_f64_c
    else:
        raise ValueError("all arrays must either have 'float32' or "
                         "'float64' dtype")

    func(abs_f.ravel(), p.ravel(), sigma, num_newton_iter, out.ravel())
    return out


def varlp_cc_integrand_cython(abs_f, p, out=None):
    """Integrand of the variable Lp convex conjugate, Cython version.

    Parameters
    ----------
    abs_f : `numpy.ndarray`
        Magnitude of the input function (scalar or vectorial) to the
        functional.
    p : `numpy.ndarray`
        Spatially varying exponent of the Lp modular. It must have the
        same ``shape`` and ``dtype`` as ``abs_f``.
    out : `numpy.ndarray`, optional
        Array to which the result is written. It must be contiguous and
        have the same ``dtype`` and ``shape`` as ``abs_f``.

    Returns
    -------
    out : `numpy.ndarray`
        Integrand of the convex conjugate. Has the same shape as ``abs_f``.
        If ``out`` was provided, the returned object is a reference to it.

    Examples
    --------
    Exponent ``p = 1`` gives the indicator of the unit ball:

    >>> abs_f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> result = varlp_cc_integrand_cython(abs_f, p1)
    >>> np.allclose(result, [0, 0, 0, np.inf, np.inf])
    True

    With ``p = 2`` one gets ``abs_f ** 2 / 4``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> result = varlp_cc_integrand_cython(abs_f, p2)
    >>> np.allclose(result, abs_f ** 2 / 4)
    True

    For other ``p`` values, the result is ``abs_f**(p/(p-1)) * r``,
    where ``r = p**(-1/(p-1)) - p**(-p/(p-1))``:

    >>> p15 = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    >>> result = varlp_cc_integrand_cython(abs_f, p15)
    >>> r = p15 ** (-1 / (p15 - 1)) - p15 ** (-p15 / (p15 - 1))
    >>> np.allclose(result, abs_f ** (p15 / (p15 - 1)) * r)
    True
    """
    if out is None:
        out = np.empty_like(abs_f)

    if all(arr.dtype == np.dtype('float32') for arr in (abs_f, p, out)):
        func = varlp_cc_integrand_f32_c
    elif all(arr.dtype == np.dtype('float64') for arr in (abs_f, p, out)):
        func = varlp_cc_integrand_f64_c
    else:
        raise ValueError("all arrays must either have 'float32' or "
                         "'float64' dtype")

    func(abs_f.ravel(), p.ravel(), out.ravel())
    return out


def varlp_moreau_integrand_cython(abs_f, p, sigma, num_newton_iter, out=None):
    """Integrand of the variable Lp Moreau envelope, Cython version.

    Parameters
    ----------
    abs_f : `numpy.ndarray`
        Magnitude of the input function (scalar or vectorial) to the
        functional.
    p : `numpy.ndarray`
        Spatially varying exponent of the Lp modular. It must have the
        same ``shape`` and ``dtype`` as ``abs_f``.
    sigma : positive float
        Step-size-like parameter of the functional.
    num_newton_iter : positive int
        Number of Newton iterations to perform for the places where
        ``1 < p < 2``.
    out : `numpy.ndarray`, optional
        Array to which the result is written. It must be contiguous and
        have the same ``dtype`` and ``shape`` as ``abs_f``.

    Returns
    -------
    out : `numpy.ndarray`
        Integrand of the Moreau envelope. Has the same shape as ``abs_f``.
        If ``out`` was provided, the returned object is a reference to it.

    Examples
    --------
    Exponent ``p = 1`` gives the Huber function of ``abs_f``, that is
    ``abs_f ** 2 / (2 * sigma)`` if ``abs_f <= sigma`` and
    ``abs_f - sigma / 2`` otherwise:

    >>> abs_f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> sigma = 1.0
    >>> result = varlp_moreau_integrand_cython(abs_f, p1, sigma,
    ...                                        num_newton_iter=1)
    >>> np.allclose(result, [0, 0.125, 0.5, 1.0, 1.5])
    True
    >>> sigma = 0.5
    >>> result = varlp_moreau_integrand_cython(abs_f, p1, sigma,
    ...                                        num_newton_iter=1)
    >>> np.allclose(result, [0, 0.25, 0.75, 1.25, 1.75])
    True

    With ``p = 2`` one gets ``abs_f ** 2 / (1 + 2 * sigma)``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> sigma = 0.5
    >>> result = varlp_moreau_integrand_cython(abs_f, p2, sigma,
    ...                                        num_newton_iter=1)
    >>> np.allclose(result, [0, 0.125, 0.5, 1.125, 2])
    True
    """
    if out is None:
        out = np.empty_like(abs_f)

    if all(arr.dtype == np.dtype('float32') for arr in (abs_f, p, out)):
        func = varlp_moreau_integrand_f32_c
    elif all(arr.dtype == np.dtype('float64') for arr in (abs_f, p, out)):
        func = varlp_moreau_integrand_f64_c
    else:
        raise ValueError("all arrays must either have 'float32' or "
                         "'float64' dtype")

    func(abs_f.ravel(), p.ravel(), sigma, num_newton_iter, out.ravel())
    return out


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
