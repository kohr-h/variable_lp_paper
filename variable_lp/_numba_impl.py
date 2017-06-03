# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Helpers for Numba implementation of variable Lp functionality."""

from __future__ import division

import numba
import numpy as np


__all__ = ('varlp_prox_factor_numba', 'varlp_cc_prox_factor_numba',
           'varlp_cc_integrand_numba', 'varlp_moreau_integrand_numba')


# --- Proximal operator --- #


def varlp_prox_factor_numba(abs_f, p, sigma, num_newton_iter, target):
    """Multiplicative factor for the variable Lp prox, Numba version.

    abs_f : `numpy.ndarray`
        Magnitude of the input function (scalar or vectorial) to the
        proximal.
    p : `numpy.ndarray`
        Spatially varying exponent of the Lp modular. Must have same
        shape and dtype as ``abs_f``.
    sigma : positive float
        Step-size-like parameter of the proximal.
    num_newton_iter : positive int
        Number of Newton iterations to perform for the places where
        ``1 < p < 2``.
    target : {'cpu', 'cuda', 'parallel'}
        Target architecture of the Numba JIT.

    Returns
    -------
    out : `numpy.ndarray`
        Factor of the proximal operator. Has the same shape as ``abs_f``.

    Examples
    --------
    When ``abs_f == 0``, the returned value is always 0.
    Otherwise, exponent ``p = 1`` gives ``max(1.0 - sigma / abs_f, 0.0)``:

    >>> abs_f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> sigma = 1.0
    >>> result = varlp_prox_factor_numba(abs_f, p1, sigma,
    ...                                  num_newton_iter=1,
    ...                                  target='cpu')
    >>> np.allclose(result, [0, 0, 0, 1.0 / 3.0, 0.5])
    True
    >>> sigma = 0.5
    >>> result = varlp_prox_factor_numba(abs_f, p1, sigma,
    ...                                  num_newton_iter=1,
    ...                                  target='cpu')
    >>> np.allclose(result, [0, 0, 0.5, 2.0 / 3.0, 0.75])
    True

    With ``p = 2`` one gets ``1 / (1 + 2 * sigma)``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> sigma = 0.5
    >>> result = varlp_prox_factor_numba(abs_f, p2, sigma,
    ...                                  num_newton_iter=1,
    ...                                  target='cpu')
    >>> np.allclose(result, [0] + [0.5] * 4)
    True

    For other ``p`` values, the result times ``abs_f`` solves the
    equation ``v + sigma * p * v**(p-1) = abs_f``:

    >>> p15 = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    >>> sigma = 1.0
    >>> result = varlp_prox_factor_numba(abs_f, p15, sigma,
    ...                                  num_newton_iter=10,
    ...                                  target='cpu')
    >>> v = result * abs_f
    >>> lhs = v + sigma * p15 * v ** (p15 - 1)
    >>> np.allclose(lhs, abs_f)
    True
    """
    if target == 'cuda':
        vec_kwargs = {}
    else:
        vec_kwargs = {'nopython': True}

    dt = numba.numpy_support.FROM_DTYPE[np.dtype(abs_f.dtype)]
    nb_int = numba.numpy_support.FROM_DTYPE[np.dtype(int)]
    type_sig = dt(dt, dt, dt, nb_int)

    @numba.vectorize([type_sig], target=target, **vec_kwargs)
    def varlp_prox_factor(abs_f, p, sigma, num_newton_iter):
        """Pointwise multiplicative factor for the variable Lp prox."""
        if abs_f <= 1e-8:
            return 0.0
        elif p <= 1.05:
            return max(1 - sigma / abs_f, 0.0)
        elif p >= 1.95:
            return 1 / (1 + 2 * sigma)
        else:
            # Newton iteration

            # Find a good starting value.
            # Compute the solution to the equation for the extreme cases
            # p=1 and p=2. Our first guess is their convex combination.
            s1 = max(abs_f - sigma, 0)
            s2 = abs_f / (1 + 2 * sigma)
            sval = (2 - p) * s1 + (p - 1) * s2

            # The condition for the first iterate to be valid is
            # sval^(p-1) < rhs / (sigma * p * (2-p))
            bound = (abs_f / (sigma * p * (2 - p))) ** (1 / (p - 1))
            if sval >= bound:
                sval = min(min(bound / 2, abs_f), 1.0)

            # The actual iteration
            it = sval
            for _ in range(num_newton_iter):
                numer = abs_f - p * (2 - p) * sigma * it ** (p - 1)
                denom = 1 + p * (p - 1) * sigma * it ** (p - 2)
                it = numer / denom

            return it / abs_f

    result = varlp_prox_factor(abs_f.ravel(), p.ravel(), sigma,
                               num_newton_iter)
    return result.reshape(abs_f.shape)


# --- Proximal operator of the convex conjugate --- #


def varlp_cc_prox_factor_numba(abs_f, p, sigma, num_newton_iter, target):
    """Multiplicative factor for the variable Lp cc prox, Numba version.

    abs_f : `numpy.ndarray`
        Magnitude of the input function (scalar or vectorial) to the
        proximal.
    p : `numpy.ndarray`
        Spatially varying exponent of the Lp modular. Must have same
        shape and dtype as ``abs_f``.
    sigma : positive float
        Step-size-like parameter of the proximal.
    num_newton_iter : positive int
        Number of Newton iterations to perform for the places where
        ``1 < p < 2``.
    target : {'cpu', 'cuda', 'parallel'}
        Target architecture of the Numba JIT.

    Returns
    -------
    out : `numpy.ndarray`
        Factor of the proximal operator of the convex conjugate.
        Has the same shape as ``abs_f``.

    Examples
    --------
    When ``abs_f == 0``, the returned value is always 0.
    Otherwise, exponent ``p = 1`` gives ``min(1, 1 / abs_f)``:

    >>> abs_f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> sigma = 1.0
    >>> result = varlp_cc_prox_factor_numba(abs_f, p1, sigma,
    ...                                     num_newton_iter=1,
    ...                                     target='cpu')
    >>> np.allclose(result, [0, 1, 1, 2.0 / 3.0, 0.5])
    True

    With ``p = 2`` one gets ``2 / (2 + sigma)``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> sigma = 0.5
    >>> result = varlp_cc_prox_factor_numba(abs_f, p2, sigma,
    ...                                     num_newton_iter=1,
    ...                                     target='cpu')
    >>> np.allclose(result, [0] + [0.8] * 4)
    True

    For other ``p`` values, the result is ``1 - v / abs_f``, where ``v``
    satisfies the equation ``v + sigma**(1-p) * p * v**(p-1) = abs_f``:

    >>> p15 = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    >>> sigma = 1.0
    >>> result = varlp_cc_prox_factor_numba(abs_f, p15, sigma,
    ...                                     num_newton_iter=5,
    ...                                     target='cpu')
    >>> v = (1 - result) * abs_f
    >>> lhs = v + sigma ** (1 - p15) * p15 * v ** (p15 - 1)
    >>> np.allclose(lhs, abs_f)
    True
    """
    if target == 'cuda':
        vec_kwargs = {}
    else:
        vec_kwargs = {'nopython': True}

    dt = numba.numpy_support.FROM_DTYPE[np.dtype(abs_f.dtype)]
    nb_int = numba.numpy_support.FROM_DTYPE[np.dtype(int)]
    type_sig = dt(dt, dt, dt, nb_int)

    @numba.vectorize([type_sig], target=target, **vec_kwargs)
    def varlp_cc_prox_factor(abs_f, p, sigma, num_newton_iter):
        """Pointwise multiplicative factor for the variable Lp cc prox."""
        if abs_f <= 1e-8:
            return 0.0
        elif p <= 1.05:
            return min(1 / abs_f, 1.0)
        elif p >= 1.95:
            return 2 / (sigma + 2.0)
        else:
            sigma_cc = sigma ** (1 - p)
            # Newton iteration

            # Find a good starting value.
            # Compute the solution to the equation for the extreme cases
            # p=1 and p=2. Our first guess is their convex combination.
            s1 = max(abs_f - sigma_cc, 0)
            s2 = abs_f / (1 + 2 * sigma_cc)
            sval = (2 - p) * s1 + (p - 1) * s2

            # The condition for the first iterate to be valid is
            # sval^(p-1) < rhs / (sigma * p * (2-p))
            bound = (abs_f / (sigma_cc * p * (2 - p))) ** (1 / (p - 1))
            if sval >= bound:
                sval = min(min(bound / 2, abs_f), 1.0)

            # The actual iteration
            it = sval
            for _ in range(num_newton_iter):
                numer = abs_f - p * (2 - p) * sigma_cc * it ** (p - 1)
                denom = 1 + p * (p - 1) * sigma_cc * it ** (p - 2)
                it = numer / denom

            return 1 - it / abs_f

    result = varlp_cc_prox_factor(abs_f.ravel(), p.ravel(), sigma,
                                  num_newton_iter)
    return result.reshape(abs_f.shape)


# --- Integrand for the convex conjugate --- #


def varlp_cc_integrand_numba(abs_f, p, target):
    """Integrand for the variable Lp convex conjugate, Numba version.

    abs_f : `numpy.ndarray`
        Magnitude of the input function (scalar or vectorial) to the
        functional.
    p : `numpy.ndarray`
        Spatially varying exponent of the Lp modular. Must have same
        shape and dtype as ``abs_f``.
    target : {'cpu', 'cuda', 'parallel'}
        Target architecture of the Numba JIT.

    Returns
    -------
    out : `numpy.ndarray`
        Integrand of the convex conjugate. Has the same shape as ``abs_f``.

    Examples
    --------
    Exponent ``p = 1`` gives the indicator of the unit ball:

    >>> abs_f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> result = varlp_cc_integrand_numba(abs_f, p1, target='cpu')
    >>> np.allclose(result, [0, 0, 0, np.inf, np.inf])
    True

    With ``p = 2`` one gets ``abs_f ** 2 / 4``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> result = varlp_cc_integrand_numba(abs_f, p2, target='cpu')
    >>> np.allclose(result, abs_f ** 2 / 4)
    True

    For other ``p`` values, the result is ``abs_f**(p/(p-1)) * r``,
    where ``r = p**(-1/(p-1)) - p**(-p/(p-1))``:

    >>> p15 = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    >>> result = varlp_cc_integrand_numba(abs_f, p15, target='cpu')
    >>> r = p15 ** (-1 / (p15 - 1)) - p15 ** (-p15 / (p15 - 1))
    >>> np.allclose(result, abs_f ** (p15 / (p15 - 1)) * r)
    True
    """
    if target == 'cuda':
        vec_kwargs = {}
    else:
        vec_kwargs = {'nopython': True}

    dt = numba.numpy_support.FROM_DTYPE[np.dtype(abs_f.dtype)]
    type_sig = dt(dt, dt)

    @numba.vectorize([type_sig], target=target, **vec_kwargs)
    def varlp_cc_integrand(abs_f, p):
        """Integrand of the variable Lp convex conjugate."""
        if abs_f <= 1e-8:
            return 0.0
        elif p <= 1.05:
            return 0.0 if abs_f <= 1 else np.inf
        elif p >= 1.95:
            return abs_f * abs_f / 4
        else:
            factor = p ** (-1 / (p - 1)) - p ** (-p / (p - 1))
            return abs_f ** (p / (p - 1)) * factor

    result = varlp_cc_integrand(abs_f.ravel(), p.ravel())
    return result.reshape(abs_f.shape)


# --- Integrand for the Moreau envelope --- #


def varlp_moreau_integrand_numba(abs_f, p, sigma, num_newton_iter, target):
    """Integrand for the variable Lp Moreau envelope, Numba version.

    abs_f : `numpy.ndarray`
        Magnitude of the input function (scalar or vectorial) to the
        functional.
    p : `numpy.ndarray`
        Spatially varying exponent of the Lp modular. Must have same
        shape and dtype as ``abs_f``.
    sigma : positive float
        Step-size-like parameter of the envelope.
    num_newton_iter : positive int
        Number of Newton iterations to perform for the places where
        ``1 < p < 2``.
    target : {'cpu', 'cuda', 'parallel'}
        Target architecture of the Numba JIT.

    Returns
    -------
    out : `numpy.ndarray`
        Integrand of the proximal operator Moreau envelope.
        Has the same shape as ``abs_f``.

    Examples
    --------
    Exponent ``p = 1`` gives the Huber function of ``abs_f``, that is
    ``abs_f ** 2 / (2 * sigma)`` if ``abs_f <= sigma`` and
    ``abs_f - sigma / 2`` otherwise:

    >>> abs_f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> sigma = 1.0
    >>> result = varlp_moreau_integrand_numba(abs_f, p1, sigma,
    ...                                       num_newton_iter=1,
    ...                                       target='cpu')
    >>> np.allclose(result, [0, 0.125, 0.5, 1.0, 1.5])
    True
    >>> sigma = 0.5
    >>> result = varlp_moreau_integrand_numba(abs_f, p1, sigma,
    ...                                       num_newton_iter=1,
    ...                                       target='cpu')
    >>> np.allclose(result, [0, 0.25, 0.75, 1.25, 1.75])
    True

    With ``p = 2`` one gets ``abs_f ** 2 / (1 + 2 * sigma)``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> sigma = 0.5
    >>> result = varlp_moreau_integrand_numba(abs_f, p2, sigma,
    ...                                       num_newton_iter=1,
    ...                                       target='cpu')
    >>> np.allclose(result, [0, 0.125, 0.5, 1.125, 2])
    True
    """
    if target == 'cuda':
        vec_kwargs = {}
    else:
        vec_kwargs = {'nopython': True}

    dt = numba.numpy_support.FROM_DTYPE[np.dtype(abs_f.dtype)]
    nb_int = numba.numpy_support.FROM_DTYPE[np.dtype(int)]
    type_sig = dt(dt, dt, dt, nb_int)

    @numba.vectorize([type_sig], target=target, **vec_kwargs)
    def varlp_moreau_integrand(abs_f, p, sigma, num_newton_iter):
        """Integrand of the variable Lp Moreau envelope."""
        if abs_f <= 1e-8:
            return 0.0
        elif p <= 1.05:
            if abs_f <= sigma:
                return abs_f * abs_f / (2 * sigma)
            else:
                return abs_f - sigma / 2
        elif p >= 1.95:
            return abs_f * abs_f / (1 + 2 * sigma)
        else:
            # Newton iteration

            # Find a good starting value.
            # Compute the solution to the equation for the extreme cases
            # p=1 and p=2. Our first guess is their convex combination.
            s1 = max(abs_f - sigma, 0)
            s2 = abs_f / (1 + 2 * sigma)
            sval = (2 - p) * s1 + (p - 1) * s2

            # The condition for the first iterate to be valid is
            # sval^(p-1) < rhs / (sigma * p * (2-p))
            bound = (abs_f / (sigma * p * (2 - p))) ** (1 / (p - 1))
            if sval >= bound:
                sval = min(min(bound / 2, abs_f), 1.0)

            # The actual iteration
            it = sval
            for _ in range(num_newton_iter):
                numer = abs_f - p * (2 - p) * sigma * it ** (p - 1)
                denom = 1 + p * (p - 1) * sigma * it ** (p - 2)
                it = numer / denom

            tmp = (abs_f - it)
            return tmp * (2 * it + p * tmp) / (2 * sigma * p)

    result = varlp_moreau_integrand(abs_f.ravel(), p.ravel(), sigma,
                                    num_newton_iter)
    return result.reshape(abs_f.shape)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
