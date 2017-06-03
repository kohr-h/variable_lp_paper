# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Helpers for Numpy implementation of variable Lp functionality."""


from __future__ import division

import numpy as np


__all__ = ('varlp_prox_factor_npy', 'varlp_cc_prox_factor_npy',
           'varlp_cc_integrand_npy', 'varlp_moreau_integrand_npy')


# --- Proximal operator --- #


def varlp_prox_factor_npy(abs_f, p, sigma, num_newton_iter, out=None):
    """Multiplicative factor for the variable Lp proximal.

    Parameters
    ----------
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
    out : `numpy.ndarray`, optional
        Array to which the result is written. It must be contiguous and
        have the same ``dtype`` and ``shape`` as ``abs_f``.

    Returns
    -------
    out : `numpy.ndarray`
        Factor for the proximal operator. Has the same shape as ``abs_f``.
        If ``out`` was provided, the returned object is a reference to it.

    Examples
    --------
    When ``abs_f == 0``, the returned value is always 0.
    Otherwise, exponent ``p = 1`` gives ``max(1.0 - sigma / abs_f, 0.0)``:

    >>> abs_f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> sigma = 1.0
    >>> result = varlp_prox_factor_npy(abs_f, p1, sigma,
    ...                                num_newton_iter=1)
    >>> np.allclose(result, [0, 0, 0, 1.0 / 3.0, 0.5])
    True
    >>> sigma = 0.5
    >>> result = varlp_prox_factor_npy(abs_f, p1, sigma,
    ...                                num_newton_iter=1)
    >>> np.allclose(result, [0, 0, 0.5, 2.0 / 3.0, 0.75])
    True

    With ``p = 2`` one gets ``1 / (1 + 2 * sigma)``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> sigma = 0.5
    >>> result = varlp_prox_factor_npy(abs_f, p2, sigma,
    ...                                num_newton_iter=1)
    >>> np.allclose(result, [0] + [0.5] * 4)
    True

    For other ``p`` values, the result times ``abs_f`` solves the
    equation ``v + sigma * p * v**(p-1) = abs_f``:

    >>> p15 = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    >>> sigma = 1.0
    >>> result = varlp_prox_factor_npy(abs_f, p15, sigma,
    ...                                num_newton_iter=10)
    >>> v = result * abs_f
    >>> lhs = v + sigma * p15 * v ** (p15 - 1)
    >>> np.allclose(lhs, abs_f)
    True
    """
    if out is None:
        out = np.empty_like(abs_f)

    abs_f_nz = (abs_f > 1e-8)

    # Set to 0 where f = 0
    out[~abs_f_nz] = 0

    # p = 2
    cur_p = (p >= 1.95)
    mask = cur_p & abs_f_nz
    out[mask] = 1 / (1 + 2 * sigma)

    # p = 1 (taking also close to one for stability)
    cur_p = (p <= 1.05)
    mask = cur_p & abs_f_nz
    out[mask] = np.maximum(1.0 - sigma / abs_f[mask], 0.0)

    # Newton iteration for other p values. We consider only those
    # entries that correspond to f != 0.
    cur_p = ~((p >= 1.95) | cur_p)  # exponents 1.05 < p < 1.95
    mask = cur_p & abs_f_nz
    cur_exp = p[mask]
    cur_out = out[mask]  # not a view, so out is not modified
    rhs = abs_f[mask]
    varlp_newton_iter_npy(rhs, cur_exp, sigma, num_newton_iter, cur_out)

    out[mask] = cur_out / rhs

    return out


# --- Proximal operator of the convex conjugate --- #


def varlp_cc_prox_factor_npy(abs_f, p, sigma, num_newton_iter, out=None):
    """Multiplicative factor for the variable Lp cc prox, Numpy version.

    Parameters
    ----------
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
    out : `numpy.ndarray`, optional
        Array to which the result is written. It must be contiguous and
        have the same ``dtype`` and ``shape`` as ``abs_f``.

    Returns
    -------
    out : `numpy.ndarray`
        Factor for the proximal operator of the convex conjugate.
        Has the same shape as ``abs_f``.
        If ``out`` was provided, the returned object is a reference to it.

    Examples
    --------
    When ``abs_f == 0``, the returned value is always 0.
    Otherwise, exponent ``p = 1`` gives ``min(1, 1 / abs_f)``:

    >>> abs_f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> sigma = 1.0
    >>> result = varlp_cc_prox_factor_npy(abs_f, p1, sigma,
    ...                                   num_newton_iter=1)
    >>> np.allclose(result, [0, 1, 1, 2.0 / 3.0, 0.5])
    True

    With ``p = 2`` one gets ``2 / (2 + sigma)``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> sigma = 0.5
    >>> result = varlp_cc_prox_factor_npy(abs_f, p2, sigma,
    ...                                   num_newton_iter=1)
    >>> np.allclose(result, [0] + [0.8] * 4)
    True

    For other ``p`` values, the result is ``1 - v / abs_f``, where ``v``
    satisfies the equation ``v + sigma**(1-p) * p * v**(p-1) = abs_f``:

    >>> p15 = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    >>> sigma = 1.0
    >>> result = varlp_cc_prox_factor_npy(abs_f, p15, sigma,
    ...                                   num_newton_iter=5)
    >>> v = (1 - result) * abs_f
    >>> lhs = v + sigma ** (1 - p15) * p15 * v ** (p15 - 1)
    >>> np.allclose(lhs, abs_f)
    True
    """
    # TODO: validate_properties shows this is wrong, fix it!
    if out is None:
        out = np.empty_like(abs_f)

    abs_f_nz = (abs_f > 1e-8)

    # Set to 0 where f = 0
    out[~abs_f_nz] = 0

    # p = 2
    cur_p = (p >= 1.95)
    mask = cur_p & abs_f_nz
    out[mask] = 2 / (sigma + 2)

    # p = 1 (taking also close to one for stability)
    cur_p = (p <= 1.05)
    mask = cur_p & abs_f_nz
    out[mask] = np.minimum(1 / abs_f[mask], 1.0)

    # Newton iteration for other p values. We consider only those
    # entries that correspond to f != 0.
    cur_p = ~((p >= 1.95) | cur_p)  # exponents 1.05 < p < 1.95
    mask = cur_p & abs_f_nz
    cur_exp = p[mask]
    cur_out = out[mask]  # not a view, so out is not modified
    rhs = abs_f[mask]
    sigma_cc = np.power(sigma, cur_exp - 1)
    varlp_newton_iter_npy(rhs, cur_exp, sigma_cc, num_newton_iter, cur_out)

    out[mask] = 1 - cur_out / rhs

    return out


def varlp_newton_iter_npy(rhs, p, sigma, niter, out):
    """Newton iteration for the variable Lp proximal, Numpy version."""
    # Used often, store in temporary array
    pm1 = p - 1
    pm2 = p - 2

    # Find a good starting value for the iteration.
    # Compute the solution to the equation for the extreme cases
    # p=1 and p=2. Our first guess is their convex combination.
    s1 = np.maximum(rhs - sigma, 0)
    s2 = rhs / (1 + 2 * sigma)
    out[:] = -pm2 * s1 + pm1 * s2

    # The condition for the first iterate to be valid is
    # sval^(p-1) < rhs / (sigma * p * (2-p))
    bound = (rhs / (-sigma * p * pm2)) ** (1 / pm1)
    large = (out >= bound)
    out[large] = np.minimum(np.minimum(bound[large] / 2, rhs[large]), 1.0)

    # The iteration itself
    tmp = np.empty_like(out)
    for _ in range(niter):
        # Denominator 1 + p * (p-1) * sigma * q**(p-2)
        np.power(out, pm2, out=tmp)
        tmp *= p
        tmp *= pm1
        tmp *= sigma
        tmp += 1.0

        # Numerator p * (p-2) * sigma * q**(p-1) + rhs
        np.power(out, pm1, out=out)
        out *= p
        out *= pm2
        out *= sigma
        out += rhs

        out /= tmp


# --- Integrand for the convex conjugate --- #


def varlp_cc_integrand_npy(abs_f, p, out=None):
    """Integrand for the variable Lp convex conjugate, Numpy version.

    abs_f : `numpy.ndarray`
        Magnitude of the input function (scalar or vectorial) to the
        proximal.
    p : `numpy.ndarray`
        Spatially varying exponent of the Lp modular. Must have same
        shape and dtype as ``abs_f``.
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
    >>> result = varlp_cc_integrand_npy(abs_f, p1)
    >>> np.allclose(result, [0, 0, 0, np.inf, np.inf])
    True

    With ``p = 2`` one gets ``abs_f ** 2 / 4``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> result = varlp_cc_integrand_npy(abs_f, p2)
    >>> np.allclose(result, abs_f ** 2 / 4)
    True

    For other ``p`` values, the result is ``abs_f**(p/(p-1)) * r``,
    where ``r = p**(-1/(p-1)) - p**(-p/(p-1))``:

    >>> p15 = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    >>> result = varlp_cc_integrand_npy(abs_f, p15)
    >>> r = p15 ** (-1 / (p15 - 1)) - p15 ** (-p15 / (p15 - 1))
    >>> np.allclose(result, abs_f ** (p15 / (p15 - 1)) * r)
    True
    """
    if out is None:
        out = np.empty_like(abs_f)

    abs_f_nz = (abs_f > 1e-8)

    # Set to 0 where f = 0
    out[~abs_f_nz] = 0

    # p = 2
    cur_p = (p >= 1.95)
    mask = cur_p & abs_f_nz
    out[mask] = abs_f[mask] * abs_f[mask] / 4

    # p = 1 (taking also close to one for stability)
    # Indicator of the unit ball
    cur_p = (p <= 1.05)
    mask = cur_p & abs_f_nz
    out[mask] = np.where(abs_f[mask] <= 1, 0.0, np.inf)

    # Other exponent values:
    # abs_f**(p/(p-1)) * (p**(-1/(p-1)) - p**(-p/(p-1)))
    cur_p = ~((p >= 1.95) | cur_p)  # exponents 1.05 < p < 1.95
    mask = cur_p & abs_f_nz
    cur_exp = p[mask]
    aux_exp = cur_exp / (cur_exp - 1)  # p/(p-1) -> 1/(p-1) = p/(p-1) - 1
    factor = cur_exp ** (1 - aux_exp) - cur_exp ** (-aux_exp)
    out[mask] = abs_f[mask] ** aux_exp * factor

    return out


# --- Integrand for the Moreau envelope --- #


def varlp_moreau_integrand_npy(abs_f, p, sigma, num_newton_iter, out=None):
    """Integrand for the variable Lp Moreau envelope, Numpy version.

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
    out : `numpy.ndarray`, optional
        Array to which the result is written. It must be contiguous and
        have the same ``dtype`` and ``shape`` as ``abs_f``.

    Returns
    -------
    out : `numpy.ndarray`
        Integrand of the proximal operator Moreau envelope.
        Has the same shape as ``abs_f``.
        If ``out`` was provided, the returned object is a reference to it.

    Examples
    --------
    Exponent ``p = 1`` gives the Huber function of ``abs_f``, that is
    ``abs_f ** 2 / (2 * sigma)`` if ``abs_f <= sigma`` and
    ``abs_f - sigma / 2`` otherwise:

    >>> abs_f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> sigma = 1.0
    >>> result = varlp_moreau_integrand_npy(abs_f, p1, sigma,
    ...                                     num_newton_iter=1)
    >>> np.allclose(result, [0, 0.125, 0.5, 1.0, 1.5])
    True
    >>> sigma = 0.5
    >>> result = varlp_moreau_integrand_npy(abs_f, p1, sigma,
    ...                                       num_newton_iter=1)
    >>> np.allclose(result, [0, 0.25, 0.75, 1.25, 1.75])
    True

    With ``p = 2`` one gets ``abs_f ** 2 / (1 + 2 * sigma)``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> sigma = 0.5
    >>> result = varlp_moreau_integrand_npy(abs_f, p2, sigma,
    ...                                       num_newton_iter=1)
    >>> np.allclose(result, [0, 0.125, 0.5, 1.125, 2])
    True
    """
    if out is None:
        out = np.empty_like(abs_f)

    abs_f_nz = (abs_f > 1e-8)

    # Set to 0 where f = 0
    out[~abs_f_nz] = 0

    # p = 2
    cur_p = (p >= 1.95)
    mask = cur_p & abs_f_nz
    out[mask] = abs_f[mask] * abs_f[mask] / (1 + 2 * sigma)

    # p = 1 (taking also close to one for stability)
    # The Huber function
    cur_p = (p <= 1.05)
    mask = cur_p & abs_f_nz
    out[mask] = np.where(abs_f[mask] <= sigma,
                         abs_f[mask] * abs_f[mask] / (2 * sigma),
                         abs_f[mask] - sigma / 2)

    # Other p values
    # Newton iteration, considering only entries where abs_f != 0.
    cur_p = ~((p >= 1.95) | cur_p)  # exponents 1.05 < p < 1.95
    mask = cur_p & abs_f_nz
    cur_exp = p[mask]
    cur_out = out[mask]  # not a view, so out is not modified
    rhs = abs_f[mask]
    sigma_cc = np.power(sigma, cur_exp - 1)
    varlp_newton_iter_npy(rhs, cur_exp, sigma_cc, num_newton_iter, cur_out)

    # Value is (r - abs_f) * (2*r + p*(r - abs_f)) / (2 * sigma * p),
    # where r is the result of the Newton iteration
    tmp = cur_out - abs_f[mask]
    out[mask] = tmp * (2 * cur_out + cur_exp * tmp) / (2 * sigma * cur_exp)

    return out


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
