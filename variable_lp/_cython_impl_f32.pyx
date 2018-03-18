# Copyright 2017, 2018 Holger Kohr
#
# This file is part of variable_lp_paper.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

# cython: profile=True

"""Cython implementation of variable Lp functionality for float32 dtype."""

import numpy as np

cimport cython
cimport numpy as np


__all__ = ('varlp_prox_factor_f32_c', 'varlp_cc_prox_factor_f32_c',
           'varlp_cc_integrand_f32_c', 'varlp_moreau_integrand_f32_c')


# --- Helpers and defs --- #


# C typedefs for reuse
ctypedef np.float32_t float_t
ctypedef np.int64_t int_t

# Max, min and abs
cdef inline float_t fmax(float_t x, float_t y):
    return x if x > y else y

cdef inline float_t fmin(float_t x, float_t y):
    return x if x < y else y

cdef inline float_t fabs(float_t x):
    return x if x >= 0 else -x


# --- Newton iteration with auxiliary functions --- #


cdef float_t start_value(float_t rhs, float_t p, float_t sigma):
    cdef float_t sval, s1, s2, bound

    # Compute the solution to the equation for the extreme cases
    # p=1 and p=2. Our first guess is their convex combination.
    s1 = fmax(rhs - sigma, 0)
    s2 = rhs / (1 + 2 * sigma)
    sval = (2 - p) * s1 + (p - 1) * s2

    # The condition for the first iterate to be valid is
    # sval^(p-1) < rhs / (sigma * p * (2-p))
    bound = (rhs / (sigma * p * (2 - p))) ** (1 / (p - 1))
    if sval >= bound:
        return fmin(fmin(bound / 2, rhs), 1.0)

    return sval


@cython.profile(False)
cdef float_t newton_iter(float_t abs_f, float_t p, float_t sigma, int_t niter):
    cdef:
        int_t i
        float_t it, denom, numer

    it = start_value(abs_f, p, sigma)
    for i in range(niter):
        numer = abs_f - p * (2 - p) * sigma * it ** (p - 1)
        denom = 1.0 + p * (p - 1) * sigma * it ** (p - 2)
        it = numer / denom

    return it


# --- Pointwise factors for the proximal --- #


@cython.boundscheck(False)
@cython.wraparound(False)
def varlp_prox_factor_f32_c(
        float_t[:] abs_f,
        float_t[:] p,
        float_t sigma,
        int_t num_newton_iter,
        float_t[:] out):
    """Pointwise factor for the variable Lp modular.

    Signature::

        void varlp_prox_factor_f32_c(np.ndarray[float32] abs_f,
                                     np.ndarray[float32] p,
                                     float32 sigma,
                                     int64 num_newton_iter,
                                     np.ndarray[float32] out)
    """
    cdef:
        int_t i
        int_t nx = abs_f.shape[0]
        float_t cur_f, cur_p

    for i in range(nx):
        cur_f = abs_f[i]
        cur_p = p[i]

        if cur_f < 1e-8:
            out[i] = 0.0

        elif cur_p <= 1.05:
            out[i] = fmax(1.0 - sigma / cur_f, 0.0)

        elif cur_p >= 1.95:
            out[i] = 1.0 / (1.0 + 2.0 * sigma)

        else:
            out[i] = newton_iter(cur_f, cur_p, sigma, num_newton_iter) / cur_f

    return out


# --- Pointwise factors for the proximal of the convex conjugate --- #


@cython.boundscheck(False)
@cython.wraparound(False)
def varlp_cc_prox_factor_f32_c(
        float_t[:] abs_f,
        float_t[:] p,
        float_t sigma,
        int_t num_newton_iter,
        float_t[:] out):
    """Pointwise factor for the convex conjugate of the variable Lp modular.

    Signature::

        void varlp_cc_prox_factor_f32_c(np.ndarray[float32] abs_f,
                                        np.ndarray[float32] p,
                                        float32 sigma,
                                        int64 num_newton_iter,
                                        np.ndarray[float32] out)
    """
    cdef:
        int_t i
        int_t nx = abs_f.shape[0]
        float_t sig_cc, cur_f, cur_p

    for i in range(nx):
        cur_f = abs_f[i]
        cur_p = p[i]

        if cur_f < 1e-8:
            out[i] = 0.0

        elif cur_p <= 1.05:
            out[i] = fmin(1.0, 1.0 / cur_f)

        elif cur_p >= 1.95:
            out[i] = 2.0 / (sigma + 2.0)

        else:
            sig_cc = sigma ** (1 - cur_p)
            out[i] = 1 - (newton_iter(cur_f, cur_p, sig_cc, num_newton_iter) /
                          cur_f)

    return out


# --- Integrand for the convex conjugate --- #


from numpy.math cimport INFINITY


@cython.boundscheck(False)
@cython.wraparound(False)
def varlp_cc_integrand_f32_c(
        float_t[:] abs_f,
        float_t[:] p,
        float_t[:] out):
    """Compute the convex conjugate integrand for the variable Lp modular.

    Signature::

        void varlp_cc_integrand_f32_c(np.ndarray[float32] abs_f,
                                      np.ndarray[float32] p,
                                      np.ndarray[float32] out)
    """
    cdef:
        int_t i
        int_t nx = abs_f.shape[0]
        float_t factor, cur_f, cur_p, aux_p

    for i in range(nx):
        cur_f = abs_f[i]
        cur_p = p[i]

        if cur_f < 1e-8:
            out[i] = 0.0

        elif cur_p <= 1.05:
            out[i] = 0.0 if cur_f <= 1 else INFINITY

        elif cur_p >= 1.95:
            out[i] = cur_f * cur_f / 4.0

        else:
            aux_p = cur_p / (cur_p - 1)
            factor = cur_p ** (1 - aux_p) - cur_p ** (-aux_p)
            out[i] = cur_f ** aux_p * factor

    return out


# --- Integrand for the Moreau envelope --- #


@cython.boundscheck(False)
@cython.wraparound(False)
def varlp_moreau_integrand_f32_c(
        float_t[:] abs_f,
        float_t[:] p,
        float_t sigma,
        int_t num_newton_iter,
        float_t[:] out):
    """Compute the Moreau envelope integrand for the variable Lp modular.

    Signature::

        void varlp_moreau_integrand_f32_c(np.ndarray[float32] abs_f,
                                          np.ndarray[float32] p,
                                          float32 sigma,
                                          int64 num_newton_iter,
                                          np.ndarray[float32] out)
    """
    cdef:
        int_t i
        int_t nx = abs_f.shape[0]
        float_t alpha, cur_f, cur_p, tmp

    for i in range(nx):
        cur_f = abs_f[i]
        cur_p = p[i]

        if cur_f < 1e-8:
            out[i] = 0.0

        elif cur_p <= 1.05:
            # The Huber function
            if cur_f <= sigma:
                out[i] = cur_f * cur_f / (2 * sigma)
            else:
                out[i] = cur_f - sigma / 2

        elif cur_p >= 1.95:
            out[i] = cur_f * cur_f / (1 + 2 * sigma)

        else:
            alpha = newton_iter(cur_f, cur_p, sigma, num_newton_iter)
            tmp = (cur_f - alpha)
            out[i] = tmp * (2 * alpha + cur_p * tmp) / (2 * sigma * cur_p)

    return out
