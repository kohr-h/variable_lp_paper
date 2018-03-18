# Copyright 2017, 2018 Holger Kohr
#
# This file is part of variable_lp_paper.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Kernels for the GpuArray version of the variable Lp functions."""


import numpy as np

try:
    import mako
    import pygpu
except ImportError:
    PYGPU_AVAILABLE = False
else:
    PYGPU_AVAILABLE = True
    DTYPE_TO_CTYPE = {v: k
                      for k, v in pygpu.dtypes.NAME_TO_DTYPE.items()
                      if k.startswith('ga')}


__all__ = ('varlp_prox_factor_gpuary', 'varlp_cc_prox_factor_gpuary',
           'varlp_cc_integrand_gpuary', 'varlp_moreau_integrand_gpuary')


# --- Preamble template strings --- #


newton_tpl_str = """
/*************************************************************************/
/*---- Newton iteration ----*/
/*************************************************************************/
/*
start_value(rhs, p, sigma)

Compute a starting value for the Newton iteration to solve the equation::

    q + p * sigma * q^(p-1) = rhs.

Since q >= 0 is required, the start value must ensure that the first
Newton iterate is positive, too.

Parameters
----------
rhs : positive real
    Right-hand side of the equation.
p : positive real
    Exponent value, must be >= 1.
sigma : positive real
    Step-like parameter for the proximal.

Returns
-------
start_value : positive real
    A start value that guarantees a valid first Newton iterate.
*/

WITHIN_KERNEL ${dtype}
start_value(${dtype} rhs,
            ${dtype} p,
            ${dtype} sigma
            ){

    ${dtype} sval, s1, s2, bound;

    // Compute the solution to the equation for the extreme cases
    // p=1 and p=2. Our first guess is their convex combination.
    s1 = ${maximum}(rhs - sigma, 0);
    s2 = rhs / (1 + 2 * sigma);
    sval = (2 - p) * s1 + (p -1) * s2;

    // The condition for the first iterate to be valid is
    // sval^(p-1) < u / (sigma * p * (2-p))
    bound = ${power}(rhs / (sigma * p * (2 - p)), 1 / (p - 1));
    if (sval < bound)
        return sval;
    else
        return ${minimum}(${minimum}(bound / 2, rhs), 1.0);
}

/*************************************************************************/
/*
newton_iter(rhs, p, sigma, niter)

Perform niter Newton iteration steps to solve the equation::

    q + p * sigma * q^(p-1) = rhs.

The iteration rule is::

    it = [rhs - p*(2-p) * sigma * it^(p - 1)] /
         [1 + p*(p - 1) * sigma * it^(p - 2)]

Parameters
----------
rhs : positive real
    Right-hand side of the equation.
p : positive real
    Exponent value, must be >= 1.
sigma : positive real
    Parameter in the equation (step-like for the prox/Moreau envelope).
niter : positive int
    Number of iterations to be performed.

Returns
-------
solution : positive real
    Approximate solution to the equation computed by Newton iteration.
*/

WITHIN_KERNEL ${dtype}
newton_iter(${dtype} rhs,
            ${dtype} p,
            ${dtype} sigma,
            int niter
            ){
    int i;
    ${dtype} it, denom, numer;

    // Get a start value for the iteration
    it = start_value(rhs, p, sigma);

    // The iteration itself
    for(i = 0; i < niter; i++){
        numer = rhs - p * (2 - p) * sigma * ${power}(it, p - 1);
        denom = 1.0 + p * (p - 1) * sigma / ${power}(it, 2 - p);
        it = numer / denom;
    }
    return it;
}
"""


prox_tpl_str = """
/*************************************************************************/
/*---- Variable Lp proximal factor implementation ----*/
/*************************************************************************/
/*

varlp_prox_factor(abs_f, p, sigma, num_newton_iter)

Factor for the variable Lp proximal operator that is multiplied with
a given input function f, using an exponent function p and a scalar
parameter sigma. The point-wise factor is given by::

           {0,                                       if f(x) == 0,
           {max(1 - sigma / |f(x)|, 0),              if p(x) == 1,
    u(x) = {1 / (1 + 2 * sigma),                     if p(x) == 2,
           {q / |f(x)| where q > 0 solves
           {q + p(x) * sigma * q^(p(x)-1) = |f(x)|,  otherwise.

Parameters
----------
abs_f : positive real
    Magnitude of the input function f.
p : positive real
    Exponent value, must be >= 1.
sigma : positive real
    Step-like parameter for the proximal.
num_newton_iter : positive int
    Number of Newton iterations that should be performed.

Returns
-------
factor : positive real
    Factor for the proximal operator.
*/

WITHIN_KERNEL ${dtype}
varlp_prox_factor(${dtype} abs_f,
                  ${dtype} p,
                  ${dtype} sigma,
                  int num_newton_iter
                  ){
    if(abs_f <= 1e-8)
        return 0.0;
    else if(p <= 1.05)  // Dont' get too close to 1 due to instability
        return ${maximum}(1.0 - sigma / abs_f, 0.0);
    else if(p >= 1.95)  // Dont' get too close to 2 due to instability
        return 1.0 / (1.0 + 2.0 * sigma);
    else {
        return newton_iter(abs_f, p, sigma, num_newton_iter) / abs_f;
    }
}
"""


cc_prox_tpl_str = """
/*************************************************************************/
/*---- Variable Lp convex conjugate proximal factor implementation ----*/
/*************************************************************************/
/*

varlp_cc_prox_factor(abs_f, p, sigma, num_newton_iter)

Factor for the variable Lp proximal operator of the convex conjugate that
is to be multiplied with a given input function f, using an exponent
function p and a scalar parameter sigma.
The point-wise factor is given by::

           {0,                                                if f(x) == 0,
           {max(1 - 1 / |f(x)|, 0),                           if p(x) == 1,
    u(x) = {sigma / (sigma + 2),                              if p(x) == 2,
           {1 - q / |f(x)| where q > 0 solves
           {q + p(x) * sigma^(1-p(x)) * q^(p(x)-1) = |f(x)|,  otherwise.

Parameters
----------
abs_f : positive real
    Magnitude of the input function f.
p : positive real
    Exponent value, must be >= 1.
sigma : positive real
    Step-like parameter for the proximal.
num_newton_iter : positive int
    Number of Newton iterations that should be performed.

Returns
-------
factor : positive real
    Factor for the proximal operator of the convex conjugate.
*/

WITHIN_KERNEL ${dtype}
varlp_cc_prox_factor(${dtype} abs_f,
                     ${dtype} p,
                     ${dtype} sigma,
                     int num_newton_iter
                     ){
    ${dtype} sigma_cc;

    if(abs_f <= 1e-8)
        return 0.0;
    else if(p <= 1.05)  // Dont' get too close to 1 due to instability
        return ${minimum}(1.0 / abs_f, 1.0);
    else if(p >= 1.95)  // Dont' get too close to 2 due to instability
        return 2.0 / (sigma + 2.0);
    else {
        sigma_cc = ${power}(sigma, 1.0 - p);
        return 1.0 - newton_iter(abs_f, p, sigma_cc, num_newton_iter) / abs_f;
    }
}
"""


cc_integr_tpl_str = """
/*************************************************************************/
/*---- Variable Lp convex conjugate integrand implementation ----*/
/*************************************************************************/
/*

varlp_cc_integrand(abs_f, p)

Integrand for the variable Lp convex conjugate, given by::

           {0,                                   if f(x) == 0,
           {0,                                   if p(x) == 1 and |f(x)| <= 1,
    I(x) = {INFINITY,                            if p(x) == 1 and |f(x)| > 1,
           {|f(x)|^2 / 4,                        if p(x) == 2,
           {|f(x)|^r * (p(x)^(1-r) - p(x)^(-r))
           {with r = p(x)/(p(x)-1),              otherwise.

Parameters
----------
abs_f : positive real
    Magnitude of the input function f.
p : positive real
    Exponent value, must be >= 1.

Returns
-------
integr : positive real
    Integrand for the convex conjugate.
*/

// This macro is not available for CUDA, need to define it
// See also https://github.com/Theano/libgpuarray/pull/359
#define INFINITY __int_as_float(0x7f800000)


WITHIN_KERNEL ${dtype}
varlp_cc_integrand(${dtype} abs_f,
                   ${dtype} p
                   ){
    ${dtype} r, factor;

    if(abs_f <= 1e-8)
        return 0.0;
    else if(p <= 1.05)
        return (abs_f <= 1.0) ? 0.0 : INFINITY;
    else if(p >= 1.95)
        return abs_f * abs_f / 4.0;
    else {
        r = p / (p - 1);
        factor = ${power}(p, 1.0 - r) - ${power}(p, -r);
        return ${power}(abs_f, r) * factor;
    }
}
"""


moreau_integr_tpl_str = """
/*************************************************************************/
/*---- Variable Lp Moreau envelope integrand implementation ----*/
/*************************************************************************/
/*

varlp_moreau_integrand(abs_f, p, sigma, num_newton_iter)

Integrand for the variable Lp Moreau envelope, given by::

           {0,                                if f(x) == 0,
           {|f(x)|^2 / (2*sigma),             if p(x) == 1 and |f(x)| <= sigma,
    I(x) = {|f(x)| - sigma/2,                 if p(x) == 1 and |f(x)| > sigma,
           {|f(x)|^2 / (1 + 2*sigma),         if p(x) == 2,
           {(z-q) * (2*q + p(z-q)) / (2*sigma*p(x)),
           {where z = |f(x)|, and q > 0 solves
           {q + p(x) * sigma * q^(p(x)-1) = |f(x)|,  otherwise.

Parameters
----------
abs_f : positive real
    Magnitude of the input function f.
p : positive real
    Exponent value, must be >= 1.
sigma : positive real
    Step-like parameter.
num_newton_iter : positive int
    Number of Newton iterations that should be performed.

Returns
-------
integr : positive real
    Integrand for the Moreau envelope.

*/
WITHIN_KERNEL ${dtype}
varlp_moreau_integrand(${dtype} abs_f,
                       ${dtype} p,
                       ${dtype} sigma,
                       int num_newton_iter
                       ){
    ${dtype} q;

    if(abs_f <= 1e-8)
        return 0.0;
    else if(p <= 1.05){
        if (abs_f <= sigma)
            return abs_f * abs_f / (2.0 * sigma);
        else
            return abs_f - sigma / 2.0;
    }
    else if(p >= 1.95)  // Dont' get too close to 2 due to instability
        return abs_f * abs_f / (1.0 + 2.0 * sigma);
    else {
        q = newton_iter(abs_f, p, sigma, num_newton_iter);
        return (abs_f - q) * (2 * q + p * (abs_f - q)) / (2 * sigma * p);
    }
}
"""


# --- Proximal operator --- #


def varlp_prox_factor_gpuary(abs_f, p, sigma, num_newton_iter, out=None):
    """Multiplicative factor for the variable Lp cc prox, GpuArray version.

    abs_f : `array-like`
        Magnitude of the input function (scalar or vectorial) to the
        proximal.
    p : `array-like`
        Spatially varying exponent of the Lp modular. Must have same
        shape and dtype as ``abs_f``.
    sigma : positive float
        Step-size-like parameter of the proximal.
    num_newton_iter : positive int
        Number of Newton iterations to perform for the places where
        ``1 < p < 2``.
    out : `pygpu.gpuarray.GpuArray`, optional
        Array where the result should be stored. Its ``shape`` and
        ``dtype`` must match those of ``abs_f``.

    Returns
    -------
    out : `pygpu.gpuarray.GpuArray`
        Factor for the proximal operator.
        If ``out`` was provided, the returned object is a reference to it.

    Examples
    --------
    When ``abs_f == 0``, the returned value is always 0.
    Otherwise, exponent ``p = 1`` gives ``max(1.0 - sigma / abs_f, 0.0)``:

    >>> abs_f = pygpu.gpuarray.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = pygpu.gpuarray.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> sigma = 1.0
    >>> result = varlp_prox_factor_gpuary(abs_f, p1, sigma,
    ...                                   num_newton_iter=1)
    >>> np.allclose(result, [0, 0, 0, 1.0 / 3.0, 0.5])
    True
    >>> sigma = 0.5
    >>> result = varlp_prox_factor_gpuary(abs_f, p1, sigma,
    ...                                   num_newton_iter=1)
    >>> np.allclose(result, [0, 0, 0.5, 2.0 / 3.0, 0.75])
    True

    With ``p = 2`` one gets ``1 / (1 + 2 * sigma)``:

    >>> p2 = pygpu.gpuarray.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> sigma = 0.5
    >>> result = varlp_prox_factor_gpuary(abs_f, p2, sigma,
    ...                                   num_newton_iter=1)
    >>> np.allclose(result, [0] + [0.5] * 4)
    True

    For other ``p`` values, the result times ``abs_f`` solves the
    equation ``v + sigma * p * v**(p-1) = abs_f``:

    >>> p15 = pygpu.gpuarray.array([1.5, 1.5, 1.5, 1.5, 1.5])
    >>> sigma = 1.0
    >>> result = varlp_prox_factor_gpuary(abs_f, p15, sigma,
    ...                                   num_newton_iter=10)
    >>> v = np.asarray(result) * abs_f
    >>> p = np.asarray(p15)
    >>> lhs = v + sigma * v ** (p - 1) * p
    >>> np.allclose(lhs, abs_f)
    True
    """
    ctx = pygpu.get_default_context()
    assert ctx is not None

    abs_f = pygpu.gpuarray.array(abs_f, copy=False)
    p = pygpu.gpuarray.array(p, copy=False)
    assert abs_f.dtype in (np.dtype('float32'), np.dtype('float64'))
    if out is None:
        out = abs_f._empty_like_me()
    sigma = float(sigma)
    num_newton_iter = int(num_newton_iter)
    args = [abs_f, p, sigma, num_newton_iter]
    argnames = ['abs_f', 'p', 'sigma', 'num_newton_iter']

    # Render the preamble code from the mako template using the specific
    # definitions of dtype, maximum and power.
    if abs_f.dtype == np.dtype('float32') and ctx.kind == b'opencl':
        raise NotImplementedError("OpenCL kernels currently not supported "
                                  "for 'float32' data type")

    # Render the preamble source from templates
    pre_tpl = mako.template.Template(newton_tpl_str + prox_tpl_str)
    power = 'powf' if abs_f.dtype == np.dtype('float32') else 'pow'
    minimum = 'fminf' if abs_f.dtype == np.dtype('float32') else 'fmin'
    maximum = 'fmaxf' if abs_f.dtype == np.dtype('float32') else 'fmax'
    preamble = pre_tpl.render(dtype=DTYPE_TO_CTYPE[abs_f.dtype],
                              maximum=maximum, minimum=minimum, power=power)

    # Define the elementwise expression
    expr = ('out = varlp_prox_factor(abs_f, p, sigma, num_newton_iter)')
    return elemwise(args, argnames, expr, preamble, out, 'out')


# --- Proximal operator of the convex conjugate --- #


def varlp_cc_prox_factor_gpuary(abs_f, p, sigma, num_newton_iter, out=None):
    """Multiplicative factor for the variable Lp cc prox, GpuArray version.

    abs_f : `array-like`
        Magnitude of the input function (scalar or vectorial) to the
        proximal.
    p : `array-like`
        Spatially varying exponent of the Lp modular. Must have same
        shape and dtype as ``abs_f``.
    sigma : positive float
        Step-size-like parameter of the proximal.
    num_newton_iter : positive int
        Number of Newton iterations to perform for the places where
        ``1 < p < 2``.
    out : `pygpu.gpuarray.GpuArray`, optional
        Array where the result should be stored. Its ``shape`` and
        ``dtype`` must match those of ``abs_f``.

    Returns
    -------
    out : `pygpu.gpuarray.GpuArray`
        Factor for the proximal operator of the convex conjugate.
        If ``out`` was provided, the returned object is a reference to it.

    Examples
    --------
    When ``abs_f == 0``, the returned value is always 0.
    Otherwise, exponent ``p = 1`` gives ``min(1, 1 / abs_f)``:

    >>> abs_f = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> sigma = 1.0
    >>> result = varlp_cc_prox_factor_gpuary(abs_f, p1, sigma,
    ...                                      num_newton_iter=1)
    >>> np.allclose(result, [0, 1, 1, 2.0 / 3.0, 0.5])
    True

    With ``p = 2`` one gets ``2 / (2 + sigma)``:

    >>> p2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> sigma = 0.5
    >>> result = varlp_cc_prox_factor_gpuary(abs_f, p2, sigma,
    ...                                      num_newton_iter=1)
    >>> np.allclose(result, [0] + [0.8] * 4)
    True

    For other ``p`` values, the result is ``1 - v / abs_f``, where ``v``
    satisfies the equation ``v + sigma**(1-p) * p * v**(p-1) = abs_f``:

    >>> p15 = pygpu.gpuarray.array([1.5, 1.5, 1.5, 1.5, 1.5])
    >>> sigma = 1.0
    >>> result = varlp_cc_prox_factor_gpuary(abs_f, p15, sigma,
    ...                                      num_newton_iter=10)
    >>> v = (1 - np.asarray(result)) * abs_f
    >>> p = np.asarray(p15)
    >>> lhs = v + sigma ** (1 - p) * p * v ** (p - 1)
    >>> np.allclose(lhs, abs_f)
    True
    """
    ctx = pygpu.get_default_context()
    assert ctx is not None

    abs_f = pygpu.gpuarray.array(abs_f, copy=False)
    p = pygpu.gpuarray.array(p, copy=False)
    assert abs_f.dtype in (np.dtype('float32'), np.dtype('float64'))
    if out is None:
        out = abs_f._empty_like_me()
    sigma = float(sigma)
    num_newton_iter = int(num_newton_iter)
    args = [abs_f, p, sigma, num_newton_iter]
    argnames = ['abs_f', 'p', 'sigma', 'num_newton_iter']

    # Render the preamble code from the mako template using the specific
    # definitions of dtype, maximum and power.
    if abs_f.dtype == np.dtype('float32') and ctx.kind == b'opencl':
        raise NotImplementedError("OpenCL kernels currently not supported "
                                  "for 'float32' data type")

    # Render the preamble source from templates
    pre_tpl = mako.template.Template(newton_tpl_str + cc_prox_tpl_str)
    power = 'powf' if abs_f.dtype == np.dtype('float32') else 'pow'
    minimum = 'fminf' if abs_f.dtype == np.dtype('float32') else 'fmin'
    maximum = 'fmaxf' if abs_f.dtype == np.dtype('float32') else 'fmax'
    preamble = pre_tpl.render(dtype=DTYPE_TO_CTYPE[abs_f.dtype],
                              maximum=maximum, minimum=minimum, power=power)

    # Define the elementwise expression
    expr = ('out = varlp_cc_prox_factor(abs_f, p, sigma, num_newton_iter)')
    return elemwise(args, argnames, expr, preamble, out, 'out')


# --- Integrand of the convex conjugate --- #


def varlp_cc_integrand_gpuary(abs_f, p, out=None):
    """Integrand for the variable Lp convex conjugate, GpuArray version.

    abs_f : `array-like`
        Magnitude of the input function (scalar or vectorial) to the
        functional.
    p : `array-like`
        Spatially varying exponent of the Lp modular. Must have same
        shape and dtype as ``abs_f``.
    out : `pygpu.gpuarray.GpuArray`, optional
        Array where the result should be stored. Its ``shape`` and
        ``dtype`` must match those of ``abs_f``.

    Returns
    -------
    out : `pygpu.gpuarray.GpuArray`
        Integrand of the convex conjugate.
        If ``out`` was provided, the returned object is a reference to it.

    Examples
    --------
    Exponent ``p = 1`` gives the indicator of the unit ball:

    >>> abs_f = pygpu.gpuarray.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = pygpu.gpuarray.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> result = varlp_cc_integrand_gpuary(abs_f, p1)
    >>> np.allclose(result, [0, 0, 0, np.inf, np.inf])
    True

    With ``p = 2`` one gets ``abs_f ** 2 / 4``:

    >>> p2 = pygpu.gpuarray.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> result = varlp_cc_integrand_gpuary(abs_f, p2)
    >>> np.allclose(result, np.asarray(abs_f) ** 2 / 4)
    True

    For other ``p`` values, the result is ``abs_f**(p/(p-1)) * r``,
    where ``r = p**(-1/(p-1)) - p**(-p/(p-1))``:

    >>> p15 = pygpu.gpuarray.array([1.5, 1.5, 1.5, 1.5, 1.5])
    >>> result = varlp_cc_integrand_gpuary(abs_f, p15)
    >>> p = np.asarray(p15)
    >>> r = p ** (-1 / (p - 1)) - p ** (-p / (p - 1))
    >>> np.allclose(result, np.asarray(abs_f) ** (p / (p - 1)) * r)
    True
    """
    ctx = pygpu.get_default_context()
    assert ctx is not None

    abs_f = pygpu.gpuarray.array(abs_f, copy=False)
    p = pygpu.gpuarray.array(p, copy=False)
    assert abs_f.dtype in (np.dtype('float32'), np.dtype('float64'))
    if out is None:
        out = abs_f._empty_like_me()
    args = [abs_f, p]
    argnames = ['abs_f', 'p']

    # Render the preamble code from the mako template using the specific
    # definitions of dtype, maximum and power.
    if abs_f.dtype == np.dtype('float32') and ctx.kind == b'opencl':
        raise NotImplementedError("OpenCL kernels currently not supported "
                                  "for 'float32' data type")

    # Render the preamble source from templates
    pre_tpl = mako.template.Template(cc_integr_tpl_str)
    power = 'powf' if abs_f.dtype == np.dtype('float32') else 'pow'
    preamble = pre_tpl.render(dtype=DTYPE_TO_CTYPE[abs_f.dtype],
                              power=power)

    # Define the elementwise expression
    expr = ('out = varlp_cc_integrand(abs_f, p)')
    return elemwise(args, argnames, expr, preamble, out, 'out')


# --- Integrand of the Moreau envelope --- #


def varlp_moreau_integrand_gpuary(abs_f, p, sigma, num_newton_iter, out=None):
    """Integrand of the variable Lp Moreau envelope, GpuArray version.

    abs_f : `array-like`
        Magnitude of the input function (scalar or vectorial) to the
        functional.
    p : `array-like`
        Spatially varying exponent of the Lp modular. Must have same
        shape and dtype as ``abs_f``.
    sigma : positive float
        Step-size-like parameter of the envlope.
    num_newton_iter : positive int
        Number of Newton iterations to perform for the places where
        ``1 < p < 2``.
    out : `pygpu.gpuarray.GpuArray`, optional
        Array where the result should be stored. Its ``shape`` and
        ``dtype`` must match those of ``abs_f``.

    Returns
    -------
    out : `pygpu.gpuarray.GpuArray`
        Factor for the proximal operator of the convex conjugate.
        If ``out`` was provided, the returned object is a reference to it.

    Examples
    --------
    Exponent ``p = 1`` gives the Huber function of ``abs_f``, that is
    ``abs_f ** 2 / (2 * sigma)`` if ``abs_f <= sigma`` and
    ``abs_f - sigma / 2`` otherwise:

    >>> abs_f = pygpu.gpuarray.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> p1 = pygpu.gpuarray.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> sigma = 1.0
    >>> result = varlp_moreau_integrand_gpuary(abs_f, p1, sigma,
    ...                                        num_newton_iter=1)
    >>> np.allclose(result, [0, 0.125, 0.5, 1.0, 1.5])
    True
    >>> sigma = 0.5
    >>> result = varlp_moreau_integrand_gpuary(abs_f, p1, sigma,
    ...                                        num_newton_iter=1)
    >>> np.allclose(result, [0, 0.25, 0.75, 1.25, 1.75])
    True

    With ``p = 2`` one gets ``abs_f ** 2 / (1 + 2 * sigma)``:

    >>> p2 = pygpu.gpuarray.array([2.0, 2.0, 2.0, 2.0, 2.0])
    >>> sigma = 0.5
    >>> result = varlp_moreau_integrand_gpuary(abs_f, p2, sigma,
    ...                                        num_newton_iter=1)
    >>> np.allclose(result, [0, 0.125, 0.5, 1.125, 2])
    True
    """
    ctx = pygpu.get_default_context()
    assert ctx is not None

    abs_f = pygpu.gpuarray.array(abs_f, copy=False)
    p = pygpu.gpuarray.array(p, copy=False)
    assert abs_f.dtype in (np.dtype('float32'), np.dtype('float64'))
    if out is None:
        out = abs_f._empty_like_me()
    sigma = float(sigma)
    num_newton_iter = int(num_newton_iter)
    args = [abs_f, p, sigma, num_newton_iter]
    argnames = ['abs_f', 'p', 'sigma', 'num_newton_iter']

    # Render the preamble code from the mako template using the specific
    # definitions of dtype, maximum and power.
    if abs_f.dtype == np.dtype('float32') and ctx.kind == b'opencl':
        raise NotImplementedError("OpenCL kernels currently not supported "
                                  "for 'float32' data type")

    # Render the preamble source from templates
    pre_tpl = mako.template.Template(newton_tpl_str + moreau_integr_tpl_str)
    power = 'powf' if abs_f.dtype == np.dtype('float32') else 'pow'
    minimum = 'fminf' if abs_f.dtype == np.dtype('float32') else 'fmin'
    maximum = 'fmaxf' if abs_f.dtype == np.dtype('float32') else 'fmax'
    preamble = pre_tpl.render(dtype=DTYPE_TO_CTYPE[abs_f.dtype],
                              maximum=maximum, minimum=minimum, power=power)

    # Define the elementwise expression
    expr = ('out = varlp_moreau_integrand(abs_f, p, sigma, num_newton_iter)')
    return elemwise(args, argnames, expr, preamble, out, 'out')


# --- Elemwise helper function --- #


def elemwise(args, argnames, expr, preamble, out, outname):
    """Run an elemwise kernel.

    Parameters
    ----------
    args : sequence of `array-like`'s or scalars
        Input arguments to the kernel. Arrays need to have identical
        shapes. Arrays of type `pygpu.gpuarray.GpuArray` must have
        identical contexts and the same context as ``out``.
    argnames : sequence of strings
        Argument names as used in the kernel code. The length of the
        sequence must coincide with the one of ``args``.
    expr : str
        String defining an expression that should be evaluated for
        all indices.
    preamble : str
        Source that should be prepended to the kernel code.
    out : `pygpu.gpuarray.GpuArray`
        Array where the result should be stored. Its ``shape`` must match
        that of the arrays in ``args``.
    outname : str
        Name of the output argument in ``expr``.

    Returns
    -------
    out : `pygpu.gpuarray.GpuArray`
        Result of the kernel evaluation.
        If ``out`` was provided, the returned object is a reference to it.
    """
    if not isinstance(out, pygpu.gpuarray.GpuArray):
        raise TypeError('`out` must be a `GpuArray` instance, got {!r}'
                        ''.format(out))

    if not all(arg.context == out.context for arg in args
               if hasattr(arg, 'context')):
        raise ValueError('all `GpuArray` objects must have the same '
                         '`context`')

    if not all(arg.shape == out.shape for arg in args
               if hasattr(arg, 'shape')):
        raise ValueError('all `GpuArray` objects must have the same '
                         '`shape`')

    # Check contiguouness to see if we can flatten the arrays
    if (out.flags.c_contiguous and
            all(arg.flags.c_contiguous for arg in args
                if hasattr(arg, 'flags'))):
        order = 'C'
    elif (out.flags.f_contiguous and
            all(arg.flags.f_contiguous for arg in args
                if hasattr(arg, 'flags'))):
        order = 'F'
    else:
        order = None

    # Prepare arguments for the kernel. out has to come first.
    if order is not None:
        out, out_orig = out.reshape(-1, order=order), out
    else:
        out_orig = out
    out_arg = pygpu._elemwise.arg(outname, out.dtype, scalar=False,
                                  read=False, write=True)

    conv_args = [out]
    kernel_args = [out_arg]
    for name, arg in zip(argnames, args):
        if np.isscalar(arg):
            conv_args.append(arg)
            kernel_args.append(
                pygpu._elemwise.arg(name, np.array(arg).dtype,
                                    scalar=True, read=True, write=False))
        else:
            array = pygpu.gpuarray.array(arg, copy=False, context=out.context,
                                         order=order)
            if order is not None:
                array = array.reshape(-1, order=order)
            conv_args.append(array)
            kernel_args.append(
                pygpu._elemwise.arg(name, array.dtype,
                                    scalar=False, read=True, write=False))

    # Initialize and run the kernel
    kernel = pygpu.elemwise.GpuElemwise(out.context, expr, kernel_args,
                                        preamble=preamble)
    kernel(*conv_args)
    return out_orig


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
