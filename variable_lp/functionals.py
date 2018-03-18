# Copyright 2017, 2018 Holger Kohr
#
# This file is part of variable_lp_paper.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Functionals related to variable Lebesgue spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

from odl.discr import DiscreteLp
from odl.operator import Operator, PointwiseNorm
from odl.space import ProductSpace
from odl.solvers import Functional
from variable_lp import _cython_impl, _numba_impl, _numpy_impl, _gpuarray_impl
from variable_lp.proximal_operators import (
    proximal_variable_lp_modular, proximal_cconj_variable_lp_modular)
from odl.util import writable_array


__all__ = ('VariableLpModular', 'VariableLpModularConvexConj')


class VariableLpModular(Functional):

    """The variable Lp modular.

    This functional is the generalization of the p-th power ``||f||_p^p``
    of the Lp-norm  for an exponent ``p`` that varies with the
    spatial location.

    Notes
    -----
    For :math:`\Omega \subset \mathbb{R}^d`, a scalar- or
    vector-valued function :math:`f: \Omega \\to \mathbb{R}^m` and an
    exponent function :math:`p: \Omega \\to [1, 2]`, the
    :math:`p`-modular is defined as

    .. math::
        \\rho_p(f) = \int_\Omega |f(x)|^{p(x)} \mathrm{d}x.

    Its proximal is a point-wise operator, given as the multiplication
    with a factor:
    :math:`\\text{prox}_{\sigma\\rho_p}(f) = u \cdot f`, where

    .. math::
        :nowrap:

        \\begin{equation*}
            u(x) =
            \\begin{cases}
                0, &
                    \\text{if } f(x) = 0  \\\\
                \\frac{\max\\left\{|f(x)| - \sigma, 0\\right\}}{|f(x)|}, &
                    \\text{if } f(x) \\neq 0 \\text{ and } p(x) = 1, \\\\
                \\frac{1}{1 + 2\sigma}, &
                    \\text{if } f(x) \\neq 0 \\text{ and } p(x) = 2, \\\\
                \\frac{q}{|f(x)|}, \\text{ where } q > 0 \\text{ solves} \\\\
                q + p(x) \sigma q^{p(x) - 1} = |f(x)|, &
                    \\text{ otherwise}.
            \end{cases}
        \end{equation*}
    """

    def __init__(self, space, exponent, impl='numpy'):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or power space of such
            Discretized function space on which the modular is defined.
        exponent : `array-like`
            The variable exponent ``p(x)``. Its shape and data type
            must be compatible with (the base space of) ``space``.
            All values must lie between 1 and 2 (not checked).
        impl : str
            Implementation back-end for the proximal operator, the convex
            conjugate and its proximal. Supported:

            - ``'numpy'`` : Reference Numpy implementation.
            - ``'numba_<target>'`` : Accelerated implementation using the
              Numba JIT for a specific ``<target>``, which can be
              ``'cpu'``, ``'parallel'`` or ``'cuda'``.
              Requires the `numba <http://numba.pydata.org/>`_ package.
            - ``'cython'`` : Native C implementation using the
              `Cython <http://cython.org/>`_ extension builder.
              In case of an ``ImportError``, the extension module must
              be built by issuing ::

                  python setup.py build_ext --inplace

              in the top-level directory.
            - ``'gpuarray'`` : Native GPU kernel code running on a
              `pygpu.gpuarray.GpuArray` object. This is most efficient
              if ``space.impl == 'gpuarray'`` as well.
        """
        if isinstance(space, DiscreteLp):
            self.exponent = space.element(exponent)
        elif isinstance(space, ProductSpace) and space.is_power_space:
            self.exponent = space[0].element(exponent)
        else:
            raise TypeError('`space` must be a `DiscreteLp` instance or '
                            'a power space of such, got {!r}'
                            ''.format(space))
        super().__init__(space, linear=False)
        self.impl, impl_in = str(impl).lower(), impl
        if impl not in ('numpy', 'cython', 'numba_cpu', 'numba_cuda',
                        'numba_parallel', 'gpuarray'):
            raise ValueError("`impl` '{}' not understood".format(impl_in))

    def _call(self, f):
        """Return ``self(f)``."""
        integrand = f.ufuncs.absolute()
        integrand = integrand.ufuncs.power(self.exponent, out=integrand)
        return self.domain.inner(integrand, self.domain.one())

    @property
    def gradient(self):
        """Gradient of the variable Lp modular.

        Notes
        -----
        The gradient of the modular :math:`\\rho_p` for exponent
        :math:`p : \Omega \\to [1, 2]` is computed using the formula

        .. math::
            \\nabla \\rho_p(f) = p\, |f|^(p-2)\, f

        This expression is invalid in :math:`f = 0`, since the functional
        is not differentiable in this point. This implementation
        puts a value of 0 there.
        """
        functional = self

        class VarLpModularGrad(Operator):

            """Gradient of the variable Lp modular."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(domain=functional.domain,
                                 range=functional.domain,
                                 linear=False)

            def _call(self, f, out=None):
                """Evaluate the gradient in the point ``f``."""
                if functional.impl != 'numpy':
                    raise NotImplementedError(
                        'gradient not implemented for `impl` {!r}'
                        ''.format(functional.impl))
                if isinstance(self.domain, ProductSpace):
                    tmp = self.domain[0].element()
                    pwnorm = PointwiseNorm(self.domain)
                    pwnorm(f, out=tmp)
                else:
                    tmp = out
                    f.ufunc.absolute(out=tmp)

                tmp.ufunc.power(self.exponent - 2, out=tmp)
                with writable_array(tmp) as tmp_arr:
                    tmp_arr[np.isnan(tmp_arr)] = 0  # Handle NaN

                if out is None:
                    out = f.copy()
                else:
                    out.assign(f)

                out *= self.exponent
                return out

        return VarLpModularGrad()

    @property
    def convex_conj(self):
        """Convex conjugate of the variable Lp modular."""
        return VariableLpModularConvexConj(self.domain, self.exponent,
                                           self.impl)

    @property
    def proximal(self):
        """Proximal operator factory of the variable Lp modular."""
        return proximal_variable_lp_modular(
            space=self.domain, exponent=self.exponent, impl=self.impl)


class VariableLpModularConvexConj(Functional):

    """The convex conjugate of the variable Lp modular.

    Notes
    -----
    For :math:`\Omega \subset \mathbb{R}^d`, a scalar- or
    vector-valued function :math:`f: \Omega \\to \mathbb{R}^m` and an
    exponent function :math:`p: \Omega \\to [1, 2]`, the
    convex conjugate of the :math:`p`-modular is defined as

    .. math::
        \\rho_p^*(f) = \int_{\Omega^*} R\\big(f(x), p(x)\\big) \mathrm{d}x,

    where

    .. math::
        :nowrap:

        \\begin{equation*}
            R(z, p) =
            \\begin{cases}
                \\iota_{|\cdot|\leq 1}(z), &
                    \\text{if } p = 1  \\\\
                \\frac{|z|^2}{4}, &
                    \\text{if } p = 2, \\\\
                |z|^{\\frac{p}{p-1}}
                \\left(p^{-\\frac{1}{p-1}} - p^{-\\frac{p}{p-1}} \\right), &
                    \\text{ otherwise}.
            \end{cases}
        \end{equation*}

    Here, :math:`\\iota_A` is the indicator function of a set :math:`A`,
    i.e., the function that takes the value 0 in :math:`A` and
    :math:`+\\infty` outside.

    The proximal operator is a point-wise operator, given as the
    multiplication with a factor:
    :math:`\\text{prox}_{\sigma\\rho_p^*}(f) = u \cdot f`, where

    .. math::
        :nowrap:

        \\begin{equation*}
            u(x) =
            \\begin{cases}
                0, &
                    \\text{if } f(x) = 0  \\\\
                \max\\left\{1 - \\frac{1}{|f(x)|}, 0\\right\}, &
                    \\text{if } f(x) \\neq 0 \\text{ and } p(x) = 1, \\\\
                \\frac{\sigma}{\sigma + 2}, &
                    \\text{if } f(x) \\neq 0 \\text{ and } p(x) = 2, \\\\
                1 - \\frac{q}{|f(x)|}, \\text{ where } q>0 \\text{ solves} \\\\
                q + p(x) \sigma^{1-p(x)} q^{p(x) - 1} = |f(x)|, &
                    \\text{ otherwise}.
            \end{cases}
        \end{equation*}

    """

    def __init__(self, space, exponent, impl='numpy'):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or power space of such
            Discretized function space on which the modular is defined.
        exponent : `array-like`
            The variable exponent ``p(x)``. Its shape and data type
            must be compatible with (the base space of) ``space``.
            All values must lie between 1 and 2 (not checked).
        impl : str
            Implementation back-end for the proximal operator, the convex
            conjugate and its proximal. Supported:

            - ``'numpy'`` : Reference Numpy implementation.
            - ``'numba_<target>'`` : Accelerated implementation using the
              Numba JIT for a specific ``<target>``, which can be
              ``'cpu'``, ``'parallel'`` or ``'cuda'``.
              Requires the `numba <http://numba.pydata.org/>`_ package.
            - ``'cython'`` : Native C implementation using the
              `Cython <http://cython.org/>`_ extension builder.
              In case of an ``ImportError``, the extension module must
              be built by issuing ::

                  python setup.py build_ext --inplace

              in the top-level directory.
            - ``'gpuarray'`` : Native GPU kernel code running on a
              `pygpu.gpuarray.GpuArray` object. This is most efficient
              if ``space.impl == 'gpuarray'`` as well.
        """
        if isinstance(space, DiscreteLp):
            self.exponent = space.element(exponent)
        elif isinstance(space, ProductSpace) and space.is_power_space:
            self.exponent = space[0].element(exponent)
        else:
            raise TypeError('`space` must be a `DiscreteLp` instance or '
                            'a power space of such, got {!r}'
                            ''.format(space))
        super().__init__(space, linear=False)
        self.impl, impl_in = str(impl).lower(), impl
        if impl not in ('numpy', 'cython', 'numba_cpu', 'numba_cuda',
                        'numba_parallel', 'gpuarray'):
            raise ValueError("`impl` '{}' not understood".format(impl_in))

    def _call(self, f):
        """Return ``self(f)``."""
        # Compute the magnitude of the input function
        if isinstance(self.domain, ProductSpace):
            pwnorm = PointwiseNorm(self.domain)
            integrand = pwnorm(f)
        else:
            integrand = f.ufuncs.absolute()

        # Compute the integrand
        if self.impl == 'numpy':
            with writable_array(integrand) as out_arr:
                _numpy_impl.varlp_cc_integrand_npy(
                    integrand.asarray(), self.exponent.asarray(), out=out_arr)

        elif self.impl.startswith('numba'):
            _, target = self.impl.split('_')
            integrand[:] = _numba_impl.varlp_cc_integrand_numba(
                integrand.asarray(), self.exponent.asarray(), target=target)

        elif self.impl == 'cython':
            if not _cython_impl.CYTHON_EXTENSION_BUILT:
                raise RuntimeError(
                    'Cython extension has not been built. '
                    'Run `python setup.py build_ext` to build the extension.'
                    'For development installations of this package '
                    '(`pip install -e`), add the `--inplace` option to the '
                    'build command.')

            with writable_array(integrand) as out_arr:
                _cython_impl.varlp_cc_integrand_cython(
                    integrand.asarray(), self.exponent.asarray(), out=out_arr)

        elif self.impl == 'gpuarray':
            if not _gpuarray_impl.PYGPU_AVAILABLE:
                raise RuntimeError(
                    '`pygpu` package not installed. You need Anaconda '
                    '(or Miniconda) to install it via `conda install pygpu`. '
                    'Alternatively, you can build it from source. See '
                    'https://github.com/Theano/libgpuarray for more '
                    'information.')

            # TODO: context manager for GPU arrays
            if isinstance(self.domain, ProductSpace):
                dom_impl = self.domain[0].impl
            else:
                dom_impl = self.domain.impl

            if dom_impl == 'gpuarray':
                out_arr = integrand.data
                integrand_arr = integrand.data
            else:
                out_arr = None
                integrand_arr = integrand.asarray()

            result = _gpuarray_impl.varlp_cc_integrand_gpuary(
                integrand_arr, self.exponent.asarray(), out=out_arr)

            if out_arr is None:
                integrand[:] = result

        else:
            raise RuntimeError('bad impl {!r}'.format(self.impl))

        # Integrate
        if isinstance(self.domain, ProductSpace):
            return integrand.inner(self.domain[0].one())
        else:
            return integrand.inner(self.domain.one())

    @property
    def convex_conj(self):
        """Biconjugate of the variable Lp modular, the modular itself."""
        return VariableLpModular(self.domain, self.exponent, self.impl)

    @property
    def proximal(self):
        """Proximal operator factory of the variable Lp modular."""
        return proximal_cconj_variable_lp_modular(
            space=self.domain, exponent=self.exponent, impl=self.impl)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
