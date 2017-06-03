# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Factory functions for creating proximal operators.

Functions with ``cconj`` mean the proximal of the convex conjugate and are
provided for convenience.

For more details see :ref:`proximal_operators` and references therein. For
more details on proximal operators including how to evaluate the proximal
operator of a variety of functionals see [PB2014]_. """

from __future__ import print_function, division, absolute_import

from odl.discr import DiscreteLp
from odl.operator import Operator, PointwiseNorm
from odl.space import ProductSpace
from odl.util import writable_array
from variable_lp import _cython_impl, _numba_impl, _numpy_impl, _gpuarray_impl


__all__ = ('proximal_variable_lp_modular',
           'proximal_cconj_variable_lp_modular')


def _varlp_prox_call(f, out, exponent, sigma, impl, num_newton_iter, conj):
    """Implementation of the proximals of the modular and its conjugate.

    Parameters
    ----------
    f : domain element
        Element at which to evaluate the operator.
    out : range element
        Element to which the result is written.
    exponent : base space element
        Variable exponent used in the modular.
    sigma : positive float
        Scaling parameter in the proximal operator.
    impl : string
        Implementation back-end for the proximal operator.
    num_newton_iter : int, optional
        Number of Newton iterations.
    conj : bool
        Apply the proximal of the convex conjugate if ``True``, otherwise
        the proximal of the variable Lp modular.
    """
    num_newton_iter = int(num_newton_iter)
    sigma = float(sigma)

    # Compute the magnitude of the input function
    if isinstance(f.space, ProductSpace):
        pwnorm = PointwiseNorm(f.space)
        prox_fac = pwnorm(f)
    else:
        prox_fac = out
        f.ufuncs.absolute(out=prox_fac)

    if conj:
        prox_npy = _numpy_impl.varlp_cc_prox_factor_npy
        prox_numba = _numba_impl.varlp_cc_prox_factor_numba
        prox_cython = _cython_impl.varlp_cc_prox_factor_cython
        prox_gpuary = _gpuarray_impl.varlp_cc_prox_factor_gpuary
    else:
        prox_npy = _numpy_impl.varlp_prox_factor_npy
        prox_numba = _numba_impl.varlp_prox_factor_numba
        prox_cython = _cython_impl.varlp_prox_factor_cython
        prox_gpuary = _gpuarray_impl.varlp_prox_factor_gpuary

    # Compute the multiplicative factor
    if impl == 'numpy':
        with writable_array(prox_fac) as out_arr:
            prox_npy(prox_fac.asarray(), exponent.asarray(), sigma,
                     num_newton_iter, out=out_arr)

    elif impl.startswith('numba'):
        _, target = impl.split('_')
        prox_fac[:] = prox_numba(prox_fac.asarray(), exponent.asarray(),
                                 sigma, num_newton_iter, target=target)

    elif impl == 'cython':
        if not _cython_impl.CYTHON_EXTENSION_BUILT:
            raise RuntimeError(
                'Cython extension has not been built. '
                'Run `python setup.py build_ext` to build the extension.'
                'For development installations of this package '
                '(`pip install -e`), add the `--inplace` option to the '
                'build command.')
        with writable_array(prox_fac) as out_arr:
            prox_cython(prox_fac.asarray(), exponent.asarray(), sigma,
                        num_newton_iter, out=out_arr)

    elif impl == 'gpuarray':
        if not _gpuarray_impl.PYGPU_AVAILABLE:
            raise RuntimeError(
                '`pygpu` package not installed. You need Anaconda '
                '(or Miniconda) to install it via `conda install pygpu`. '
                'Alternatively, you can build it from source. See '
                'https://github.com/Theano/libgpuarray for more information.')

        # TODO: context manager for GPU arrays
        if isinstance(f.space, ProductSpace):
            dom_impl = f.space[0].impl
        else:
            dom_impl = f.space.impl

        if dom_impl == 'gpuarray':
            out_arr = prox_fac.tensor.data
            prox_fac_arr = prox_fac.tensor.data
        else:
            out_arr = None
            prox_fac_arr = prox_fac.asarray()

        result = prox_gpuary(prox_fac_arr, exponent.asarray(), sigma,
                             num_newton_iter, out=out_arr)

        if out_arr is None:
            prox_fac[:] = result

    else:
        raise RuntimeError('bad impl {!r}'.format(impl))

    # Perform the multiplication
    if isinstance(f.space, ProductSpace):
        # out is empty
        out.assign(f)
        out *= prox_fac
    else:
        # out already stores the factor
        out *= f

    return f


def proximal_variable_lp_modular(space, exponent, lam=1.0, g=None,
                                 impl='numpy'):
    """Return the proximal operator of the variable Lebesgue modular.

    Parameters
    ----------
    space : `DiscreteLp` or power space of such
        Space on which the proximal operator acts.
    exponent : `array-like`
        Variable exponent used in the modular.
    lam : positive float
        Scaling factor or regularization parameter.
    g : ``space`` element-like
        An element in ``space``
    impl : str
        Implementation back-end for the proximal operator.
        Supported:

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

    Notes
    -----
    For :math:`\Omega \subset \mathbb{R}^d` and an exponent mapping
    :math:`p:\Omega \\to [0, 2]`, the proximal of the variable
    :math:`L^p` modular can be explicitly computed. It is a point-wise
    operator, given as the multiplication with a factor:
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
    impl, impl_in = str(impl).lower(), impl
    if impl not in ('numpy', 'cython', 'numba_cpu', 'numba_cuda',
                    'numba_parallel', 'gpuarray'):
        raise ValueError("`impl` '{}' not understood".format(impl_in))

    class VarLpModularProx(Operator):

        """Proximal operator of the variable Lebesgue modular."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Scaling parameter in the proximal operator.
            """
            if isinstance(space, ProductSpace) and space.is_power_space:
                self.exponent = space[0].element(exponent)
            elif isinstance(space, DiscreteLp):
                self.exponent = space.element(exponent)
            else:
                raise TypeError('space must be a `DiscreteLp` instance or '
                                'a power space of those, got {!r}'
                                ''.format(space))

            if g is not None:
                self.g = self.domain.element(g)
            else:
                self.g = None

            Operator.__init__(self, domain=space, range=space, linear=False)
            self.sigma = float(sigma)
            self.impl = impl

        def _call(self, f, out, **kwargs):
            """Implement ``self(x, out, **kwargs)``.

            Parameters
            ----------
            f : domain element
                Element at which to evaluate the operator
            out : range element
                Element to which the result is written
            num_newton_iter : int, optional
                Number of Newton iterations.
                Default: 10
            """
            num_newton_iter = int(kwargs.pop('num_newton_iter', 5))
            step = self.sigma * float(lam)
            if self.g is not None:
                f = f - self.g

            _varlp_prox_call(f, out, self.exponent, step, self.impl,
                             num_newton_iter, conj=False)

            return out

    return VarLpModularProx


def proximal_cconj_variable_lp_modular(space, exponent, lam=1.0, g=None,
                                       impl='numpy'):
    """Return the proximal of the variable Lebesgue modular conjugate.

    Parameters
    ----------
    space : `DiscreteLp` or power space of such
        Space on which the proximal operator acts.
    exponent : `array-like`
        Variable exponent used in the modular.
    lam : positive float
        Scaling factor or regularization parameter.
    g : ``space`` element-like
        An element in ``space``
    impl : str
        Implementation back-end for the proximal operator.
        Supported:

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

    Notes
    -----
    For :math:`\Omega \subset \mathbb{R}^d` and an exponent mapping
    :math:`p:\Omega \\to [0, 2]`, the proximal of the variable
    :math:`L^p` modular can be explicitly computed. It is a point-wise
    operator, given as the multiplication with a factor:
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
    impl, impl_in = str(impl).lower(), impl
    if impl not in ('numpy', 'cython', 'numba_cpu', 'numba_cuda',
                    'numba_parallel', 'gpuarray'):
        raise ValueError("`impl` '{}' not understood".format(impl_in))

    class VarLpModularCconjProx(Operator):

        """Proximal of the variable Lebesgue modular convex conjugate."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Scaling parameter in the proximal operator.
            """
            if isinstance(space, ProductSpace) and space.is_power_space:
                self.exponent = space[0].element(exponent)
            elif isinstance(space, DiscreteLp):
                self.exponent = space.element(exponent)
            else:
                raise TypeError('space must be a `DiscreteLp` instance or '
                                'a power space of those, got {!r}'
                                ''.format(space))

            if g is not None:
                self.g = self.domain.element(g)
            else:
                self.g = None

            Operator.__init__(self, domain=space, range=space, linear=False)
            self.sigma = float(sigma)
            self.impl = impl

        def _call(self, f, out, **kwargs):
            """Implement ``self(x, out, **kwargs)``.

            Parameters
            ----------
            f : domain element
                Element at which to evaluate the operator
            out : range element
                Element to which the result is written
            num_newton_iter : int, optional
                Number of Newton iterations.
                Default: 10
            """
            num_newton_iter = int(kwargs.pop('num_newton_iter', 5))
            step = self.sigma * float(lam)
            if self.g is not None:
                f = f - self.g

            _varlp_prox_call(f, out, self.exponent, step, self.impl,
                             num_newton_iter, conj=True)

            return out

    return VarLpModularCconjProx


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
