import numpy as np

import odl
import pygpu
from variable_lp._cython_impl import varlp_cc_integrand_cython
from variable_lp._gpuarray_impl import varlp_cc_integrand_gpuary
from variable_lp._numba_impl import varlp_cc_integrand_numba
from variable_lp._numpy_impl import varlp_cc_integrand_npy

size = int(1e6)
dtype = np.dtype('float64')

abs_f = np.random.uniform(low=0, high=5, size=size).astype(dtype)
p = np.random.uniform(low=1, high=2, size=size).astype(dtype)
out = np.empty_like(abs_f)

abs_f_gpu = pygpu.gpuarray.array(abs_f)
p_gpu = pygpu.gpuarray.array(p)
out_gpu = abs_f_gpu._empty_like_me()

with odl.util.Timer('Numpy'):
    varlp_cc_integrand_npy(abs_f, p)

with odl.util.Timer('Numpy in-place'):
    varlp_cc_integrand_npy(abs_f, p, out=out)

with odl.util.Timer('Numba CPU'):
    varlp_cc_integrand_numba(abs_f, p, 'cpu')

with odl.util.Timer('Numba parallel'):
    varlp_cc_integrand_numba(abs_f, p, 'parallel')

with odl.util.Timer('Numba CUDA'):
    varlp_cc_integrand_numba(abs_f, p, 'cuda')

with odl.util.Timer('Cython'):
    varlp_cc_integrand_cython(abs_f, p)

with odl.util.Timer('Cython in-place'):
    varlp_cc_integrand_cython(abs_f, p, out=out)

with odl.util.Timer('GpuArray 1st time'):
    varlp_cc_integrand_gpuary(abs_f, p)

with odl.util.Timer('GpuArray 2nd time'):
    varlp_cc_integrand_gpuary(abs_f, p)

with odl.util.Timer('GpuArray in-place'):
    varlp_cc_integrand_gpuary(abs_f, p, out=out_gpu)

with odl.util.Timer('GpuArray without copying'):
    varlp_cc_integrand_gpuary(abs_f_gpu, p, out=out_gpu)
