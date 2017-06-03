import numpy as np
import pygpu

import odl
from odl.solvers.nonsmooth._varlp.varlp_npy import varlp_prox_factor_npy
from odl.solvers.nonsmooth._varlp.varlp_numba import varlp_prox_factor_numba
from odl.solvers.nonsmooth._varlp.varlp_cython import varlp_prox_factor_cython
from odl.solvers.nonsmooth._varlp.varlp_gpuary import varlp_prox_factor_gpuary

size = int(1e6)
sigma = 1.0
dtype = np.dtype('float64')

abs_f = np.random.uniform(low=0, high=5, size=size).astype(dtype)
p = np.random.uniform(low=1, high=2, size=size).astype(dtype)
out = np.empty_like(abs_f)

abs_f_gpu = pygpu.gpuarray.array(abs_f)
p_gpu = pygpu.gpuarray.array(p)
out_gpu = abs_f_gpu._empty_like_me()

with odl.util.Timer('Numpy'):
    varlp_prox_factor_npy(abs_f, p, sigma, 10)

with odl.util.Timer('Numpy in-place'):
    varlp_prox_factor_npy(abs_f, p, sigma, 10, out=out)

with odl.util.Timer('Numba CPU'):
    varlp_prox_factor_numba(abs_f, p, sigma, 10, 'cpu')

with odl.util.Timer('Numba parallel'):
    varlp_prox_factor_numba(abs_f, p, sigma, 10, 'parallel')

with odl.util.Timer('Numba CUDA'):
    varlp_prox_factor_numba(abs_f, p, sigma, 10, 'cuda')

with odl.util.Timer('Cython'):
    varlp_prox_factor_cython(abs_f, p, sigma, 10)

with odl.util.Timer('Cython in-place'):
    varlp_prox_factor_cython(abs_f, p, sigma, 10, out=out)

with odl.util.Timer('GpuArray 1st time'):
    varlp_prox_factor_gpuary(abs_f, p, sigma, 10)

with odl.util.Timer('GpuArray 2nd time'):
    varlp_prox_factor_gpuary(abs_f, p, sigma, 10)

with odl.util.Timer('GpuArray in-place'):
    varlp_prox_factor_gpuary(abs_f, p, sigma, 10, out=out_gpu)

with odl.util.Timer('GpuArray without copying'):
    varlp_prox_factor_gpuary(abs_f_gpu, p, sigma, 10)
