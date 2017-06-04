"""Script to verify properties of the variable Lp modular.

This code checks if the proximal operator of the modular, the
convex conjugate and its proximal really solve their respective
optimization problems. The check is performed by picking a random
point, evaluating the functional/operator of interest and then
verifying that no other random point in the neighborhood has a
larger (convex conjugate) or smaller (proximal) value, respectively.
"""

import numpy as np
import odl
from odl.util import noise_element
import variable_lp

space = odl.uniform_discr(0, 1, 3)
exponent = space.element([1, 1.5, 2])

modular = variable_lp.VariableLpModular(space, exponent, impl='numba_cpu')

# Margin of error
EPS = 1e-6


# --- Check the convex conjugate --- #


def cconj_objective(functional, x, y):
    """Objective function of the convex conjugate problem."""
    return x.inner(y) - functional(x)


functional = modular
f_cconj = functional.convex_conj

# Select y randomly
y = noise_element(f_cconj.domain)
f_cconj_y = f_cconj(y)

# Test 100 other random points
for _ in range(100):

    x = noise_element(functional.domain)
    lhs = x.inner(y) - functional(x)

    if np.isnan(f_cconj_y) or np.isnan(lhs):
        print('NaN results: {} {}'.format(f_cconj_y, lhs))
        assert False

    if lhs > f_cconj_y + EPS:
        print('x:', x)
        print('y:', y)
        print('lhs:', lhs)
        print('f_cconj_y:', f_cconj_y)
        assert False


# --- Check the proximal operators --- #


def proximal_objective(functional, x, y):
    """Objective function of the proximal optimization problem."""
    return functional(y) + (1.0 / 2.0) * (x - y).norm() ** 2


# Proximal of the modular for 3 different sigma values
for sigma in [0.1, 1.0, 10.0]:
    functional = modular
    proximal = functional.proximal(sigma)

    # Select x randomly
    x = noise_element(proximal.domain) * 10
    prox_x = proximal(x)
    f_prox_x = proximal_objective(sigma * functional, x, prox_x)

    # Test 100 other random points
    for _ in range(100):
        y = noise_element(proximal.domain)
        f_y = proximal_objective(sigma * functional, x, y)

        if np.isnan(f_prox_x) or np.isnan(f_y):
            print('NaN results: {} {}'.format(f_prox_x, f_y))
            assert False

        if f_prox_x > f_y + EPS:
            print('sigma:', sigma)
            print('x:', x)
            print('y:', y)
            print('prox_x:', prox_x)
            print('f_prox_x:', f_prox_x)
            print('f_y:', f_y)
            assert False


# Proximal of the convex conjugate for 3 different sigma values
for sigma in [0.1, 1.0, 10.0]:
    functional = modular.convex_conj
    proximal = functional.proximal(sigma)

    # Select x randomly
    x = noise_element(proximal.domain) * 10
    prox_x = proximal(x)
    f_prox_x = proximal_objective(sigma * functional, x, prox_x)

    # Test 100 other random points
    for _ in range(100):
        y = noise_element(proximal.domain)
        f_y = proximal_objective(sigma * functional, x, y)

        if np.isnan(f_prox_x) or np.isnan(f_y):
            print('NaN results: {} {}'.format(f_prox_x, f_y))
            assert False

        if f_prox_x > f_y + EPS:
            print('sigma:', sigma)
            print('x:', x)
            print('y:', y)
            print('prox_x:', prox_x)
            print('f_prox_x:', f_prox_x)
            print('f_y:', f_y)
            assert False
