# Author: Pierre Ablin <pierreablin@gmail.com>
# License: MIT

import autograd.numpy as np
from numpy.testing import assert_allclose

from autoptim import minimize


def test_rosenbrock():
    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    x0 = np.zeros(2)

    x_min, res = minimize(rosenbrock, x0)
    assert res['success']


def test_multiple_shapes():
    def f(x, y, z, a):
        return np.sum(x ** 2) + np.sum((y - 3) ** 2) + np.sum((z + a) ** 4)

    a = 2
    shapes = [(2, 3), (2, 2), (3,)]
    optim_vars_init = [np.ones(shape) for shape in shapes]
    optim_vars, res = minimize(f, optim_vars_init, args=(a,))
    assert res['success']
    assert [var.shape for var in optim_vars] == shapes
    for var, target in zip(optim_vars, [0, 3, -a]):
        assert_allclose(var, target, atol=1e-1)
