# Author: Pierre Ablin <pierreablin@gmail.com>
# License: MIT

import numpy as np
from numpy.testing import assert_array_equal

from .autoptim import _convert_bounds, _vectorize, _split


def test_convert_bounds():
    shapes = [(2, 2),
              (2,),
              (1, 2, 1)]
    bounds = [(None, None),
              [(None, 1), (0, 1)],
              (0, np.inf)]
    target_bounds = [(None, None), (None, None), (None, None), (None, None),
                     (None, 1), (0, 1),
                     (0, np.inf), (0, np.inf)]
    output_bounds = _convert_bounds(bounds, shapes)
    assert len(output_bounds) == np.sum([np.prod(shape) for shape in shapes])
    assert target_bounds == output_bounds


def test_vectorize():
    shapes = [(1, 2), (3, 2), (2,)]
    optim_vars = [np.arange(np.prod(shape)).reshape(*shape)
                  for shape in shapes]
    target = np.array([0, 1, 0, 1, 2, 3, 4, 5, 0, 1])
    x, output_shape = _vectorize(optim_vars)
    assert output_shape == shapes
    assert_array_equal(target, x)


def test_split():
    x = np.array([0, 1, 0, 1, 2, 3, 4, 5, 0, 1])
    shapes = [(1, 2), (3, 2), (2,)]
    target_vars = [np.arange(np.prod(shape)).reshape(*shape)
                   for shape in shapes]
    optim_vars = _split(x, shapes)
    for optim_var, target_var in zip(optim_vars, target_vars):
        assert_array_equal(optim_var, target_var)
