# Author: Pierre Ablin <pierreablin@gmail.com>
# License: MIT

# Example with several variables

import autograd.numpy as np

from autoptim import minimize

n = 1000
n_components = 3

x = np.concatenate((np.random.randn(n) - 1,
                    3 * np.random.randn(n),
                    np.random.randn(n) + 2))


# Here, the model should fit both the means and the variances. Using
# scipy.optimize.minimize, one would have to vectorize by hand these variables.

def loss(means, variances, x):
    tmp = np.zeros(n_components * n)
    for m, v in zip(means, variances):
        tmp += np.exp(-(x - m) ** 2 / (2 * v ** 2)) / v
    return -np.sum(np.log(tmp))


# autoptim can handle lists of unknown variables

means0 = np.random.randn(n_components)
variances0 = np.random.rand(n_components)
optim_vars = [means0, variances0]
# The variances should be constrained to positivity. To do so, we can pass
# a `bounds` list to `minimize`. Bounds are automatically broadcasted to
# match the input size.

bounds = [(None, None),  # corresponds to means: no constraint
          (0, None)]  # corresponds to variances: positivity constraint.
(means, variances), _ = minimize(loss, optim_vars, args=(x,),
                                 bounds=bounds)

print(means, variances)  # Notice that they have the correct shape.
