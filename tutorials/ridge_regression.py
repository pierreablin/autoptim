# Author: Pierre Ablin <pierreablin@gmail.com>
# License: MIT

# An example with additional variables


import numpy as np
from autoptim import minimize


n = 10
p = 5

X = np.random.randn(n, p)
y = np.random.randn(n)
lbda = 0.1

# The loss shoulb be optimized over beta, with the other parameters fixed.
# Bear in mind that X, y, and lbda will be converted to torch tensors under
# the hood.


def loss(beta, X, y, lbda):
    return ((X.mv(beta) - y) ** 2).sum() + lbda * (beta ** 2).sum()


beta0 = np.random.randn(p)

beta_min, _ = minimize(loss, beta0, args=(X, y, lbda))
print(beta_min)
