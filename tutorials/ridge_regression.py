# Author: Pierre Ablin <pierreablin@gmail.com>
# License: MIT

# An example with additional variables


import autograd.numpy as np
from autoptim import minimize


n = 10
p = 5

X = np.random.randn(n, p)
y = np.random.randn(n)
lbda = 0.1

# The loss shoulb be optimized over beta, with the other parameters fixed.


def loss(beta, X, y, lbda):
    return np.sum((np.dot(X, beta) - y) ** 2) + lbda * np.sum(beta ** 2)


beta0 = np.random.randn(p)

beta_min, _ = minimize(loss, beta0, args=(X, y, lbda))
print(beta_min)
