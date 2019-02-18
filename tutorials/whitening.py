# Author: Pierre Ablin <pierreablin@gmail.com>
# License: MIT


# Example of multi-dimensional arrays
import numpy as np
from autoptim import minimize


n = 100
p = 2

X = np.random.randn(n, p)

# The loss is minimized when W.dot(X) is decorrelated.


def loss(W, X):
    Y = X.mm(W)
    return -W.logdet() + 0.5 * (Y ** 2).sum() / n


# The input is a square matrix
W0 = np.eye(p)

W, _ = minimize(loss, W0, args=(X, ))
print(W)
Y = X.dot(W)
print(Y.T.dot(Y) / n)  # Equal to identity
