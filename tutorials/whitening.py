# Author: Pierre Ablin <pierreablin@gmail.com>
# License: MIT


# Example of multi-dimensional arrays
import autograd.numpy as np
from autoptim import minimize


n = 100
p = 2

X = np.random.randn(n, p)

# The loss is minimized when X.dot(W) is decorrelated.


def loss(W, X):
    Y = np.dot(X, W)
    return -np.linalg.slogdet(W)[1] + 0.5 * np.sum(Y ** 2) / n


# The input is a square matrix
W0 = np.eye(p)

W, _ = minimize(loss, W0, args=(X, ))
print(W)
Y = X.dot(W)
print(Y.T.dot(Y) / n)  # Equal to identity
