# Author: Pierre Ablin <pierreablin@gmail.com>
# License: MIT

# An example about preconditioning: the problem is to minimize
# || y - X.dot(beta)|| ** 2 + lambda * || beta || ** 2.
# The Hessian of the problem is 2 * (X^TX + lambda * Id).
# For approximately decorellated X, it is well approximated by its diagonal,
# whose square root gives a natural preconditioner. It is extremely simple to
# implement using autoptim !
from time import time

import autograd.numpy as np
from autoptim import minimize


n = int(1e6)
p = 20

# Define the parameters : the rows of X have different powers.
X = np.random.randn(n, p) * np.arange(1, p+1)
y = np.random.randn(n)
lbda = 0.1

# Define the loss :


def loss(beta, X, y, lbda):
    return np.sum((np.dot(X, beta) - y) ** 2) + lbda * np.sum(beta ** 2)

# Define the preconditioner. Note that it is much faster to compute than the
# whole Hessian (O(p * n ^2))


hessian_approx = 2 * lbda * np.ones(p)
for i in range(p):
    hessian_approx[i] += 2 * np.dot(X[:, i], X[:, i])

diag_precon = np.sqrt(hessian_approx)

# Define the forward and backward preconditioning functions. They should have
# the same signature as the objective function. Here, we dot not use the extra
# arguments.


def precon_fwd(beta, X, y, lbda):
    return beta * diag_precon


def precon_bwd(beta_precon, X, y, lbda):
    return beta_precon / diag_precon


beta0 = np.random.randn(p)
# Run the minimization with the preconditioning
t0 = time()
beta_min, _ = minimize(loss, beta0, args=(X, y, lbda), precon_fwd=precon_fwd,
                       precon_bwd=precon_bwd)
print('Minimization with preconditioning took %.2f sec.' % (time() - t0))
print(beta_min)

# It gives the same output without preconditioning:
t0 = time()
beta_min, _ = minimize(loss, beta0, args=(X, y, lbda))
print('Minimization without preconditioning took %.2f sec.' % (time() - t0))
print(beta_min)

# But it is faster with preconditioning (about twice in this example, but it
# can give more impressing speedups)!
