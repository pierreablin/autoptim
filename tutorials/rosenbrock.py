import numpy as np
from autoptim import minimize


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


x0 = np.zeros(2)

x_min, _ = minimize(rosenbrock, x0)
print(x_min)
