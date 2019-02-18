import numpy as np
import torch

from autoptim import minimize

n = 1000
n_components = 3

x = np.concatenate((np.random.randn(n),
                    2 * np.random.randn(n),
                    np.random.randn(n) + 1))


def loss(means, variances, x):
    tmp = torch.zeros(n_components * n).double()
    for m, v in zip(means, variances):
        tmp += torch.exp(-(x - m) ** 2 / (2 * v ** 2)) / v
    return -torch.log(tmp).sum()


means0 = np.random.randn(n_components)
variances0 = np.random.rand(n_components)

bounds = [(None, None), (0, None)]  # Variance should be >0
(means, variances), _ = minimize(loss, (means0, variances0), args=(x,),
                                 bounds=bounds)

print(means, variances)
