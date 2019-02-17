import numpy as np
import torch

from scipy.optimize import minimize as minimize_


def scipy_func(objective_function, x, shapes, args=()):
    optim_vars = split(x, shapes)

    torch_vars = [torch.tensor(var, requires_grad=True) for var in optim_vars]
    del optim_vars
    obj = objective_function(*torch_vars, *args)
    obj.backward()
    gradients = [var.grad.numpy() for var in torch_vars]
    g_vectorized, _ = vectorize(gradients)
    return obj.item(), g_vectorized


def minimize(objective_function, optim_vars, args=(), **kwargs):
    # Convert to torch
    args_torch = [torch.tensor(arg).double() for arg in args]

    if type(optim_vars) is np.ndarray:
        input_is_array = True
        optim_vars = (optim_vars,)
    else:
        input_is_array = False
    # Vectorize optimization variables
    x0, shapes = vectorize(optim_vars)

    def func(x):
        return scipy_func(objective_function, x, shapes, args_torch)
    res = minimize_(func, x0, jac=True, **kwargs)
    output = split(res['x'], shapes)
    if input_is_array:
        output = output[0]
    return output, res


def vectorize(optim_vars):
    shapes = [var.shape for var in optim_vars]
    x = np.concatenate([var.ravel() for var in optim_vars])
    return x, shapes


def split(x, shapes):
    x_split = np.split(x, np.cumsum([np.prod(shape) for shape in shapes[:-1]]))
    optim_vars = [var.reshape(*shape) for (var, shape) in zip(x_split, shapes)]
    return optim_vars


if __name__ == '__main__':
    shapes_target = [(2, 3, 4), (5,), (2, 1, 4, 5)]
    optim_vars = [np.arange(np.prod(s)).reshape(*s) for s in shapes_target]
    x, shapes = vectorize(optim_vars)
    output = split(x, shapes)
