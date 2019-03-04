# Author: Pierre Ablin <pierreablin@gmail.com>
# License: MIT

import numpy as np_

import autograd.numpy as np

from autograd import grad

from scipy.optimize import minimize as minimize_


def _scipy_func(objective_function, gradient, x, shapes, args=()):
    optim_vars = _split(x, shapes)
    obj = objective_function(optim_vars, *args)
    gradients = gradient(optim_vars, *args)
    g_vectorized, _ = _vectorize(gradients)
    return obj, g_vectorized


def _convert_to_tuple(optim_vars):
    if type(optim_vars) not in (list, tuple):
        return (optim_vars,)
    return optim_vars


def minimize(objective_function, optim_vars, args=(), precon_fwd=None,
             precon_bwd=None, **kwargs):
    """A wrapper to call scipy.optimize.minimize while computing the gradients
       using autograd's auto-differentiation.
        Parameters
        ----------
        objective_function : callable
            The objective function to be minimized.
                ``fun(optim_vars, *args) -> float``
            or
                ``fun(*optim_vars, *args) -> float``
            where optim_vars is either a numpy array or a list of numpy
            arrays and `args` is a tuple of fixed parameters needed to
            completely specify the function.
        optim_vars : ndarray or list of ndarrays
            Initial guess.
        args : tuple, optional
            Extra arguments passed to the objective function.
        precon_fwd : callable, optional
            The forward preconditioning.
                ``fun(optim_vars, *args) -> precon_optim_vars``
            or
                ``fun(*optim_vars, *args) -> precon_optim_vars``
            where optim_vars is either a numpy array or a list of numpy
            arrays and `args` is a tuple of fixed parameters needed to
            completely specify the function.
            The optimized function will be the composition:
            `objective_function(precon_fwd(optim_vars))`.
        precon_bwd : callable, optional
            The backward preconditioning.
                ``fun(precon_optim_vars, *args) -> optim_vars``
            or
                ``fun(*precon_optim_vars, *args) -> optim_vars``
            where optim_vars is either a numpy array or a list of numpy
            arrays and `args` is a tuple of fixed parameters needed to
            completely specify the function.
            This should be the reciprocal function of precon_fwd.
        kwargs : dict, optional
            Extra arguments passed to scipy.optimize.minimize. See
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            for the full list of available keywords.
        Returns
        -------
        output : ndarray or list of ndarrays
            The solution, of same shape as the input `optim_vars`.
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.
        """
    # Check if there is preconditioning:
    precondition = precon_fwd is not None
    if precondition != (precon_bwd is not None):
        error_string = {True: 'precon_fwd', False: 'precon_bwd'}[precondition]
        raise ValueError('You should specify both precon_fwd and precon_bwd,'
                         ' you only specified %s' % error_string)
    if precondition:  # Run `minimize` in the preconditioned space:

        optim_vars = _convert_to_tuple(optim_vars)
        precon_optim_vars = precon_fwd(*optim_vars, *args)
        n_args = len(args)

        def precon_objective(*precon_optim_vars_and_args):
            args = precon_optim_vars_and_args[-n_args:]
            optim_vars = precon_bwd(*precon_optim_vars_and_args)
            optim_vars = _convert_to_tuple(optim_vars)
            return objective_function(*optim_vars, *args)

        precon_result, res = minimize(precon_objective, precon_optim_vars,
                                      args=args, precon_fwd=None,
                                      precon_bwd=None, **kwargs)
        precon_result = _convert_to_tuple(precon_result)
        return precon_bwd(*precon_result, *args), res
    # Check if there are bounds:
    bounds = kwargs.get('bounds')
    bounds_in_kwargs = bounds is not None

    # Convert input to a list if it is a single array
    if type(optim_vars) is np.ndarray:
        input_is_array = True
        optim_vars = (optim_vars,)
        if bounds_in_kwargs:
            bounds = (bounds,)
    else:
        input_is_array = False

    # Convert loss to readable autograd format

    def objective_converted(optim_vars, *args):
        return objective_function(*optim_vars, *args)

    # Compute the gradient
    gradient = grad(objective_converted)
    # Vectorize optimization variables
    x0, shapes = _vectorize(optim_vars)

    # Convert bounds to the correct format
    if bounds_in_kwargs:
        bounds = _convert_bounds(bounds, shapes)
        kwargs['bounds'] = bounds

    # Define the scipy optimized function and run scipy.minimize
    def func(x):
        return _scipy_func(objective_converted, gradient, x, shapes, args)
    res = minimize_(func, x0, jac=True, **kwargs)

    # Convert output to the input format
    output = _split(res['x'], shapes)
    if input_is_array:
        output = output[0]
    return output, res


def _convert_bounds(bounds, shapes):
    output_bounds = []
    for shape, bound in zip(shapes, bounds):
        # Check is the bound is already parsable by scipy.optimize
        b = bound[0]
        if isinstance(b, (list, tuple, np.ndarray)):
            output_bounds += bound
        else:
            output_bounds += [bound, ] * np.prod(shape)
    return output_bounds


def _vectorize(optim_vars):
    shapes = [var.shape for var in optim_vars]
    x = np.concatenate([var.ravel() for var in optim_vars])
    return x, shapes


def _split(x, shapes):
    x_split = np.split(x, np.cumsum([np.prod(shape) for shape in shapes[:-1]]))
    optim_vars = [var.reshape(*shape) for (var, shape) in zip(x_split, shapes)]
    return optim_vars
