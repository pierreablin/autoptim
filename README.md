# autoptim: automatic differentiation + optimization

Do you have a new machine learning model that you want to optimize, and do not want to bother computing the gradients? Autoptim is for you.

## Short presentation
Autoptim is a small Python package that blends Pytorch's automatic differentiation in `scipy.optimize.minimize`.

The gradients are computed under the hood using automatic differentiation; the user only provides the objective function:

```python
import numpy as np
from autoptim import minimize


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


x0 = np.zeros(2)

x_min, _ = minimize(rosenbrock, x0)
print(x_min)

>>> [0.99999913 0.99999825]
```

It comes with the following features:

- **Minimal Pytorch use**: The user only needs to write the objective function in a Pytorch -compatible way. The input/ output of `autoptim.minimize` are Numpy arrays.

- **Smart input processing**: `scipy.optimize.minimize` is only meant to deal with one-dimensional arrays as input. In `autoptim`, variables can be multi-dimensional arrays or lists of arrays.


### Disclaimer

This package is meant to be as easy to use as possible. As so, some compromises on the speed of minimization are made.
## Installation
  To install, use `pip`:
  ```
  pip install autoptim
  ```
## Dependencies
- numpy>=1.12
- scipy>=0.18.0
- Pytorch>=0.4.1


## Examples
Several examples can be found in `autoptim/tutorials`
