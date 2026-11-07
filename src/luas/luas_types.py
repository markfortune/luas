from typing import Any, Union
import numpy as np
import jax.numpy as jnp
from abc import ABCMeta
import numbers

__all__ = [
    "JAXArray",
    "Scalar",
    "Array",
    "PyTree",
]

Scalar = Any
Array = Any
JAXArray = jnp.ndarray
PyTree = Any

class Kernel(metaclass=ABCMeta):
    # Base class for Kernel classes
    pass


def is_scalar(x):
    zero_d_array = hasattr(x, "shape") and x.shape == ()
    pure_python_num = isinstance(x, numbers.Number)
    return zero_d_array or pure_python_num

