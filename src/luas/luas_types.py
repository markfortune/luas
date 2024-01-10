from typing import Any, Union
import numpy as np
import jax.numpy as jnp
from abc import ABCMeta

Scalar = Any
Array = Any
JAXArray = jnp.ndarray
PyTree = Any

class Kernel(metaclass=ABCMeta):
    # Base class for Kernel classes
    pass