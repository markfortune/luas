import jax.numpy as jnp
from jax import vmap
from typing import Callable
from .luas_types import JAXArray, Scalar

__all__ = [
    "evaluate_kernel",
    "squared_exp_kernel",
    "matern32_kernel",
    "matern52_kernel",
    "rational_quadratic_kernel",
    "exp_sine_squared_kernel",
    "cosine_kernel",
    "distanceL1",
    "distanceL2",
]

def evaluate_kernel(kernel_fn: Callable, x: JAXArray, y: JAXArray, l: Scalar) -> JAXArray:
    """Uses JAX's vmap function to efficiently build the covariance matrix from
    a given kernel function
    
    Args:
        kernel_fn (Callable): The desired kernel function to use
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        l (Scalar): The length scale for the kernel function
    
    Returns:
        JAXArray: The constructed covariance matrix
        
    """
    K = vmap(lambda x1: vmap(lambda y1: kernel_fn(x1, y1, l))(y))(x)
    return K


def squared_exp_kernel(x: JAXArray, y: JAXArray, l: Scalar) -> JAXArray:
    r"""Squared exponential kernel function, also known as the radial basis function,
    used with evaluate_kernel to build a covariance matrix.
    
    .. math::

        k(x, y) = \exp\Bigg( -\frac{|x - y|^2}{2l^2}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        l (Scalar): The length scale to use
        
    Returns:
        Scalar: Covariance between two input vectors
    """
    
    tau_sq = distanceL2(x, y)/l**2
    return jnp.exp(-0.5 * tau_sq)


def matern32_kernel(x: JAXArray, y: JAXArray, l: Scalar) -> JAXArray:
    """Matern 3/2 kernel function, used with evaluate_kernel
    to build covariance matrices.
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        l (Scalar): The length scale to use
        
    Returns:
        Scalar: Covariance between two input vectors
    """
    
    delta_t = jnp.sqrt(3)*distanceL1(x, y)/l
    return (1+delta_t)*jnp.exp(-delta_t)


def matern52_kernel(x: JAXArray, y: JAXArray, l: Scalar) -> JAXArray:
    """Matern 5/2 kernel function, used with evaluate_kernel
    to build covariance matrices.
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        l (Scalar): The length scale to use
        
    Returns:
        Scalar: Covariance between two input vectors
    """
    delta_t = jnp.sqrt(5)*distanceL1(x, y)/l
    return (1+delta_t+jnp.square(delta_t)/3)*jnp.exp(-delta_t)


def rational_quadratic_kernel(x: JAXArray, y: JAXArray, l: Scalar, alpha: Scalar) -> JAXArray:
    """Rational quadratic kernel function, used with evaluate_kernel
    to build covariance matrices.
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        l (Scalar): The length scale to use
        
    Returns:
        Scalar: Covariance between two input vectors
    """
    tau_sq = distanceL2(x, y)/l**2
    return (1. + 0.5*tau_sq/alpha)**(-1/alpha)



def exp_sine_squared_kernel(x: JAXArray, y: JAXArray, l: Scalar) -> JAXArray:
    """Exponential sine squared kernel, used with evaluate_kernel
    to build covariance matrices which have periodic covariance.
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        l (Scalar): The length scale to use
        
    Returns:
        JAXArray: Covariance between two input vectors
    """
    
    tau_sq = jnp.sum(jnp.square(jnp.sin(x - y)/l))
    return jnp.exp(-2.0 * tau_sq)


def cosine_kernel(x: JAXArray, y: JAXArray, l: Scalar) -> JAXArray:
    delta_t = distanceL1(x, y)/l
    return jnp.sum(jnp.cos(2*jnp.pi*delta_t))


def distanceL1(x: JAXArray, y: JAXArray) -> JAXArray:
    return jnp.sum(jnp.abs(x - y))


def distanceL2(x: JAXArray, y: JAXArray) -> JAXArray:
    return jnp.sum(jnp.square(x - y))
    