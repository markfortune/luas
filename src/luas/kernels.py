import jax.numpy as jnp
from jax import vmap
from typing import Callable
from .luas_types import JAXArray, Scalar

__all__ = [
    "evaluate_kernel",
    "distanceL1",
    "distanceL2Sq",
    "squared_exp",
    "matern12",
    "matern32",
    "matern52",
    "rational_quadratic",
    "exp_sine_squared",
    "cosine",
]

def evaluate_kernel(kernel_fn: Callable, x: JAXArray, y: JAXArray, *args) -> JAXArray:
    """Uses JAX's vmap function to efficiently build the covariance matrix from
    a given kernel function
    
    Args:
        kernel_fn (Callable): The desired kernel function to use
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        l (Scalar): Length scale
    
    Returns:
        JAXArray: The constructed covariance matrix
    """
    
    K = vmap(lambda x1: vmap(lambda y1: kernel_fn(x1, y1, *args))(y))(x)
    return K
    

def distanceL1(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    r"""Evaluates the L1 norm of two input vectors divided by a length scale.
    
    .. math::

        L1(x, y) = \frac{|x - y|}{L}
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: L1 norm between two input vectors
    """
    
    return jnp.sum(jnp.abs(x - y)/L)


def distanceL2Sq(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    r"""Evaluates the Squared L2 norm of two input vectors divided by a length scale.
    
    .. math::

        L2^2(x, y) = \frac{|x - y|^2}{L^2}
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: L2 norm between two input vectors
    """
    
    return jnp.sum(jnp.square(x - y)/L**2)
    

def squared_exp(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    r"""Squared exponential kernel function, also known as the radial basis function,
    used with evaluate_kernel to build a covariance matrix.
    
    .. math::

        k(x, y) = \exp\Bigg( -\frac{|x - y|^2}{2L^2}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
    """
    
    return evaluate_kernel(squared_exp_calc, x, y, L)
    
    
def squared_exp_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by squared_exp to evaluate the squared exponential kernel function. 
    """

    tau_sq = distanceL2Sq(x, y, L)
    return jnp.exp(-0.5 * tau_sq.sum())


def exp(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    r"""Exponential kernel function, used with evaluate_kernel
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(\frac{|x - y|}{L}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
    """
    
    return evaluate_kernel(exp_calc, x, y, L)


def exp_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by exp to evaluate the Exponential kernel function. 
    """

    delta_t = distanceL1(x, y, L).sum()
    return jnp.exp(-delta_t)
   

def matern32(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    r"""Matern 3/2 kernel function, used with evaluate_kernel
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(1 + \sqrt{3} \frac{|x - y|}{L}\Bigg) \exp\Bigg( -\sqrt{3} \frac{|x - y|}{L}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
    """
    
    return evaluate_kernel(matern32_calc, x, y, L)


def matern32_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by matern32 to evaluate the Matern 3/2 kernel function. 
    """

    delta_t = jnp.sqrt(3)*distanceL1(x, y, L).sum()
    return (1+delta_t)*jnp.exp(-delta_t)
    

def matern52(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    r"""Matern 5/2 kernel function, used with evaluate_kernel
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(1 + \sqrt{5} \frac{|x - y|}{L} + \frac{5|x - y|^2}{3L^2}\Bigg) \exp\Bigg( -\sqrt{5}\frac{|x - y|}{L}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
    """
    
    return evaluate_kernel(matern52_calc, x, y, L)


def matern52_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by matern52 to evaluate the Matern 5/2 kernel function. 
    """

    delta_t = jnp.sqrt(5)*distanceL1(x, y, L).sum()
    return (1+delta_t+jnp.square(delta_t)/3)*jnp.exp(-delta_t)


def rational_quadratic(x: JAXArray, y: JAXArray, L: Scalar, alpha: Scalar) -> JAXArray:
    r"""Rational quadratic kernel function, used with evaluate_kernel
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(1 + \frac{|x - y|^2}{2 \alpha L^2}\Bigg)^{-\alpha}
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        alpha (Scalar): Scale mixture parameter
        
    Returns:
        Scalar: Covariance between two input vectors
    """
    return evaluate_kernel(rational_quadratic_calc, x, y, L, alpha)


def rational_quadratic_calc(x: JAXArray, y: JAXArray, L: Scalar, alpha: Scalar) -> JAXArray:
    """Function used by rational_quadratic to evaluate the rational quadratic kernel function. 
    """

    tau_sq = distanceL2Sq(x, y, L).sum()
    return (1. + 0.5*tau_sq/alpha)**(-alpha)


def exp_sine_squared(x: JAXArray, y: JAXArray, L: Scalar, P: Scalar) -> JAXArray:
    r"""Exponential sine squared kernel, used with evaluate_kernel
    to build covariance matrices which have periodic covariance.
    
    .. math::

        k(x, y) = \exp\Bigg( -\frac{2 \sin^2(\pi(x - y)/P)}{L^2}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        P (Scalar): Period
        
    Returns:
        JAXArray: Covariance between two input vectors
    """
    
    return evaluate_kernel(exp_sine_squared_calc, x, y, L, P)


def exp_sine_squared_calc(x: JAXArray, y: JAXArray, L: Scalar, P: Scalar) -> JAXArray:
    """Function used by exp_sine_squared to evaluate the exponential sine squared kernel function. 
    """

    tau_sq = (jnp.sum(jnp.square(jnp.sin(jnp.pi*(x - y)/P)/L))).sum()
    return jnp.exp(-2.0 * tau_sq)
    

def cosine(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Cosine kernel, used with evaluate_kernel
    to build covariance matrices which have periodic covariance.
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        l (Scalar): Length scale
        
    Returns:
        JAXArray: Covariance between two input vectors
    """
    
    return evaluate_kernel(cosine_calc, x, y, L)


def cosine_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by cosine to evaluate the cosine kernel function. 
    """
        
    delta_t = distanceL1(x, y, L).sum()
    return jnp.cos(2*jnp.pi*delta_t)
