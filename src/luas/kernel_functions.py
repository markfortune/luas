import jax.numpy as jnp
from jax import vmap
from typing import Callable
from .luas_types import JAXArray, Scalar

__all__ = [
    "evaluate_kernel",
    "periodic_kernel",
    "cosine_kernel",
    "rbf_kernel",
    "matern32_kernel",
    "ration_quad_kernel",
]

def evaluate_kernel(kernel_fn: Callable, x: JAXArray, y: JAXArray, l: Scalar) -> JAXArray:
    K = vmap(lambda x1: vmap(lambda y1: kernel_fn(x1, y1, l))(y))(x)
    return K


def periodic_kernel(x: JAXArray, y: JAXArray, l: Scalar) -> JAXArray:
    tau = jnp.sum(jnp.square(jnp.sin(x - y)/l))
    return jnp.exp(-2.0 * tau)


def cosine_kernel(x: JAXArray, y: JAXArray, l: Scalar) -> JAXArray:
    return jnp.sum(jnp.cos(x-y))


def rbf_kernel(x: JAXArray, y: JAXArray, l: Scalar) -> JAXArray:
    tau = jnp.sum(jnp.square(x / l - y / l))
    return jnp.exp(-0.5 * tau)


def matern32_kernel(x: JAXArray, y: JAXArray, l: Scalar) -> JAXArray:
    delta_t = jnp.sum(jnp.abs(x / l - y / l))
    return (1+jnp.sqrt(3)*delta_t)*jnp.exp(-jnp.sqrt(3)*delta_t)


def ration_quad_kernel(x: JAXArray, y: JAXArray, l: Scalar) -> JAXArray:
    tau = jnp.sum(jnp.square(x / l - y / l))
    return jnp.reciprocal(jnp.sqrt(1 + tau))
