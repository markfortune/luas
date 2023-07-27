import jax.numpy as jnp
import jax
from typing import Callable

jax.config.update("jax_enable_x64", True)


def evaluate_kernel(kernel_fn: Callable, x: jnp.array, y: jnp.array, l: jnp.float64) -> jnp.array:
    K = jax.vmap(lambda x1: jax.vmap(lambda y1: kernel_fn(x1, y1, l))(y))(x)
    return K


def periodic_kernel(x: jnp.array, y: jnp.array, l: jnp.float64) -> jnp.array:
    tau = jnp.square(jnp.sin(x - y)/l)
    return jnp.exp(-2.0 * tau)


def cosine_kernel(x: jnp.array, y: jnp.array, l: jnp.float64) -> jnp.array:
    return jnp.cos(x-y)


def rbf_kernel(x: jnp.array, y: jnp.array, l: jnp.float64) -> jnp.array:
    tau = jnp.sum(jnp.square(x / l - y / l))
    return jnp.exp(-0.5 * tau)


def ration_quad_kernel(x: jnp.array, y: jnp.array, l: jnp.float64) -> jnp.array:
    tau = jnp.sum(jnp.square(x / l - y / l))
    return jnp.reciprocal(jnp.sqrt(1 + tau))
