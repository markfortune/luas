import jax.numpy as jnp
import jax
from jax import custom_jvp, jit
import numpy as np
from typing import Callable, Tuple, Union, Any, Optional
from .luas_types import JAXArray, Scalar, PyTree
from functools import partial

__all__ = [
    "make_vec",
    "make_mat",
    "kron_prod",
    "cyclic_transpose",
    "vmap_for_tensors",
    "tensor_mult",
    "tensor_arb_op",
    "kron_prod_dim_d",
    "calc_total_size",
    "calc_data_shape",
]


def make_vec(R: JAXArray) -> JAXArray:
    r"""Function for converting a matrix of shape ``(N_l, N_t)`` into
    a vector of shape ``(N_l * N_t,)``.
    
    .. math::

        \mathbf{R}_{ij} = r_{i N_t + j}

    Args:
        R (JAXArray): Matrix of shape ``(N_l, N_t)``
        
    Returns:
        JAXArray: A vector of shape ``(N_l * N_t,)``
        
    """
    
    return R.ravel("C")


def make_mat(
    r: JAXArray,
    N_l: int,
    N_t: int
) -> JAXArray:
    r"""Function for converting a vector of shape ``(N_l * N_t,)``
    into an array of shape ``(N_l, N_t)``.
    
    .. math::

        r_{i N_t + j} = \mathbf{R}_{ij}
    
    Args:
        r (JAXArray): Vector of shape ``(N_l * N_t,)``
        N_l (int): Size of wavelength/vertical dimension
        N_t (int): Size of time/horizontal dimension
        
    Returns:
        JAXArray: An array of shape ``(N_l, N_t)``
        
    """
    return r.reshape((N_l, N_t))


def kron_prod(
    A: JAXArray,
    B: JAXArray,
    R: JAXArray
) -> JAXArray:
    r"""Computes the matrix vector product of the kronecker product of two matrices
    ``A`` and ``B`` times a vector ``r``, stored as an ``(N_l, N_t)`` array ``R``.
    
    .. math::

        [\mathbf{A} \otimes \mathbf{B}] \vec{r} = \mathbf{A} \mathbf{R} \mathbf{B}^T
    
    Args:
        A (JAXArray): Matrix on the left side of the kronecker product, of shape ``(N_l, N_l)``.
        B (JAXArray): Matrix on the right side of the kronecker product, of shape ``(N_t, N_t)``.
        R (JAXArray): Vector to right multiply, stored as an ``(N_l, N_t)`` array.
        
    Returns:
        JAXArray: The result of the multiplication as a JAXArray array of shape ``(N_l, N_t)``.
    """
    
    return A @ R @ B.T


@partial(jit, static_argnums=1)
def cyclic_transpose(R, d):
    """Cyclically shifts the axes of an array by ``d`` places, so that axis ``d`` becomes the first axis.

    The new axis ``i`` is the old axis ``(i + d) % R.ndim``. For example, an array of shape
    ``(N_1, N_2, N_3)`` has shape ``(N_2, N_3, N_1)`` after shifting by ``d = 1``, and a 2D array is
    transposed by ``d = 1``. Shifting by ``-d`` undoes a shift by ``d``. Used to apply matrices along
    each dimension of an array in turn, as described in Saatçi (2011).

    Args:
        R (JAXArray): The array to transpose.
        d (int): The number of places to shift the axes by. Must be a static (non-traced) value
            as this function is JIT compiled.

    Returns:
        JAXArray: The transposed array.

    """
    ndim = R.ndim
    axes = tuple((i + d) % ndim for i in range(ndim))
    return jnp.transpose(R, axes)


def vmap_for_tensors(f):
    """Wraps a function acting on 2D arrays so that it acts on arrays with any number of dimensions.

    The wrapped function ``jax.vmap``s ``f`` over every axis except the last two, so ``f`` is applied
    to each 2D slice ``R[..., :, :]`` of the input.

    Args:
        f (Callable): A function ``f(R, **kwargs)`` taking a 2D array of shape ``(N_l, N_t)``.

    Returns:
        Callable: A function ``wrapped(R, **kwargs)`` which takes an array of shape
        ``(..., N_l, N_t)`` and applies ``f`` to each of its 2D slices.

    """
    def wrapped(R, **kwargs):
        *leading, N_l, N_t = R.shape
        
        R_flat = R.reshape(-1, N_l, N_t)
        
        f_new = lambda R: f(R, **kwargs)
        R_prime_flat = jax.vmap(f_new)(R_flat)
        
        return R_prime_flat.reshape(*leading, *R_prime_flat.shape[1:])
    
    return wrapped

    
def tensor_mult(kron_mat_list, X1, X2, R, **kwargs):
    r"""Multiplies an array by the Kronecker product of covariance matrices along each dimension.

    .. math::

        [\mathbf{K}_1 \otimes \mathbf{K}_2 \otimes \dots \otimes \mathbf{K}_D] \vec{r}

    where :math:`\vec{r}` is the vectorised ``R`` and :math:`\mathbf{K}_d` is the covariance matrix
    from ``kron_mat_list[d]`` between the input locations ``X1[d]`` and ``X2[d]``. Each matrix is
    applied along its own dimension of ``R`` in turn using its ``matmul`` method, so the full
    Kronecker product is never built. For 2D data this is :math:`\mathbf{K}_1 \mathbf{R} \mathbf{K}_2^T`.

    Args:
        kron_mat_list (:obj:`list` of CovType): The covariance object along each dimension.
        X1 (tuple): Input locations along each dimension for the rows of each covariance matrix.
        X2 (tuple): Input locations along each dimension for the columns of each covariance matrix.
            Must match the shape of ``R``.
        R (JAXArray): The array to multiply, with one axis per dimension.
        **kwargs: Passed to each covariance object's ``matmul`` method, e.g. ``wn = False``.

    Returns:
        JAXArray: The result of the multiplication, with one axis per dimension.

    """

    gp_dim = len(kron_mat_list)
    R_prime = cyclic_transpose(R, 2)
    
    for d in range(gp_dim):
        method = getattr(kron_mat_list[d], "matmul")

        matmul_fn = lambda R, **kwargs: method(X1[d], X2[d], R, **kwargs)

        if gp_dim > 2:
            matmul_fn = vmap_for_tensors(matmul_fn)
            
        R_prime = matmul_fn(R_prime, **kwargs)
        R_prime = cyclic_transpose(R_prime, 1)
        
    R_prime = cyclic_transpose(R_prime, -2)

    return R_prime


def tensor_arb_op(kron_mat_list, method_name, R, **kwargs):

    gp_dim = len(kron_mat_list)
    R_prime = cyclic_transpose(R, 2)
    
    for d in range(gp_dim):
        method = getattr(kron_mat_list[d], "matmul")
        
        R_prime = method(X1[d], X2[d], R_prime, **kwargs)
        R_prime = cyclic_transpose(R_prime, 1)
        
    R_prime = cyclic_transpose(R_prime, -2)

    return R_prime
    

def kron_prod_dim_d(
    A: JAXArray,
    R: JAXArray,
    d: int,
) -> JAXArray:
    r"""Multiplies an array by a Kronecker product of identity matrices with the matrix ``A`` along
    dimension ``d``, i.e. applies ``A`` along axis ``d`` of ``R``.

    .. math::

        [\mathbf{I} \otimes \dots \otimes \mathbf{A} \otimes \dots \otimes \mathbf{I}] \vec{r}

    Uses the cyclic transpose (see :func:`cyclic_transpose`) as described in Saatçi (2011). For a
    2D array this is :math:`\mathbf{A}\mathbf{R}` for ``d = 0`` and :math:`\mathbf{R}\mathbf{A}^T`
    for ``d = 1``.

    Note:
        Currently only gives the correct result for 2D arrays ``R``.

    Args:
        A (JAXArray): Matrix to apply, of shape ``(N_d, N_d)`` where ``N_d`` is the size of axis ``d``
            of ``R``.
        R (JAXArray): The array to multiply.
        d (int): The axis of ``R`` to apply ``A`` along.

    Returns:
        JAXArray: The result of the multiplication, with the same shape as ``R``.
    """

    R_trans = cyclic_transpose(R, d)
    A_R = A @ R_trans
    return cyclic_transpose(A_R, -d)
    

def calc_total_size(x):
    """Calculates the total number of data points from the input locations.

    Args:
        x (JAXArray | tuple): The input locations, either a single JAXArray for 1D data or a tuple with
            one JAXArray per dimension. Each has shape ``(N_d,)`` or ``(n, N_d)``.

    Returns:
        int: The total number of data points, i.e. the product of the sizes ``N_d`` of each dimension.

    """
    # If it's a single array, just return its size
    if isinstance(x, jnp.ndarray):
        return x.shape[-1]
        
    # Otherwise assume it's an iterable of arrays
    return jnp.prod(jnp.array([xi.shape[-1] for xi in x]))


def calc_data_shape(X):
    """Calculates the shape of the data set from the input locations.

    Args:
        X (JAXArray | tuple): The input locations, either a single array for 1D data or a tuple with
            one JAXArray per dimension. Each has shape ``(N_d,)`` or ``(n, N_d)``, where ``n`` is
            the number of input variables along that dimension.

    Returns:
        tuple: The shape of the data set, ``(N_1, N_2, ...)``.

    """
    # If it's a single array, just return its size
    if isinstance(X, jnp.ndarray) or isinstance(X, np.ndarray):
        return (X.shape[-1],)
        
    # Otherwise assume it's an iterable of arrays
    return sum([(x_i.shape[-1],) for x_i in X], ())

