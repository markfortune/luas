import jax.numpy as jnp
from .luas_types import JAXArray, Scalar, PyTree

__all__ = [
    "make_vec",
    "make_mat",
    "kron_prod",
    "kronecker_inv_vec"
]


def make_vec(R: JAXArray) -> JAXArray:
    """Function for converting a matrix of the same shape as the input data
    Y into a vector.
    
    Args:
        R (JAXArray): Matrix of shape (N_l, N_t)
        
    Returns:
        A JAXArray vector of length N_l * N_t
    """
    return R.ravel("C")

def make_mat(r: JAXArray, N_l: int, N_t: int) -> JAXArray:
    """Function for converting a vector of length N_l * N_t
    into an array of the same shape as the input data
    Y into a vector.
    
    Args:
        r (JAXArray): Vector of size N_l * N_t
        N_l (int): Size of first dimension l
        N_t (int): Size of second dimension t
        
    Returns:
        A JAXArray array of shape (N_l, N_t)
    """
    return r.reshape((N_l, N_t))

def kron_prod(A: JAXArray, B: JAXArray, R: JAXArray) -> JAXArray:
    """Function for converting a vector of length N_l * N_t
    into an array of the same shape as the input data
    Y into a vector.
    
    Args:
        r (JAXArray): Vector of size N_l * N_t
        N_l (int): Size of first dimension l
        N_t (int): Size of second dimension t
        
    Returns:
        The result of multiplying the matrix A KRON B times the vector R
        as a JAXArray array of shape (N_l, N_t).
    """
    
    return A @ R @ B.T

def kronecker_inv_vec(R: JAXArray, storage_dict: PyTree) -> JAXArray:
    """Function for converting a vector of length N_l * N_t
    into an array of the same shape as the input data
    Y into a vector.
    
    Args:
        R (JAXArray): Array of shape (N_l, N_t)
        storage_dict (PyTree): PyTree storing information about decomposition
            of the covariance matrix
        
    Returns:
        The product of the inverse of the covariance matrix multiplied on the right
        by the vector represented by R as a JAXArray array of shape (N_l, N_t).
    """
    
    b = kron_prod(storage_dict["W_l"].T, storage_dict["W_t"].T, R)
    b = jnp.multiply(storage_dict["D_inv"], b)
    b = kron_prod(storage_dict["W_l"], storage_dict["W_t"], b)

    return b
