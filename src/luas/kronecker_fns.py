import jax.numpy as jnp
from jax import custom_jvp
from typing import Callable, Tuple, Union, Any, Optional
from .luas_types import JAXArray, Scalar, PyTree

__all__ = [
    "make_vec",
    "make_mat",
    "kron_prod",
    "K_inv_vec",
    "r_K_inv_r",
    "logdetK_calc",
    "logdetK_calc_hessianable",
]


def make_vec(R: JAXArray) -> JAXArray:
    r"""Function for converting a matrix of shape ``(N_l, N_t)`` into
    a vector of shape ``(N_l * N_t,)``.
    
    .. math::

        \mathbf{R}_{ij} = r_{i N_l + j}
    
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

        r_{i N_l + j} = \mathbf{R}_{ij}
    
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
        A (JAXArray): Matrix on the left side of the kronecker product.
        B (JAXArray): Matrix on the right side of the kronecker product.
        R (JAXArray): Vector to right multiply, stored as an ``(N_l, N_t)`` array.
        
    Returns:
        JAXArray: The result of the multiplication as a JAXArray array of shape ``(N_l, N_t)``.
    """
    
    return A @ R @ B.T


@custom_jvp
def K_inv_vec(
    R: JAXArray,
    stored_values: PyTree
) -> JAXArray:
    r"""Computes the matrix vector product of the inverse of the covariance matrix ``K``
    times a given input vector ``R`` which must be stored as an ``(N_l, N_t)`` array.
    
    This function will give numerically stable and exact results for first and second derivatives
    evaluated using ``jax.grad`` and ``jax.hessian`` functions. Higher order derivatives will not necessarily give
    correct results.
    
    Works by computing:
    
    .. math::

        K^{-1} \vec{r} = [\mathbf{W}_\lambda \otimes \mathbf{W}_t] D^{-1} [\mathbf{W}_\lambda^T \otimes \mathbf{W}_t^T] \vec{r}
    
    Where the definitions of each term can be found in the tutorial "An Introduction into 2D Gaussian Processes" at
    "https://luas.readthedocs.io/en/latest/tutorials/2D_GP_intro.html".
    
    Args:
        R (JAXArray): Array of shape ``(N_l, N_t)``
        stored_values (PyTree): PyTree storing information about decomposition
            of the covariance matrix.
        
    Returns:
        JAXArray: The product of the inverse of the covariance matrix multiplied on the right
        by the vector represented by ``R`` as a JAXArray of shape ``(N_l, N_t)``.
    
    """
    
    b = kron_prod(stored_values["W_l"].T, stored_values["W_t"].T, R)
    b = jnp.multiply(stored_values["D_inv"], b)
    b = kron_prod(stored_values["W_l"], stored_values["W_t"], b)

    return b


@K_inv_vec.defjvp
def K_inv_vec_derivative(
    primals: Tuple[JAXArray, PyTree],
    tangents: Tuple[JAXArray, PyTree],
) -> Tuple[JAXArray, JAXArray]:
    r"""Custom derivative of ``K_inv_vec`` defined because automatic differentiation can fail to give a numerically stable result
    in some situations.
    
    Works by computing:
    
    .. math::

       \frac{\partial K^{-1} \vec{r}}{\partial p} = - K^{-1} \frac{\partial K}{\partial p} K^{-1} \vec{r} + K^{-1} \frac{\partial \vec{r}}{\partial p}
    
    """
    
    # The values input into K_inv_vec
    R, stored_values = primals
    
    # The derivatives of the values input into K_inv_vec
    R_dot, stored_values_dot = tangents

    # Solves K^-1 R
    K_inv_R = K_inv_vec(R, stored_values)
    
    # These two lines compute dK (K^-1 R)
    dK_K_inv_R =  kron_prod(stored_values_dot["Kl"], stored_values["Kt"], K_inv_R) + kron_prod(stored_values["Kl"], stored_values_dot["Kt"], K_inv_R)
    dK_K_inv_R += kron_prod(stored_values_dot["Sl"], stored_values["St"], K_inv_R) + kron_prod(stored_values["Sl"], stored_values_dot["St"], K_inv_R)

    # Finally compute K^-1 (dK K^-1 R)
    K_inv_dK_K_inv_R = K_inv_vec(dK_K_inv_R, stored_values)

    # Include derivative wrt R
    dK_inv_R = -K_inv_dK_K_inv_R + K_inv_vec(R_dot, stored_values)
    
    # custom_jvp requires we return the value as well as the derivative as a tuple
    return K_inv_R, dK_inv_R


@custom_jvp
def r_K_inv_r(
    R: JAXArray,
    stored_values: PyTree,
) -> Scalar:
    r"""Computes the vector matrix vector product of the inverse of the covariance matrix K
    multiplied on the left and right by a given input vector ``R`` which must be stored
    as an ``(N_l, N_t)`` array.
    
    .. math::

        f(K, r) = \vec{r}^T \mathbf{K}^{-1} \vec{r}
    
    Args:
        R (JAXArray): Array of shape ``(N_l, N_t)``
        stored_values (PyTree): PyTree storing information about decomposition
            of the covariance matrix
        
    Returns:
        Scalar: The product of the inverse of the covariance matrix multiplied on the left and right
        by the vector represented by ``R`` as a JAXArray array of shape ``(N_l, N_t)``.
    
    """
    
    alpha1 = kron_prod(stored_values["W_l"].T, stored_values["W_t"].T, R)
    alpha2 = jnp.multiply(stored_values["D_inv"], alpha1)
    
    return jnp.multiply(alpha1, alpha2).sum()


@r_K_inv_r.defjvp
def r_K_inv_r_derivative(
    primals: Tuple[JAXArray, PyTree],
    tangents: Tuple[JAXArray, PyTree],
) -> Tuple[Scalar, Scalar]:
    r"""Custom derivative of ``r_K_inv_r`` defined because automatic differentiation can fail to give a numerically stable result
    in some situations.
    
    Works by computing:
    
    .. math::

       \frac{\partial \vec{r}^T K^{-1} \vec{r}}{\partial p} = - \vec{r}^T K^{-1} \frac{\partial K}{\partial p} K^{-1} \vec{r}
                                                              + 2 \frac{\partial \vec{r}^T}{\partial p} K^{-1} \frac{\partial \vec{r}}{\partial p}
    
    """
    # The values input into r_K_inv_r
    R, stored_values = primals
    
    # The derivatives of the values input into r_K_inv_r
    R_dot, stored_values_dot = tangents

    # Solves K^-1 R
    K_inv_R = K_inv_vec(R, stored_values)
    
    # These two lines compute dK (K^-1 R)
    dK_K_inv_R =  kron_prod(stored_values_dot["Kl"], stored_values["Kt"], K_inv_R) + kron_prod(stored_values["Kl"], stored_values_dot["Kt"], K_inv_R)
    dK_K_inv_R += kron_prod(stored_values_dot["Sl"], stored_values["St"], K_inv_R) + kron_prod(stored_values["Sl"], stored_values_dot["St"], K_inv_R)
    
    # Calculate derivative wrt K^-1 term
    dr_K_inv_r = - jnp.multiply(dK_K_inv_R, K_inv_R).sum()
    
    # Add derivative wrt R term
    dr_K_inv_r += 2*jnp.multiply(R_dot, K_inv_R).sum() 

    # custom_jvp requires we return the value as well as the derivative as a tuple
    return jnp.multiply(R, K_inv_R).sum(), dr_K_inv_r

    
@custom_jvp
def logdetK_calc(stored_values: PyTree) -> Scalar:
    """Returns the log determinant of a covariance matrix ``K`` given that the matrix
    has already been decomposed using eigendecomposition.
    
    The first order derivatives of this function taken using ``jax.grad`` should be numerically stable
    and exact, however the second order derivatives of this function may not be correct or numerically stable.
    Use ``logdetK_calc_hessianable`` as a slightly more expensive to compute alternative which gives correct values
    for the log determinant as well as its first and second order derivatives (as given by ``jax.grad`` and ``jax.hessian``).
    
    Args:
        stored_values (PyTree): PyTree storing information about decomposition
            of the covariance matrix
        
    Returns:
        Scalar: The log determinant of the covariance matrix ``K``.
    """
    
    return stored_values["logdetK"]

@logdetK_calc.defjvp
def logdetK_derivative(
    primals: PyTree,
    tangents: PyTree,
) -> Tuple[Scalar, Scalar]:
    """Custom derivative of ``logdetK_calc`` defined to give the correct results given a previously computed
    decomposition of the covariance matrix ``K``. This derivative is only accurate for first order derivatives.
    
    """
    
    # Get the stored values from the decomposition of K as well as the derivative of those values
    stored_values, = primals
    stored_values_dot, = tangents

    W_l = stored_values["W_l"]
    W_t = stored_values["W_t"]
    D_inv = stored_values["D_inv"]
    
    # Efficiently compute just the required diagonal elements of W.T @ K @ W for each matrix
    W_Kl_W_diag = jnp.multiply(W_l.T, (stored_values["Kl"] @ W_l).T).sum(1)
    W_Kl_W_diag_dot = jnp.multiply(W_l.T, (stored_values_dot["Kl"] @ W_l).T).sum(1)
    
    W_Kt_W_diag = jnp.multiply(W_t.T, (stored_values["Kt"] @ W_t).T).sum(1)
    W_Kt_W_diag_dot = jnp.multiply(W_t.T, (stored_values_dot["Kt"] @ W_t).T).sum(1)
    
    W_Sl_W_diag = jnp.multiply(W_l.T, (stored_values["Sl"] @ W_l).T).sum(1)
    W_Sl_W_diag_dot = jnp.multiply(W_l.T, (stored_values_dot["Sl"] @ W_l).T).sum(1)
    
    W_St_W_diag = jnp.multiply(W_t.T, (stored_values["St"] @ W_t).T).sum(1)
    W_St_W_diag_dot = jnp.multiply(W_t.T, (stored_values_dot["St"] @ W_t).T).sum(1)

    # Implements the product rule to define the derivative of the log determinant
    K_deriv = jnp.outer(W_Kl_W_diag_dot, W_Kt_W_diag)
    K_deriv += jnp.outer(W_Kl_W_diag, W_Kt_W_diag_dot)
    K_deriv += jnp.outer(W_Sl_W_diag_dot, W_St_W_diag)
    K_deriv += jnp.outer(W_Sl_W_diag, W_St_W_diag_dot)

    # custom_jvp requires we return the value as well as the derivative as a tuple
    return stored_values["logdetK"], jnp.multiply(D_inv, K_deriv).sum()


@custom_jvp
def logdetK_calc_hessianable(
    stored_values: PyTree
) -> Scalar:
    """Returns the log determinant of a covariance matrix ``K`` given that the matrix
    has already been decomposed using eigendecomposition.
    
    This is an alternative to ``logdetK_calc`` which gives correct values for the log determinant
    as well as its first and second order derivatives as given by ``jax.grad`` and ``jax.hessian``.
    However, the first derivative calculation of this function can be slightly more expensive
    to compute so for purposes which do not require second order derivatives of the log determinant,
    ``logdetK_calc`` is preferred.
    
    Args:
        stored_values (PyTree): PyTree storing information about decomposition
            of the covariance matrix
        
    Returns:
        Scalar: The log determinant of the covariance matrix ``K`` as a Scalar.
    """
    
    return stored_values["logdetK"]


@logdetK_calc_hessianable.defjvp
def logdetK_calc_hessianable_derivative(
    primals: PyTree,
    tangents: PyTree,
) -> Tuple[Scalar, Scalar]:
    """Custom derivative of ``logdetK_calc_hessianable`` defined to give the correct results given a previously computed
    decomposition of the covariance matrix ``K``. This derivative is numerically stable and accurate for first and second order derivatives.
    
    """
    # Get the stored values from the decomposition of K as well as the derivative of those values
    stored_values, = primals
    stored_values_dot, = tangents

    W_l = no_deriv(stored_values["W_l"])
    W_t = no_deriv(stored_values["W_t"])
    D_inv = no_deriv(stored_values["D_inv"])

    # Will require all elements of W.T @ K @ W to be calculated so cannot just calculate diagonal entries
    W_Kl_W = W_l.T @ stored_values["Kl"] @ W_l
    W_Kl_W_dot = W_l.T @ stored_values_dot["Kl"] @ W_l

    W_Kt_W = W_t.T @ stored_values["Kt"] @ W_t
    W_Kt_W_dot = W_t.T @ stored_values_dot["Kt"] @ W_t

    W_Sl_W = W_l.T @ stored_values["Sl"] @ W_l
    W_Sl_W_dot = W_l.T @ stored_values_dot["Sl"] @ W_l

    W_St_W = W_t.T @ stored_values["St"] @ W_t
    W_St_W_dot = W_t.T @ stored_values_dot["St"] @ W_t
    
    # This term is calculated same as before
    K_deriv = jnp.outer(jnp.diag(W_Kl_W_dot), jnp.diag(W_Kt_W))
    K_deriv += jnp.outer(jnp.diag(W_Kl_W), jnp.diag(W_Kt_W_dot))
    K_deriv += jnp.outer(jnp.diag(W_Sl_W_dot), jnp.diag(W_St_W))
    K_deriv += jnp.outer(jnp.diag(W_Sl_W), jnp.diag(W_St_W_dot))

    # For the hessian calculation this extra calculation is needed as an extra term appears in the second derivative
    # due to the product rule.
    # However, in order to avoid the first derivative being incorrectly calculated we specify which terms need to
    # have the derivative taken and which terms must not be differentiated.
    # We use the convenience functions no_deriv and must_deriv to specify which terms must not be and must be differentiated.
    K_deriv -= kron_prod(must_deriv(W_Kl_W)*no_deriv(W_Kl_W.T), no_deriv(W_Kt_W)*no_deriv(W_Kt_W_dot.T), D_inv)
    K_deriv -= kron_prod(no_deriv(W_Kl_W)*no_deriv(W_Kl_W.T), must_deriv(W_Kt_W)*no_deriv(W_Kt_W_dot.T), D_inv)
    
    K_deriv -= kron_prod(must_deriv(W_Kl_W)*no_deriv(W_Kl_W_dot.T), no_deriv(W_Kt_W)*no_deriv(W_Kt_W.T), D_inv)
    K_deriv -= kron_prod(no_deriv(W_Kl_W)*no_deriv(W_Kl_W_dot.T), must_deriv(W_Kt_W)*no_deriv(W_Kt_W.T), D_inv)
    
    K_deriv -= kron_prod(must_deriv(W_Sl_W)*no_deriv(W_Sl_W.T),     no_deriv(W_St_W)*no_deriv(W_St_W_dot.T), D_inv)
    K_deriv -= kron_prod(no_deriv(W_Sl_W)*no_deriv(W_Sl_W.T),     must_deriv(W_St_W)*no_deriv(W_St_W_dot.T), D_inv)
    
    K_deriv -= kron_prod(must_deriv(W_Sl_W)*no_deriv(W_Sl_W_dot.T), no_deriv(W_St_W)*no_deriv(W_St_W.T), D_inv)
    K_deriv -= kron_prod(no_deriv(W_Sl_W)*no_deriv(W_Sl_W_dot.T), must_deriv(W_St_W)*no_deriv(W_St_W.T), D_inv)

    K_deriv -= kron_prod(must_deriv(W_Kl_W)*no_deriv(W_Sl_W.T), no_deriv(W_Kt_W)*no_deriv(W_St_W_dot.T), D_inv)
    K_deriv -= kron_prod(no_deriv(W_Kl_W)*no_deriv(W_Sl_W.T), must_deriv(W_Kt_W)*no_deriv(W_St_W_dot.T), D_inv)
    
    K_deriv -= kron_prod(must_deriv(W_Kl_W)*no_deriv(W_Sl_W_dot.T), no_deriv(W_Kt_W)*no_deriv(W_St_W.T), D_inv)
    K_deriv -= kron_prod(no_deriv(W_Kl_W)*no_deriv(W_Sl_W_dot.T), must_deriv(W_Kt_W)*no_deriv(W_St_W.T), D_inv)
    
    K_deriv -= kron_prod(must_deriv(W_Sl_W)*no_deriv(W_Kl_W.T), no_deriv(W_St_W)*no_deriv(W_Kt_W_dot.T), D_inv)
    K_deriv -= kron_prod(no_deriv(W_Sl_W)*no_deriv(W_Kl_W.T), must_deriv(W_St_W)*no_deriv(W_Kt_W_dot.T), D_inv)
    
    K_deriv -= kron_prod(must_deriv(W_Sl_W)*no_deriv(W_Kl_W_dot.T), no_deriv(W_St_W)*no_deriv(W_Kt_W.T), D_inv)
    K_deriv -= kron_prod(no_deriv(W_Sl_W)*no_deriv(W_Kl_W_dot.T), must_deriv(W_St_W)*no_deriv(W_Kt_W.T), D_inv)

    # custom_jvp requires we return the value as well as the derivative as a tuple
    return stored_values["logdetK"], jnp.multiply(D_inv, K_deriv).sum()


@custom_jvp
def no_deriv(M: JAXArray) -> JAXArray:
    """Convenience function used for defining the derivative of ``logdetK_calc_hessianable``.
    Takes an array and will return the same array but the derivative of the array will
    be an array of zeros of the same shape. Useful for defining custom derivatives.
    
    Args:
        M (JAXArray): An array of any shaoe
        
    Returns:
        The input JAXArray ``M`` unaltered. The gradient of this function taken using ``jax.grad`` will return
        a JAXArray of zeros in the same shape as ``M``.
    
    """
    return M

@no_deriv.defjvp
def no_deriv_derivative(
    primals: JAXArray,
    tangents: JAXArray,
) -> JAXArray:
    """Custom derivative of no_deriv which returns an array of zeros if the gradient of the function is taken.
    
    """
    M, = primals
    M_dot, = tangents

    return M, jnp.zeros_like(M_dot)


@custom_jvp
def must_deriv(M: JAXArray) -> JAXArray:
    """Convenience function used for defining the derivative of ``logdetK_calc_hessianable``.
    Takes an array and will return an array of zeros of the same shape but will return
    the derivative of the array correctly. Useful for defining custom derivatives.
    
    Args:
        M (JAXArray): An array of any shape
        
    Returns:
        JAXArray: An array of zeros of the same shape as ``M``. However, the gradient of this function
        taken using ``jax.grad`` will return the correct gradient of ``M``.
    
    """
    return jnp.zeros_like(M)

@must_deriv.defjvp
def must_deriv_derivative(
    primals: JAXArray,
    tangents: JAXArray,
) -> JAXArray:
    """Custom derivative of ``must_deriv`` which returns the correct gradient of an array.
    
    """
    M, = primals
    M_dot, = tangents

    return jnp.zeros_like(M), M_dot

