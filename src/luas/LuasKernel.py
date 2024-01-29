import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import grad, value_and_grad, hessian, vmap, custom_jvp, jit
from jax.flatten_util import ravel_pytree
from copy import deepcopy
from tqdm import tqdm
from typing import Callable, Tuple, Union, Any, Optional
from functools import partial

from .luas_types import Kernel, PyTree, JAXArray, Scalar
from .kronecker_fns import kron_prod, logdetK_calc, r_K_inv_r, K_inv_vec, logdetK_calc_hessianable
from .jax_convenience_fns import array_to_pytree_2D, get_corr_mat


__all__ = [
    "diag_eigendecomp",
    "decomp_S",
    "decomp_K_tilde",
    "LuasKernel",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


def diag_eigendecomp(K: JAXArray) -> Tuple[JAXArray, JAXArray]:
    """Calculates the eigenvalues and eigenvectors of a matrix ``K`` assuming it is diagonal.
    Can be significantly faster than ``jax.numpy.linalg.eigh`` if a matrix is known in advance to be diagonal.
    
    Args:
        K (JAXArray): The matrix to decompose
        
    Returns:
        (JAXArray, JAXArray): Returns a tuple of the eigenvalues and eigenvector matrix of ``K``.
    
    """
    return jnp.diag(K), jnp.eye(K.shape[0])


def decomp_S(
    S: JAXArray,
    eigen_fn: Optional[Callable] = jnp.linalg.eigh
) -> Tuple[JAXArray, JAXArray]:
    """Calculates the eigenvalues and matrix inverse square root of a matrix ``S``. Needs to be performed
    on the ``Sl`` and ``St`` matrices for log likelihood calculations with the :class:`LuasKernel`.
    
    Args:
        S: The matrix to decompose.
        eigen_fn: The function to use to solve for the eigenvalues and eigenvectors of ``S``.
        
    Returns:
        (JAXArray, JAXArray): Returns a tuple of the eigenvalues as well as the matrix inverse square root
        of ``S``.
    
    """
    
    # Solve for eigenvalues and eigenvectors
    lam_S, Q_S = eigen_fn(S)
    
    # Calculate the matrix inverse square root
    S_inv_sqrt = Q_S @ jnp.diag(jnp.sqrt(jnp.reciprocal(lam_S)))

    return lam_S, S_inv_sqrt


def decomp_K_tilde(
    K: JAXArray,
    S_inv_sqrt: JAXArray,
    eigen_fn: Optional[Callable] = jnp.linalg.eigh
) -> Tuple[JAXArray, JAXArray]:
    r"""Creates the transformed matrix ``K_tilde`` from ``K`` and the matrix inverse square root of ``S``. 
    Then calculates the eigendecomposition of ``K_tilde``. Finally calculates the required ``W`` matrices.
    Returns the eigenvalues of ``K_tilde`` and the calculated ``W`` matrix. Needs to be separately performed on the
    ``Kl`` and ``Kt`` matrices for log likelihood calculations with the :class:`LuasKernel`.
    
    For ``K = Kl`` this function will solve for ``Kl_tilde`` using:
    
    .. math::
    
        \tilde{K}_l = (S_l^{-\frac{1}{2}})^T K_l S_l^{-\frac{1}{2}}
    
    Then eigendecompose ``Kl_tilde`` and calculate the required ``W_l`` matrix with:
    
    .. math::
        W_l = S_l^{-\frac{1}{2}} Q_{\tilde{K}_l}
    
    and similarly for ``K = Kt``.
    
    Args:
        S: The matrix to decompose.
        eigen_fn: The function to use to solve for the eigenvalues and eigenvectors of ``S``.
        
    Returns:
        (JAXArray, JAXArray): Returns a tuple of the eigenvalues and eigenvectors of the transformed matrix
        ``K_tilde``.
    
    """
    
    # Solve K_tilde
    K_tilde = S_inv_sqrt.T @ K @ S_inv_sqrt
    
    # Eigendecompose K_tilde
    lam_K_tilde, Q_K_tilde = eigen_fn(K_tilde)
    
    # Calculate the required W matrix
    W = S_inv_sqrt @ Q_K_tilde

    return lam_K_tilde, W


class LuasKernel(Kernel):
    r"""Kernel class which solves for the log likelihood for any covariance matrix which
    is the sum of two kronecker products of the covariance matrix in each of two dimensions
    i.e. the full covariance matrix K is given by:
    
    .. math::
        K = K_l \otimes K_t + S_l \otimes S_t
    
    although we can avoid calculating ``K`` for many calculations implemented here.
        
    The ``Kl`` and ``Sl`` functions should both return ``(N_l, N_l)`` matrices which will be the covariance
    matrices in the wavelength/vertical direction.
    
    The ``Kt`` and ``St`` functions should both return ``(N_t, N_t)`` matrices which will by the covariance
    matrices in the time/horizontal direction.
    
    .. code-block:: python

        >>> from luas import LuasKernel, kernels
        >>> def Kl_fn(hp, x_l1, x_l2, wn = True):
        >>> ... return hp["h"]**2*kernels.squared_exp(x_l1, x_l2, hp["l_l"])
        >>> def Kt_fn(hp, x_t1, x_t2, wn = True):
        >>> ... return kernels.squared_exp(x_t1, x_t2, hp["l_t"])
        >>> # ... And similarly for Sl_fn, St_fn
        >>> kernel = LuasKernel(Kl = Kl_fn, Kt = Kt_fn, Sl = Sl_fn, St = St_fn)
        ... )
    
    See https://luas.readthedocs.io/en/latest/tutorials.html for more detailed tutorials on how to use.
        
    Args:
        Kl (Callable): Function which returns the covariance matrix Kl, should be of the form
            ``Kl(hp, x_l1, x_l2, wn = True)``.
        Kt (Callable): Function which returns the covariance matrix Kt, should be of the form
            ``Kt(hp, x_t1, x_t2, wn = True)``.
        Sl (Callable): Function which returns the covariance matrix Sl, should be of the form
            ``Sl(hp, x_l1, x_l2, wn = True)``.
        St (Callable): Function which returns the covariance matrix St, should be of the form
            ``St(hp, x_t1, x_t2, wn = True)``.
        use_stored_values (bool, optional): Whether to perform checks if any of the component
            covariance matrices have changed and to make use of previously stored values for
            the decomposition of those matrices if they're the same. If ``False`` then will
            not perform these checks and will compute the eigendecomposition of all matrices
            for every calculation.
    
    """
    
    def __init__(
        self,
        Kl: Callable = None,
        Kt: Callable = None,
        Sl: Callable = None,
        St: Callable = None,
        use_stored_values: Optional[bool] = True,
    ):
        
        self.Kl = Kl
        self.Kt = Kt
        self.Sl = Sl
        self.St = St
           
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            self.decomp_fn = self.eigendecomp_use_stored_values
        else:
            self.decomp_fn = self.eigendecomp_no_stored_values

        # Identify how to eigendecompose each of the matrices
        # Note that for Kl and Kt it will actually be the transformed matrices
        # Kl_tilde and Kt_tilde being eigendecomposed
        for fn in [self.Sl, self.St, self.Kl, self.Kt]:
            if hasattr(fn, "decomp"):
                if fn.decomp == "diag":
                    fn.decomp = diag_eigendecomp
            else:
                fn.decomp = jnp.linalg.eigh
                
        if self.Kl.decomp == diag_eigendecomp and not self.Sl.decomp == diag_eigendecomp:
            print("""NOTE: The transformation of Kl is set to be diagonal but the matrix Sl is not
            set to diagonal.This may be possible for example if Kl is a scalar times the identity
            matrix or Kl shares the same eigenvectors as Sl but it is not true if Kl is any general
            diagonal matrix. Alternatively perhaps Sl is also diagonal and you forgot to add
            Sl.decomp = 'diag'. Be careful to ensure the transformation of Kl is diagonal or else
            log likelihood values will be incorrect!""")
        if self.Kt.decomp == diag_eigendecomp and not self.St.decomp == diag_eigendecomp:
            print("""NOTE: The transformation of Kt is set to be diagonal but the matrix St is not
            set to diagonal. This may be possible for example if Kt is a scalar times the identity
            matrix or Kt shares the same eigenvectors as St but it is not true if Kt is any general
            diagonal matrix. Alternatively perhaps St is also diagonal and you forgot to add
            St.decomp = 'diag'. Be careful to ensure the transformation of Kt is diagonal or else
            log likelihood values will be incorrect!""")

        # Specify how each of the 4 matrices will be decomposed
        self.Sl_decomp_fn = lambda Sl: decomp_S(Sl, eigen_fn = self.Sl.decomp)
        self.St_decomp_fn = lambda St: decomp_S(St, eigen_fn = self.St.decomp)
        self.Kl_tilde_decomp_fn = lambda Kl, Sl_inv_sqrt: decomp_K_tilde(Kl, Sl_inv_sqrt, eigen_fn = self.Kl.decomp)
        self.Kt_tilde_decomp_fn = lambda Kt, St_inv_sqrt: decomp_K_tilde(Kt, St_inv_sqrt, eigen_fn = self.Kt.decomp)

    
    def eigendecomp_no_stored_values(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:
        """Required calculations for the decomposition of the overall matrix ``K`` where the previously
        stored decomposition of ``K`` cannot be used for the calculation of a new decomposition.
        This avoids checking if any of the matrices have changed but may result in performing the
        same eigendecomposition calculations multiple times.
        
        We can decompose the inverse of ``K`` into the matrices:

        .. math::
        
            K^{-1} = [W_l \otimes W_t] D^{-1} [W_l^T \otimes W_t^T]
        
        Where this function will calculate ``W_l``, ``W_t`` and ``D_inv`` and stored them in the
        ``stored_values`` PyTree for future log likelihood calculations.
        
        Note:
            Values still need to be stored for any log likelihood calculations so this method does
            not save memory over ``eigendecomp_use_stored_values``. It may however reduce runtimes
            by avoiding checking if matrices have changed so it could be beneficial if all hyperparameters
            are being varied simultaneously for each calculation.
            
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            stored_values (PyTree): This may contain stored values from the decomposition of ``K`` but
                this method will not make use of it. This dictionary will simply be overwritten with
                new stored values from the decomposition of ``K``.
        
        Returns:
            PyTree: Stored values from the decomposition of the covariance matrices. For
            :class:`LuasKernel` this consists of values computed using the eigendecomposition
            of each matrix and also the log determinant of ``K``.
        
        """
        
        # Calculate each of the four component matrices and decompose them into the required matrices for log likelihood calculations
        stored_values["Sl"] = self.Sl(hp, x_l, x_l)
        stored_values["lam_Sl"], stored_values["Sl_inv_sqrt"] = self.Sl_decomp_fn(stored_values["Sl"])

        stored_values["St"] = self.St(hp, x_t, x_t)
        stored_values["lam_St"], stored_values["St_inv_sqrt"] = self.St_decomp_fn(stored_values["St"])

        # See decomp_K_tilde for how W_l and the eigenvalues of Kl_tilde are calculated
        stored_values["Kl"] = self.Kl(hp, x_l, x_l)
        stored_values["lam_Kl_tilde"], stored_values["W_l"] = self.Kl_tilde_decomp_fn(stored_values["Kl"], stored_values["Sl_inv_sqrt"])
        
        stored_values["Kt"] = self.Kt(hp, x_t, x_t)
        stored_values["lam_Kt_tilde"], stored_values["W_t"] = self.Kt_tilde_decomp_fn(stored_values["Kt"], stored_values["St_inv_sqrt"])

        # D is needed for calculation the log determinant of K
        D = jnp.outer(stored_values["lam_Kl_tilde"], stored_values["lam_Kt_tilde"]) + 1.
        
        # D^-1 is needed for calculating K^-1 r
        stored_values["D_inv"] = jnp.reciprocal(D)

        # Computes the log determinant of K
        lam_S = jnp.outer(stored_values["lam_Sl"], stored_values["lam_St"])
        stored_values["logdetK"] = jnp.log(jnp.multiply(D, lam_S)).sum()
        
        return stored_values

    
    def eigendecomp_use_stored_values(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray, 
        stored_values: Optional[PyTree] = {},
        rtol: Optional[Scalar] = 1e-12,
        atol: Optional[Scalar] = 1e-12,
    ) -> PyTree:
        """Required calculations for the decomposition of the overall matrix ``K`` where the previously
        stored decomposition of ``K`` may be used for the calculation of a new decomposition.
        This checking if any of the matrices have changed and if they are similar within the given
        tolerances a previously computed eigendecomposition can be used to avoid recalculating it.
        This can provide significant runtime savings if some hyperparameters are being kept fixed
        including if blocked Gibbs sampling is being used on groups of hyperparameters.
        
        We can decompose the inverse of ``K`` into the matrices:

        .. math::
        
            K^{-1} = [W_l \otimes W_t] D^{-1} [W_l^T \otimes W_t^T]
        
        Where this function will calculate ``W_l``, ``W_t`` and ``D_inv`` and stored them in the
        stored_values PyTree for future log likelihood calculations.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            stored_values (PyTree): Stored values from the decomposition of the covariance matrices. For
                :class:`LuasKernel` this consists of values computed using the eigendecomposition
                of each matrix and also the log determinant of ``K``.
            rtol (Scalar): The relative tolerance value any of the component covariance matrices
                must be within in order for the matrix to be considered unchanged and stored values for
                its decomposition to be used.
            atol (Scalar): The absolute tolerance values any of the component covariance matrices
                must be within in order for the matrix to be considered unchanged and stored values for
                its decomposition to be used.
        
        Returns:
            PyTree: Stored values from the decomposition of the covariance matrices. For
            :class:`LuasKernel` this consists of values computed using the eigendecomposition
            of each matrix and also the log determinant of ``K``.
        
        """

        stored_values = deepcopy(stored_values)
        
        # Calculate each of the four component matrices
        Sl = self.Sl(hp, x_l, x_l)
        St = self.St(hp, x_t, x_t)
        Kl = self.Kl(hp, x_l, x_l)
        Kt = self.Kt(hp, x_t, x_t)
        
        if stored_values:
            # Check if any of the 4 component matrices have changed from their values in stored_values
            
            # Note JAX requires the two possible outputs of the conditional to be functions
            # so we use functions which just return True or False
            Sl_diff = jax.lax.cond(jnp.allclose(Sl, stored_values["Sl"], rtol = rtol, atol = atol),
                                   lambda : False, lambda : True)
            St_diff = jax.lax.cond(jnp.allclose(St, stored_values["St"], rtol = rtol, atol = atol),
                                   lambda : False, lambda : True)
            
            # Note that if Sl is different than Kl_tilde is also almost certainly different
            # so even if Kl hasn't changed we still need to recompute the decomposition of Kl_tilde and similarly for Kt
            Kl_tilde_diff = jax.lax.cond(jnp.allclose(Kl, stored_values["Kl"], rtol = rtol, atol = atol),
                                        lambda : Sl_diff, lambda : True)
            Kt_tilde_diff = jax.lax.cond(jnp.allclose(Kt, stored_values["Kt"], rtol = rtol, atol = atol),
                                        lambda : St_diff, lambda : True)
        else:
            Sl_diff = St_diff = Kl_tilde_diff = Kt_tilde_diff = True
            
            N_l = x_l.shape[-1]
            N_t = x_t.shape[-1]
            
            # JAX requires that the two outputs of any conditional statements have the same shape
            # so must define matrices of same shape as their actual values even though they will be overwritten
            stored_values["lam_Sl"] = jnp.zeros(N_l)
            stored_values["Sl_inv_sqrt"] = jnp.zeros((N_l, N_l))
            stored_values["lam_St"] = jnp.zeros(N_t)
            stored_values["St_inv_sqrt"] = jnp.zeros((N_t, N_t))
            stored_values["lam_Kl_tilde"] = jnp.zeros(N_l)
            stored_values["W_l"] = jnp.zeros((N_l, N_l))
            stored_values["lam_Kt_tilde"] = jnp.zeros(N_t)
            stored_values["W_t"] = jnp.zeros((N_t, N_t))


        # For each of the 4 component matrices conditionally decompose them if they have changed since the last calculation
        stored_values["lam_Sl"], stored_values["Sl_inv_sqrt"] = jax.lax.cond(Sl_diff,
                                                                           self.Sl_decomp_fn,
                                                                           lambda *args: (stored_values["lam_Sl"],
                                                                                          stored_values["Sl_inv_sqrt"]),
                                                                           Sl)
        
        stored_values["lam_St"], stored_values["St_inv_sqrt"] = jax.lax.cond(St_diff,
                                                                           self.St_decomp_fn,
                                                                           lambda *args: (stored_values["lam_St"], 
                                                                                          stored_values["St_inv_sqrt"]),
                                                                           St)

        stored_values["lam_Kl_tilde"], stored_values["W_l"] = jax.lax.cond(Kl_tilde_diff,
                                                                         self.Kl_tilde_decomp_fn,
                                                                         lambda *args: (stored_values["lam_Kl_tilde"],
                                                                                        stored_values["W_l"]),
                                                                         Kl, stored_values["Sl_inv_sqrt"])

        stored_values["lam_Kt_tilde"], stored_values["W_t"] = jax.lax.cond(Kt_tilde_diff,
                                                                         self.Kt_tilde_decomp_fn,
                                                                         lambda *args: (stored_values["lam_Kt_tilde"],
                                                                                        stored_values["W_t"]),
                                                                         Kt, stored_values["St_inv_sqrt"])

        # D is needed for calculation the log determinant of K
        D = jnp.outer(stored_values["lam_Kl_tilde"], stored_values["lam_Kt_tilde"]) + 1.
        
        # D^-1 is needed for calculating K^-1 r
        stored_values["D_inv"] = jnp.reciprocal(D)

        # Computes the log determinant of K
        lam_S = jnp.outer(stored_values["lam_Sl"], stored_values["lam_St"])
        stored_values["logdetK"] = jnp.log(jnp.multiply(D, lam_S)).sum()

        # Store in order to perform checks for the next call of this function
        stored_values["Sl"] = Sl
        stored_values["St"] = St
        stored_values["Kl"] = Kl
        stored_values["Kt"] = Kt

        return stored_values
    
    
    def logL(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        R: JAXArray,
        stored_values: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        """Computes the log likelihood using the method originally presented in Rakitsch et al. (2013)
        and also outlined in Fortune at al. (2024). Also returns stored values from the matrix decomposition.
        
        Note:
            Calculating the hessian of this function with ``jax.hessian`` may not produce numerically stable
            results. ``LuasKernel.logL_hessianable`` is recommended is values of the hessian are needed.
            This method typically outperforms ``LuasKernel.logL_hessianable`` in runtime for gradient
            calculations however.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            R (JAXArray): Residuals to be fit calculated from the observed data by subtracting the deterministic
                mean function. Must have the same shape as the observed data (N_l, N_t).
            stored_values (PyTree): Stored values from the decomposition of the covariance matrices. For
                :class:`LuasKernel` this consists of values computed using the eigendecomposition
                of each matrix and also the log determinant of ``K``.
        
        Returns:
            (Scalar, PyTree): A tuple where the first element is the value of the log likelihood.
            The second element is a PyTree which contains stored values from the decomposition of the
            covariance matrix.
            
        """

        # Calculate the decomposition of K
        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = stored_values)
        
        # Use functions with custom derivatives to accurately calculate the log
        # likelihood and its gradient
        rKr = r_K_inv_r(R, stored_values)
        logdetK = logdetK_calc(stored_values)
        logL = -0.5 * rKr - 0.5 * logdetK  - 0.5 * R.size * jnp.log(2*jnp.pi)

        return  logL, stored_values

    
    def logL_hessianable(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        R: JAXArray,
        stored_values: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        """Computes the log likelihood using the method originally presented in Rakitsch et al. (2013)
        and also outlined in Fortune at al. (2024).
        
        Note:
            The hessian of this log likelihood function can be calculated using ``jax.hessian`` and
            should be more numerically stable for this than ``LuasKernel.logL``.
            However, this function is slower for calculating the gradients of the log likelihood so
            ``LuasKernel.logL`` is preferred unless the hessian is needed. Also returns stored values
            from the matrix decomposition.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            R (JAXArray): Residuals to be fit calculated from the observed data by subtracting the deterministic
                mean function. Must have the same shape as the observed data (N_l, N_t).
            stored_values (PyTree): Stored values from the decomposition of the covariance matrices. For
                :class:`LuasKernel` this consists of values computed using the eigendecomposition
                of each matrix and also the log determinant of ``K``.
                
        Returns:
            (Scalar, PyTree): A tuple where the first element is the value of the log likelihood.
            The second element is a PyTree which contains stored values from the decomposition of the
            covariance matrix.
        
        """
        
        # Calculate the decomposition of K
        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = stored_values)
        
        # Use functions with custom derivatives to accurately calculate the log
        # likelihood, its gradient and hessian
        rKr = r_K_inv_r(R, stored_values)
        logdetK = logdetK_calc_hessianable(stored_values)
        logL =  -0.5 * rKr - 0.5 * logdetK  - 0.5 * R.size * jnp.log(2*jnp.pi)

        return  logL, stored_values
        
    
    def predict(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_l_pred: JAXArray,
        x_t: JAXArray,
        x_t_pred: JAXArray,
        R: JAXArray,
        M_s: JAXArray,
        stored_values: Optional[PyTree] = {},
        wn = True,
        return_std_dev = True,
    ) -> Tuple[JAXArray, JAXArray]:
        r"""Performs GP regression and computes the GP predictive mean and the GP predictive
        uncertainty as the standard devation at each location or else can return the full
        covariance matrix. Requires the input kernel function ``K`` to have a ``wn`` keyword
        argument that defines the kernel when white noise is included (``wn = True``) and
        when white noise isn't included (``wn = False``).
        
        Currently assumes the same input hyperparameters for both the observed and predicted
        locations. The predicted locations ``x_l_pred`` and ``x_t_pred`` may deviate from
        the observed locations ``x_l`` and ``x_t`` however.
        
        The GP predictive mean is defined as:
        
        .. math::

            \mathbb{E}[\vec{y}_*] = \vec{\mu}_* + \mathbf{K}_*^T \mathbf{K}^{-1} \vec{r}
            
        And the GP predictive covariance is given by:
        
        .. math::
            Var[\vec{y}_*] = \mathbf{K}_{**} - \mathbf{K}_*^T \mathbf{K}^{-1} \mathbf{K}_*
        
        Note:
            The calculation of the full predictive covariance matrix when ``return_std_dev = False``
            is still experimental and may come with numerically stability issues. It is also very
            memory intensive and may cause code to crash.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_l_pred (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the prediction locations (which may be the same as the observed locations).
                May be of shape ``(N_l_pred,)`` or ``(d_l,N_l_pred)`` for ``d_l`` different
                wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            x_t_pred (JAXArray): Array containing time/horizontal dimension regression variable(s) for
                the prediction locations (which may be the same as the observed locations). May be of
                shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different time/horizontal regression variables.
            R (JAXArray): Residuals to be fit, equal to the observed data minus the deterministic
                mean function. Must have the same shape as the observed data ``(N_l, N_t)``.
            M_s (JAXArray): Mean function evaluated at the locations of the predictions ``x_l_pred``, ``x_t_pred``.
                Must have shape ``(N_l_pred, N_t_pred)`` where ``N_l_pred`` is the number of wavelength/vertical
                dimension predictions and ``N_t_pred`` the number of time/horizontal dimension predictions.
            stored_values (PyTree): Stored values from the decomposition of the covariance matrices. For
                :class:`LuasKernel` this consists of values computed using the eigendecomposition
                of each matrix and also the log determinant of ``K``.
            wn (bool, optional): Whether to include white noise in the uncertainty at the predicted locations.
                Defaults to True.
            return_std_dev (bool, optional): If ``True`` will return the standard deviation of uncertainty at the predicted
                locations. Otherwise will return the full predictive covariance matrix. Defaults to True.
        
        Returns:
            (JAXArray, JAXArray): Returns a tuple of two elements, where the first element is
            the GP predictive mean at the prediction locations, the second element is either the
            standard deviation of the predictions if ``return_std_dev = True``, otherwise it will be
            the full covariance matrix of the predicted values.
        
        """
        # Calculate the decomposition of K
        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = stored_values)
        
        # Calculate the covariance between the observed and predicted points
        Kl_s = self.Kl(hp, x_l, x_l_pred, wn = False)
        Kt_s = self.Kt(hp, x_t, x_t_pred, wn = False)
        Sl_s = self.Sl(hp, x_l, x_l_pred, wn = False)
        St_s = self.St(hp, x_t, x_t_pred, wn = False)
        
        # Calculate the covariance between predicted points with other predicted points
        Kl_ss = self.Kl(hp, x_l_pred, x_l_pred, wn = wn)
        Kt_ss = self.Kt(hp, x_t_pred, x_t_pred, wn = wn)
        Sl_ss = self.Sl(hp, x_l_pred, x_l_pred, wn = wn)
        St_ss = self.St(hp, x_t_pred, x_t_pred, wn = wn)

        # Calculate K^-1 R
        K_inv_R = K_inv_vec(R, stored_values)

        # Calculates the GP mean including the deterministic mean function at the prediction locations
        gp_mean = M_s + kron_prod(Kl_s.T, Kt_s.T, K_inv_R) + kron_prod(Sl_s.T, St_s.T, K_inv_R)

        # Prepare matrices for calculating the predictive covariance
        KW_l = Kl_s.T @ stored_values["W_l"]
        KW_t = Kt_s.T @ stored_values["W_t"]
        SW_l = Sl_s.T @ stored_values["W_l"]
        SW_t = St_s.T @ stored_values["W_t"]

        if return_std_dev:
            # Efficiently solves for the diagonal of the predictive covariance
            pred_err = jnp.outer(jnp.diag(Kl_ss), jnp.diag(Kt_ss))
            pred_err += jnp.outer(jnp.diag(Sl_ss), jnp.diag(St_ss))

            # K_s.T K^-1 K_s term can be broken into these three terms
            pred_err -= kron_prod(KW_l**2, KW_t**2, stored_values["D_inv"])
            pred_err -= kron_prod(SW_l**2, SW_t**2, stored_values["D_inv"])
            pred_err -= 2*kron_prod(KW_l * SW_l, KW_t * SW_t, stored_values["D_inv"])
            
            # Take the sqrt of the diagonal to get the std dev
            pred_err = jnp.sqrt(pred_err)
            
        else:
            # Get the length of each prediction dimension
            N_l_pred = x_l_pred.shape[-1]
            N_t_pred = x_t_pred.shape[-1]

            # Useful to define to calculate elementwise products between different columns
            def K_mult(K1, K2):
                return K1*K2
            vmap_K_mult = jax.vmap(K_mult, in_axes = (0, None), out_axes = 0)

            # First solve for the predictive covariance but in a matrix that will be
            # of shape (N_l_pred*N_l_pred, N_t_pred*N_t_pred)
            cov_wrong_order = jnp.zeros((N_l_pred**2, N_t_pred**2))
            for (Kl1, Kt1) in [(KW_l, KW_t), (SW_l, SW_t)]:
                for (Kl2, Kt2) in [(KW_l, KW_t), (SW_l, SW_t)]:

                    Kl_cube = vmap_K_mult(Kl1, Kl2)
                    Kt_cube = vmap_K_mult(Kt1, Kt2)

                    Kl_cube = Kl_cube.reshape((N_l_pred**2, N_l_pred))
                    Kt_cube = Kt_cube.reshape((N_t_pred**2, N_t_pred))

                    cov_wrong_order += (Kl_cube @ stored_values["D_inv"] @ Kt_cube.T)

            # Begin reshaping to the correct shape of (N_l_pred*N_t_pred, N_l_pred*N_t_pred)
            cov_wrong_order = cov_wrong_order.reshape((N_l_pred**2*N_t_pred, N_t_pred))
            pred_err = jnp.zeros((N_l_pred*N_t_pred, N_l_pred*N_t_pred))
            
            # Loops through blocks of rows placing elements into the correct order
            for j in range(N_l_pred):
                cov_wrt_x_l_j = cov_wrong_order[j*N_l_pred*N_t_pred:(j+1)*N_l_pred*N_t_pred, :]
                pred_err = pred_err.at[:, j*N_t_pred:(j+1)*N_t_pred].set(-cov_wrt_x_l_j)

            # Add the K_ss term
            pred_err += jnp.kron(Kl_ss, Kt_ss) + jnp.kron(Sl_ss, St_ss)
        
        return gp_mean, pred_err


    def generate_noise(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        size: Optional[int] = 1,
    ) -> JAXArray:
        r"""Generate noise with the covariance matrix returned by this kernel using the input
        hyperparameters ``hp``.
        
        Solves for the matrix square root of K and then multiplies this by a random normal vector.
        Doing it this way has numerical stability advantages over generating noise separately for
        each of the two kronecker products of K as they might not both be well-conditioned matrices.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            size (int, optional): The number of different draws of noise to generate. Defaults to 1.
                
        Returns:
            JAXArray: If ``size = 1`` will generate noise of shape ``(N_l, N_t)``, otherwise if ``size > 1`` then
            generated noise will be of shape ``(N_l, N_t, size)``.
        
        """
        
        N_l = x_l.shape[-1]
        N_t = x_t.shape[-1]
        
        # Solve for the matrix sqrt and matrix inv sqrt for Sl and St
        Sl = self.Sl(hp, x_l, x_l)
        lam_Sl, Q_Sl = self.Sl.decomp(Sl)
        Sl_sqrt = Q_Sl @ jnp.diag(jnp.sqrt(lam_Sl))
        Sl_inv_sqrt = Q_Sl @ jnp.diag(jnp.sqrt(jnp.reciprocal(lam_Sl)))

        St = self.St(hp, x_t, x_t)
        lam_St, Q_St = self.St.decomp(St)
        St_sqrt = Q_St @ jnp.diag(jnp.sqrt(lam_St))
        St_inv_sqrt = Q_St @ jnp.diag(jnp.sqrt(jnp.reciprocal(lam_St)))

        # Solve for the eigenvalues and eigenvectors of Kl_tilde, Kt_tilde
        Kl = self.Kl(hp, x_l, x_l)
        Kl_tilde = Sl_inv_sqrt.T @ Kl @ Sl_inv_sqrt
        lam_Kl_tilde, Q_Kl_tilde = self.Kl.decomp(Kl_tilde)
        
        Kt = self.Kt(hp, x_t, x_t)
        Kt_tilde = St_inv_sqrt.T @ Kt @ St_inv_sqrt
        lam_Kt_tilde, Q_Kt_tilde = self.Kt.decomp(Kt_tilde)

        # Computes the sqrt of the diagonal matrix D
        D_half = jnp.sqrt(jnp.outer(lam_Kl_tilde, lam_Kt_tilde) + 1.)
        D_half = D_half.reshape((N_l, N_t, 1))
    
        # vmap kron_prod so that it will work for z of shape (N_l, N_t, size)
        kron_prod_vmap = jax.vmap(kron_prod, in_axes = (None, None, 2), out_axes = 2)
        
        # Generate random normal vector
        z = np.random.normal(size = (N_l, N_t, size))
        
        # Multiply by the matrix sqrt of K
        z = jnp.multiply(D_half, z)
        z = kron_prod_vmap(Q_Kl_tilde, Q_Kt_tilde, z)
        R = kron_prod_vmap(Sl_sqrt, St_sqrt, z)
        
        # If size = 1 then return as shape (N_l, N_t) instead of (N_l, N_t, 1)
        if size == 1:
            R = R.reshape((N_l, N_t))

        return R


    def K(
        self,
        hp: PyTree,
        x_l1: JAXArray,
        x_l2: JAXArray,
        x_t1: JAXArray,
        x_t2: JAXArray,
        **kwargs,
    ) -> JAXArray:
        r"""Generates the full covariance matrix K formed from the sum of two kronecker products:
        
        .. math::

            K = K_l \otimes K_t + S_l \otimes S_t
        
        Not needed for any calculations with the ``LuasKernel`` but useful for creating a :class:`GeneralKernel`
        object with the same kernel function as a :class:`LuasKernel`.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
            x_l1 (JAXArray): The first array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_l2 (JAXArray): Second array containing wavelength/vertical dimension regression variable(s).
            x_t1 (JAXArray): The first array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            x_t2 (JAXArray): Second array containing time/horizontal dimension regression variable(s).
        
        Returns:
            JAXArray: The full covariance matrix K of shape ``(N_l*N_t, N_l*N_t)``.
        
        """

        # Build 4 component matrices
        Kl = self.Kl(hp, x_l1, x_l2, **kwargs)
        Kt = self.Kt(hp, x_t1, x_t2, **kwargs)
        Sl = self.Sl(hp, x_l1, x_l2, **kwargs)
        St = self.St(hp, x_t1, x_t2, **kwargs)
        
        K = jnp.kron(Kl, Kt) + jnp.kron(Sl, St)
        
        return K
    
    
    def visualise_covariance_in_data(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        i: int,
        j: int,
        corr: Optional[bool] = False,
        wn: Optional[bool] = True,
        x_l_plot: Optional[JAXArray] = None,
        x_t_plot: Optional[JAXArray] = None,
        **kwargs,
    ) -> plt.Figure:
        """Creates a plot to aid in visualising how the kernel function is defining the covariance between
        different points in the observed data. Calculates the covariance of each point in the observed data
        with a point located at ``(i, j)`` in the observed data. The plot then displays this covariance using
        ``plt.pcolormesh`` with every other point in the observed data.
        
        If ``corr = True`` this will display the correlation instead of the covariance. Also if ``wn = False``
        then white noise will be excluded from the calculation of the covariance/correlation between each point.
        This can be helpful if the white noise has a much larger amplitude than correlated noise which can make
        it difficult to visualise how points are correlated.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            i (int): The wavelength/vertical location of the point to visualise covariance with.
            j (int): The time/horizontal location of the point to visualise covariance with.
            corr (bool, optional): If ``True`` will plot the correlation between points instead of the
                covariance. Defaults to ``False``.
            wn (bool, optional): Whether to include white noise in the calculation of covariance.
                Defaults to ``True``.
            x_l_plot (JAXArray, optional): The values on the y-axis used by ``plt.pcolormesh`` for the plot.
                If not included will default to ``x_l`` if ``x_l`` is of shape ``(N_l,)`` or to ``x_l[0, :]``
                if ``x_l`` is of shape ``(d_l, N_l)``.
            x_t_plot (JAXArray, optional): The values on the x-axis used by ``plt.pcolormesh`` for the plot.
                If not included will default to ``x_t`` if ``x_t`` is of shape ``(N_t,)`` or to ``x_t[0, :]``
                if ``x_t`` is of shape ``(d_t, N_t)``.
        
        Returns:
            plt.Figure: A figure displaying the covariance of each point in the observed data with the
            selected point located at ``(i, j)`` in the observed data ``Y``.
        
        """
        
        # If no x and y axes for the plots specified, defaults to x_l, x_t
        # If x_l or x_t contain multiple rows then pick the first row
        if x_l_plot is None:
            if x_l.ndim == 1:
                x_l_plot = x_l
            else:
                x_l_plot = x_l[0, :]
                
        if x_t_plot is None:
            if x_t.ndim == 1:
                x_t_plot = x_t
            else:
                x_t_plot = x_t[0, :]
                
        # Build 4 component matrices
        Kl = self.Kl(hp, x_l1, x_l2, wn = wn)
        Kt = self.Kt(hp, x_t1, x_t2, wn = wn)
        Sl = self.Sl(hp, x_l1, x_l2, wn = wn)
        St = self.St(hp, x_t1, x_t2, wn = wn)
        
        # Calculate covariance wrt point at (i, j)
        Kl_i = Kl[i, :]
        Kt_j = Kt[j, :]
        Sl_i = Sl[i, :]
        St_j = St[j, :]
        
        # Calculate covariance with the same shape as the observed data Y
        cov = jnp.outer(Kl_i, Kt_j) + jnp.outer(Sl_i, St_j)
        
        if corr:
            # If calculating the correlation matrix must divide by the standard deviation along
            # each row and column of the covariance matrix
            Kl_diag = jnp.diag(Kl)
            Kt_diag = jnp.diag(Kt)
            Sl_diag = jnp.diag(Sl)
            St_diag = jnp.diag(St)
            
            cov /= jnp.sqrt(cov[i, j]*(jnp.outer(Kl_diag, Kt_diag) + jnp.outer(Sl_diag, St_diag)))
            
        # Generate plot as a pcolormesh
        fig = plt.pcolormesh(x_t_plot, x_l_plot, cov, **kwargs)
        plt.gca().invert_yaxis()
        plt.xlabel("$x_t$")
        plt.ylabel("$x_l$")
        plt.colorbar()
        
        return fig
    
    
    def visualise_covariance_matrix(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        corr: Optional[bool] = False,
        wn: Optional[bool] = True,
        x_l_plot: Optional[JAXArray] = None,
        x_t_plot: Optional[JAXArray] = None,
        full: Optional[bool] = False,
    ) -> plt.Figure:
        """Visualise the covariance matrix/matrices generated by the input hyperparameters.
        
        Note:
            Default behaviour is to separately visualise each of the 4 component covariance matrices
            ``Kl``, ``Kt``, ``Sl``, ``St`` which are used to calculate the full covariance matrix ``K``.
            If ``full = True`` then will instead build the full covariance matrix ``K`` but this is very
            memory intensive as it requires creating a JAXArray with ``(N_l*N_t, N_l*N_t)`` entries.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            corr (bool, optional): If ``True`` will plot the correlation between points instead of the
                covariance. Defaults to ``False``.
            wn (bool, optional): Whether to include white noise in the calculation of covariance.
                Defaults to ``True``.
            x_l_plot (JAXArray, optional): The values on the y-axis used by ``plt.pcolormesh`` for the plot.
                If not included will default to ``x_l`` if ``x_l`` is of shape ``(N_l,)`` or to ``x_l[0, :]``
                if ``x_l`` is of shape ``(d_l, N_l)``.
            x_t_plot (JAXArray, optional): The values on the x-axis used by ``plt.pcolormesh`` for the plot.
                If not included will default to ``x_t`` if ``x_t`` is of shape ``(N_t,)`` or to ``x_t[0, :]``
                if ``x_t`` is of shape ``(d_t, N_t)``.
            full (bool, optional): If ``True`` will build and visualise the full constructed covariance matrix
        
        Returns:
            plt.Figure: A figure displaying the covariance of each point in the observed data with the
            selected point located at ``(i, j)`` in the observed data ``Y``.
        
        """
    
        # If no x and y axes for the plots specified, defaults to x_l, x_t
        # If x_l or x_t contain multiple rows then pick the first row
        if x_l_plot is None:
            if x_l.ndim == 1:
                x_l_plot = x_l
            else:
                x_l_plot = x_l[0, :]
                
        if x_t_plot is None:
            if x_t.ndim == 1:
                x_t_plot = x_t
            else:
                x_t_plot = x_t[0, :]
        
        fig, ax = plt.subplots(2, 2, figsize = (10, 10))

        # Build each component matrix
        Kl = self.Kl(hp, x_l, x_l, wn = wn)
        Kt = self.Kt(hp, x_t, x_t, wn = wn)
        Sl = self.Sl(hp, x_l, x_l, wn = wn)
        St = self.St(hp, x_t, x_t, wn = wn)
        

        if full:
            # Plot full covariance matrix K
            # Warning: Can be very memory intensive as it builds an array with (N_l*N_t)**2 entries
            K = jnp.kron(Kl, Kt) + jnp.kron(Sl, St)
            
            if corr:
                K = get_corr_mat(K)
                
            fig = plt.imshow(K)
            
        else:
            # Individually plot each of the 4 component covariance matrices
            
            if corr:
                # Convert to correlation matrices if corr = True
                Kl = get_corr_mat(Kl)
                Kt = get_corr_mat(Kt)
                Sl = get_corr_mat(Sl)
                St = get_corr_mat(St)
            
            # Separately plot each of the 4 component matrices
            ax[0][0].set_title("K$_l$")
            ax[0][0].set_ylabel("$x_l$")
            ax[0][0].set_xlabel("$x_l$")
            img1 = ax[0][0].pcolormesh(x_l_plot, x_l_plot, Kl)
            ax[0][0].invert_yaxis()
            plt.colorbar(mappable = img1, ax = ax[0][0])

            ax[0][1].set_ylabel("$x_t$")
            ax[0][1].set_xlabel("$x_t$")
            ax[0][1].set_title("K$_t$")
            img2 = ax[0][1].pcolormesh(x_t_plot, x_t_plot, Kt)
            ax[0][1].invert_yaxis()
            plt.colorbar(mappable = img2, ax = ax[0][1])

            ax[1][0].set_ylabel("$x_l$")
            ax[1][0].set_xlabel("$x_l$")
            ax[1][0].set_title("$\Sigma_l$")
            img3 = ax[1][0].pcolormesh(x_l_plot, x_l_plot, Sl)
            ax[1][0].invert_yaxis()
            plt.colorbar(mappable = img3, ax = ax[1][0])

            ax[1][1].set_ylabel("$x_t$")
            ax[1][1].set_xlabel("$x_t$")
            ax[1][1].set_title("$\Sigma_t$")
            img4 = ax[1][1].pcolormesh(x_t_plot, x_t_plot, St)
            ax[1][1].invert_yaxis()
            plt.colorbar(mappable = img4, ax = ax[1][1])
            plt.tight_layout()

        return fig
