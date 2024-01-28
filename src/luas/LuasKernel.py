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
    "LuasKernel",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


class LuasKernel(Kernel):
    r"""Kernel object which solves for the log likelihood for any covariance matrix which
    is the sum of two kronecker products of the covariance matrix in each of two dimensions.
    
    The Kl and Sl functions should both return ``(N_l, N_l)`` matrices which will be the covariance
    matrices in the wavelength/vertical direction.
    
    The Kt and St functions should both return ``(N_t, N_t)`` matrices which will by the covariance
    matrices in the time/horizontal direction.
    
    The full covariance matrix K is given by:
    
    .. math::
        K = K_l \otimes K_t + S_l \otimes S_t
        
    although it is not required to be computed for log likelihood calculations.
        
    Args:
        Kl (Callable): Function which returns the covariance matrix Kl, should be of the form
            ``Kl(hp, x_l1, x_l2, wn = True)``.
        Kt (Callable): Function which returns the covariance matrix Kt, should be of the form
            ``Kt(hp, x_t1, x_t2, wn = True)``.
        Sl (Callable): Function which returns the covariance matrix Sl, should be of the form
            ``Sl(hp, x_l1, x_l2, wn = True)``.
        St (Callable): Function which returns the covariance matrix St, should be of the form
            ``St(hp, x_t1, x_t2, wn = True)``.
        use_stored_results (bool, optional): Whether to perform checks if any of the component
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
        use_stored_results: Optional[bool] = True,
    ):
        
        self.Kl = Kl
        self.Kt = Kt
        self.Sl = Sl
        self.St = St
           
        if use_stored_results:
            self.decomp_fn = self.eigendecomp_use_stored_results
        else:
            self.decomp_fn = self.eigendecomp_no_stored_results

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

    
    def eigendecomp_no_stored_results(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        storage_dict: Optional[PyTree] = {},
    ) -> PyTree:
        """
        
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
            storage_dict (PyTree): This may contain stored values from the decomposition of ``K`` but
                this method will not make use of it. This dictionary will simply be overwritten with
                new stored values from the decomposition of ``K``.
        
        Returns:
            PyTree: Stored values from the decomposition of the covariance matrices. For
            :class:`LuasKernel` this consists of values computed using the eigendecomposition
            of each matrix and also the log determinant of ``K``.
        
        """
            
        storage_dict["Sl"] = self.Sl(hp, x_l, x_l)
        storage_dict["lam_Sl"], storage_dict["Sl_inv_sqrt"] = self.Sl_decomp_fn(storage_dict["Sl"])

        storage_dict["St"] = self.St(hp, x_t, x_t)
        storage_dict["lam_St"], storage_dict["St_inv_sqrt"] = self.St_decomp_fn(storage_dict["St"])

        storage_dict["Kl"] = self.Kl(hp, x_l, x_l)
        storage_dict["lam_Kl_tilde"], storage_dict["W_l"] = self.Kl_tilde_decomp_fn(storage_dict["Kl"], storage_dict["Sl_inv_sqrt"])
        
        storage_dict["Kt"] = self.Kt(hp, x_t, x_t)
        storage_dict["lam_Kt_tilde"], storage_dict["W_t"] = self.Kt_tilde_decomp_fn(storage_dict["Kt"], storage_dict["St_inv_sqrt"])

        D = jnp.outer(storage_dict["lam_Kl_tilde"], storage_dict["lam_Kt_tilde"]) + 1.
        storage_dict["D_inv"] = jnp.reciprocal(D)

        lam_S = jnp.outer(storage_dict["lam_Sl"], storage_dict["lam_St"])
        storage_dict["logdetK"] = jnp.log(jnp.multiply(D, lam_S)).sum()
        
        return storage_dict

    
    def eigendecomp_use_stored_results(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray, 
        storage_dict: Optional[PyTree] = {},
        rtol: Optional[Scalar] = 1e-12,
        atol: Optional[Scalar] = 1e-12,
    ) -> PyTree:
        """
        Generate
        
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
            storage_dict (PyTree): Stored values from the decomposition of the covariance matrices. For
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

        storage_dict = deepcopy(storage_dict)
        
        Sl = self.Sl(hp, x_l, x_l)
        St = self.St(hp, x_t, x_t)
        Kl = self.Kl(hp, x_l, x_l)
        Kt = self.Kt(hp, x_t, x_t)

        N_l = Sl.shape[0]
        N_t = St.shape[0]

        
        if storage_dict: 
            Sl_diff = jax.lax.cond(jnp.allclose(Sl, storage_dict["Sl"], rtol = rtol, atol = atol),
                                   lambda hp: False, lambda hp: True, hp)
            St_diff = jax.lax.cond(jnp.allclose(St, storage_dict["St"], rtol = rtol, atol = atol),
                                   lambda hp: False, lambda hp: True, hp)
            Kl_diff = jax.lax.cond(jnp.allclose(Kl, storage_dict["Kl"], rtol = rtol, atol = atol),
                                   lambda hp: Sl_diff, lambda hp: True, hp)
            Kt_diff = jax.lax.cond(jnp.allclose(Kt, storage_dict["Kt"], rtol = rtol, atol = atol),
                                   lambda hp: St_diff, lambda hp: True, hp)
        else:
            Sl_diff = St_diff = Kl_diff = Kt_diff = True
            
            storage_dict["lam_Sl"] = jnp.zeros(N_l)
            storage_dict["Sl_inv_sqrt"] = jnp.zeros((N_l, N_l))
            storage_dict["lam_St"] = jnp.zeros(N_t)
            storage_dict["St_inv_sqrt"] = jnp.zeros((N_t, N_t))
            storage_dict["lam_Kl_tilde"] = jnp.zeros(N_l)
            storage_dict["W_l"] = jnp.zeros((N_l, N_l))
            storage_dict["lam_Kt_tilde"] = jnp.zeros(N_t)
            storage_dict["W_t"] = jnp.zeros((N_t, N_t))


        storage_dict["lam_Sl"], storage_dict["Sl_inv_sqrt"] = jax.lax.cond(Sl_diff,
                                                                           self.Sl_decomp_fn,
                                                                           lambda *args: (storage_dict["lam_Sl"],
                                                                                          storage_dict["Sl_inv_sqrt"]),
                                                                           Sl)
        
        storage_dict["lam_St"], storage_dict["St_inv_sqrt"] = jax.lax.cond(St_diff,
                                                                           self.St_decomp_fn,
                                                                           lambda *args: (storage_dict["lam_St"], 
                                                                                          storage_dict["St_inv_sqrt"]),
                                                                           St)

        storage_dict["lam_Kl_tilde"], storage_dict["W_l"] = jax.lax.cond(Kl_diff,
                                                                         self.Kl_tilde_decomp_fn,
                                                                         lambda *args: (storage_dict["lam_Kl_tilde"],
                                                                                        storage_dict["W_l"]),
                                                                         Kl, storage_dict["Sl_inv_sqrt"])

        storage_dict["lam_Kt_tilde"], storage_dict["W_t"] = jax.lax.cond(Kt_diff,
                                                                         self.Kt_tilde_decomp_fn,
                                                                         lambda *args: (storage_dict["lam_Kt_tilde"],
                                                                                        storage_dict["W_t"]),
                                                                         Kt, storage_dict["St_inv_sqrt"])

        D = jnp.outer(storage_dict["lam_Kl_tilde"], storage_dict["lam_Kt_tilde"]) + 1.
        storage_dict["D_inv"] = jnp.reciprocal(D)

        lam_S = jnp.outer(storage_dict["lam_Sl"], storage_dict["lam_St"])
        storage_dict["logdetK"] = jnp.log(jnp.multiply(D, lam_S)).sum()

        storage_dict["Sl"] = Sl
        storage_dict["St"] = St
        storage_dict["Kl"] = Kl
        storage_dict["Kt"] = Kt

        return storage_dict
    
    
    def logL(self, hp, x_l, x_t, R, storage_dict):
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
            storage_dict (PyTree): Stored values from the decomposition of the covariance matrices. For
                :class:`LuasKernel` this consists of values computed using the eigendecomposition
                of each matrix and also the log determinant of ``K``.
        
        Returns:
            (Scalar, PyTree): A tuple where the first element is the value of the log likelihood.
            The second element is a PyTree which contains stored values from the decomposition of the
            covariance matrix.
            
        """

        storage_dict = self.decomp_fn(hp, x_l, x_t, storage_dict = storage_dict)
        
        rKr = r_K_inv_r(R, storage_dict)
        logdetK = logdetK_calc(storage_dict)
        logL = -0.5 * rKr - 0.5 * logdetK  - 0.5 * R.size * jnp.log(2*jnp.pi)

        return  logL, storage_dict

    
    def logL_hessianable(self, hp, x_l, x_t, R, storage_dict):
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
            storage_dict (PyTree): Stored values from the decomposition of the covariance matrices. For
                :class:`LuasKernel` this consists of values computed using the eigendecomposition
                of each matrix and also the log determinant of ``K``.
                
        Returns:
            (Scalar, PyTree): A tuple where the first element is the value of the log likelihood.
            The second element is a PyTree which contains stored values from the decomposition of the
            covariance matrix.
        
        """

        storage_dict = self.decomp_fn(hp, x_l, x_t, storage_dict = storage_dict)
        
        rKr = r_K_inv_r(R, storage_dict)
        logdetK = logdetK_calc_hessianable(storage_dict)
        logL =  -0.5 * rKr - 0.5 * logdetK  - 0.5 * R.size * jnp.log(2*jnp.pi)

        return  logL, storage_dict
        
    
    def predict(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_l_pred: JAXArray,
        x_t: JAXArray,
        x_t_pred: JAXArray,
        R: JAXArray,
        M_s: JAXArray,
        storage_dict: Optional[PyTree] = {},
        wn = True,
        return_std_dev = True,
    ) -> Tuple[JAXArray, JAXArray, JAXArray]:
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
            storage_dict (PyTree): Stored values from the decomposition of the covariance matrices. For
                :class:`LuasKernel` this consists of values computed using the eigendecomposition
                of each matrix and also the log determinant of ``K``.
            wn (bool, optional): Whether to include white noise in the uncertainty at the predicted locations.
                Defaults to True.
            return_std_dev (bool, optional): If ``True`` will return the standard deviation of uncertainty at the predicted
                locations. Otherwise will return the full predictive covariance matrix. Defaults to True.
        
        Returns:
            JAXArray: The GP predictive mean at the prediction locations.
            JAXArray: The GP predictive uncertainty in standard deviations of shape ``(N_l_pred, N_t_pred)``
            if ``return_std_dev = True``, otherwise the full GP predictive covariance matrix of shape
            ``(N_l_pred*N_t_pred, N_l_pred*N_t_pred)``.
        
        """
        
        storage_dict = self.decomp_fn(hp, x_l, x_t, storage_dict = storage_dict)
        
        Kl_s = self.Kl(hp, x_l, x_l_pred, wn = False)
        Kt_s = self.Kt(hp, x_t, x_t_pred, wn = False)
        Sl_s = self.Sl(hp, x_l, x_l_pred, wn = False)
        St_s = self.St(hp, x_t, x_t_pred, wn = False)
        
        Kl_ss = self.Kl(hp, x_l_pred, x_l_pred, wn = wn)
        Kt_ss = self.Kt(hp, x_t_pred, x_t_pred, wn = wn)
        Sl_ss = self.Sl(hp, x_l_pred, x_l_pred, wn = wn)
        St_ss = self.St(hp, x_t_pred, x_t_pred, wn = wn)

        alpha = K_inv_vec(R, storage_dict)

        gp_mean = M_s + kron_prod(Kl_s.T, Kt_s.T, alpha) + kron_prod(Sl_s.T, St_s.T, alpha)

        Y_l = Kl_s.T @ storage_dict["W_l"]
        Y_t = Kt_s.T @ storage_dict["W_t"]
        Z_l = Sl_s.T @ storage_dict["W_l"]
        Z_t = St_s.T @ storage_dict["W_t"]

        if return_std_dev:
            pred_err = jnp.outer(jnp.diag(Kl_ss), jnp.diag(Kt_ss))
            pred_err += jnp.outer(jnp.diag(Sl_ss), jnp.diag(St_ss))

            pred_err -= kron_prod(Y_l**2, Y_t**2, storage_dict["D_inv"])
            pred_err -= kron_prod(Z_l**2, Z_t**2, storage_dict["D_inv"])
            pred_err -= 2*kron_prod(Y_l * Z_l, Y_t * Z_t, storage_dict["D_inv"])
            
            pred_err = jnp.sqrt(pred_err)
        else:
            N_l_pred = x_l_pred.shape[-1]
            N_t_pred = x_t_pred.shape[-1]

            KW_l = Kl_s.T @ storage_dict["W_l"]
            KW_t = Kt_s.T @ storage_dict["W_t"]
            SW_l = Sl_s.T @ storage_dict["W_l"]
            SW_t = St_s.T @ storage_dict["W_t"]

            def K_mult(K1, K2):
                return K1*K2
            vmap_K_mult = jax.vmap(K_mult, in_axes = (0, None), out_axes = 0)

            cov_wrong_order = jnp.zeros((N_l_pred**2, N_t_pred**2))
            for (Kl1, Kt1) in [(KW_l, KW_t), (SW_l, SW_t)]:
                for (Kl2, Kt2) in [(KW_l, KW_t), (SW_l, SW_t)]:

                    Kl_cube = vmap_K_mult(Kl1, Kl2)
                    Kt_cube = vmap_K_mult(Kt1, Kt2)

                    Kl_cube = Kl_cube.reshape((N_l_pred**2, N_l_pred))
                    Kt_cube = Kt_cube.reshape((N_t_pred**2, N_t_pred))

                    cov_wrong_order += (Kl_cube @ storage_dict["D_inv"] @ Kt_cube.T)

            cov_wrong_order = cov_wrong_order.reshape((N_l_pred**2*N_t_pred, N_t_pred))

            pred_err = jnp.zeros((N_l_pred*N_t_pred, N_l_pred*N_t_pred))
            for j in range(N_l_pred):
                cov_wrt_x_l_j = cov_wrong_order[j*N_l_pred*N_t_pred:(j+1)*N_l_pred*N_t_pred, :]
                pred_err = pred_err.at[:, j*N_t_pred:(j+1)*N_t_pred].set(-cov_wrt_x_l_j)

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
        
        Sl = self.Sl(hp, x_l, x_l)
        lam_Sl, Q_Sl = self.Sl.decomp(Sl)
        Sl_inv_sqrt = Q_Sl @ jnp.diag(jnp.sqrt(jnp.reciprocal(lam_Sl)))

        St = self.St(hp, x_t, x_t)
        lam_St, Q_St = self.St.decomp(St)
        St_inv_sqrt = Q_St @ jnp.diag(jnp.sqrt(jnp.reciprocal(lam_St)))

        Kl = self.Kl(hp, x_l, x_l)
        Kl_tilde = Sl_inv_sqrt.T @ Kl @ Sl_inv_sqrt
        lam_Kl_tilde, Q_Kl_tilde = self.Kl.decomp(Kl_tilde)
        
        Kt = self.Kt(hp, x_t, x_t)
        Kt_tilde = St_inv_sqrt.T @ Kt @ St_inv_sqrt
        lam_Kt_tilde, Q_Kt_tilde = self.Kt.decomp(Kt_tilde)

        lam_S_half = jnp.outer(jnp.sqrt(lam_Sl), jnp.sqrt(lam_St))
        lam_S_half = lam_S_half.reshape((N_l, N_t, 1))
        D_half = jnp.sqrt(jnp.outer(lam_Kl_tilde, lam_Kt_tilde) + 1.)
        D_half = D_half.reshape((N_l, N_t, 1))
        
        z = np.random.normal(size = (N_l, N_t, size))
        
        D_half_z = jnp.multiply(D_half, z)
        
        kron_prod_vmap = jax.vmap(kron_prod, in_axes = (None, None, 2), out_axes = 2)
        z = kron_prod_vmap(Q_Kl_tilde, Q_Kt_tilde, z)
        z = jnp.multiply(lam_S_half, z)
        R = kron_prod_vmap(Q_Sl, Q_St, z)
        
        if size == 1:
            R = R.reshape((N_l, N_t))

        return R


    def K(self, hp, x_l1, x_l2, x_t1, x_t2, **kwargs):
        r"""Generates the full covariance matrix K formed from the sum of two kronecker products:
        
        .. math::

            K = K_l \otimes K_t + S_l \otimes S_t
        
        Useful for creating a :class:`GeneralKernel` object with the same kernel function as a :class:`LuasKernel`.
        
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
        
        Kl_i = self.Kl(hp, x_l, x_l, wn = wn)[i, :]
        Kt_j = self.Kt(hp, x_t, x_t, wn = wn)[j, :]
        Sl_i = self.Sl(hp, x_l, x_l, wn = wn)[i, :]
        St_j = self.St(hp, x_t, x_t, wn = wn)[j, :]
        
        cov = jnp.outer(Kl_i, Kt_j) + jnp.outer(Sl_i, St_j)
        
        if corr:
            Kl_diag = jnp.diag(self.Kl(hp, x_l, x_l, wn = wn))
            Kt_diag = jnp.diag(self.Kt(hp, x_t, x_t, wn = wn))
            Sl_diag = jnp.diag(self.Sl(hp, x_l, x_l, wn = wn))
            St_diag = jnp.diag(self.St(hp, x_t, x_t, wn = wn))
            
            cov /= jnp.sqrt(cov[i, j]*(jnp.outer(Kl_diag, Kt_diag) + jnp.outer(Sl_diag, St_diag)))
            
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

    
def diag_eigendecomp(K):
        return jnp.diag(K), jnp.eye(K.shape[0])


def decomp_S(S: JAXArray, eigen_fn: Optional[Callable] = jnp.linalg.eigh) -> Tuple[JAXArray, JAXArray]:

    lam_S, Q_S = eigen_fn(S)
    S_inv_sqrt = Q_S @ jnp.diag(jnp.sqrt(jnp.reciprocal(lam_S)))

    return lam_S, S_inv_sqrt
        
    
def decomp_K_tilde(K: JAXArray, S_inv_sqrt: JAXArray, eigen_fn: Optional[Callable] = jnp.linalg.eigh) -> Tuple[JAXArray, JAXArray]:
        
    K_tilde = S_inv_sqrt.T @ K @ S_inv_sqrt
    lam_K_tilde, Q_K_tilde = eigen_fn(K_tilde)
    W_K_tilde = S_inv_sqrt @ Q_K_tilde

    return lam_K_tilde, W_K_tilde
