import numpy as np
from typing import Optional, Callable, Tuple, Any, Union

import jax.numpy as jnp
import jax.scipy.linalg as JLA

from luas.kernels.covtype import CovType, General
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import tensor_mult

__all__ = ["GeneralKernel"]

class GeneralKernel(CovType):
    r"""Unoptimised reference implementation which builds the full covariance matrix
    and decomposes it with a general Cholesky decomposition.

    Accepts the same kernel structure as the optimised Kronecker kernels, i.e. a
    sum of Kronecker products of covariance matrices in each dimension:

    .. math::

        \mathbf{K} = \mathbf{\Sigma}_\lambda \otimes \mathbf{\Sigma}_t
        + \sum_i \mathbf{K}_{\lambda; i} \otimes \mathbf{K}_{t; i}
        \; \left[+ \, \mathbf{K}_\mathrm{dense}\right],

    optionally plus an arbitrary dense matrix :math:`\mathbf{K}_\mathrm{dense}`.
    Any terms accepted by the optimised kernels (including low-rank terms from
    ``luas.kernels.lowrank``) can be used, for data with two or more dimensions.

    No structure is exploited, so it is useful mainly for testing: validating the
    log-likelihood of an optimised kernel, benchmarking, or kernels with no
    available optimisation. Force it with ``luas.GP(..., use_kernel =
    luas.GeneralKernel)``; :class:`luas.GP` never selects it automatically.

    **Runtime scaling.** :math:`\mathcal{O}(N^3)` runtime and
    :math:`\mathcal{O}(N^2)` memory for :math:`N = N_\lambda N_t` total data
    points.

    Warning:
        The full :math:`(N, N)` covariance matrix is built in memory, which becomes
        impractical beyond roughly :math:`N \sim 10^4` data points (a
        :math:`(10^3, 10^3)` dataset would need ~8 TB at double precision).

    Example:
        .. code-block:: python

            def kf(p, X):
                s_l = luas.kernels.Noise(jnp.exp(p["log_sigma"]))
                s_t = luas.kernels.KroneckerDelta()
                k_l = luas.kernels.Constant(jnp.exp(p["log_h"][0]))
                k_t = luas.kernels.Matern32(jnp.exp(p["log_l_t"][0]))
                return (s_l, s_t), (k_l, k_t)

            gp_fast = luas.GP(p, X, kf = kf)
            gp_ref = luas.GP(p, X, kf = kf, use_kernel = luas.GeneralKernel)

            # Should agree to numerical precision
            gp_fast.logL(p, Y), gp_ref.logL(p, Y)

    Args:
        Sigma (tuple[CovType] or CovType): Either a tuple containing one covariance
            object per dimension for the first Kronecker product term, or (when the
            kernel function returns a single tuple) the first dimension's covariance
            object, with the remaining dimensions given in ``*K_list``.
        *K_list (tuple[CovType]): Any further Kronecker product terms, each a tuple
            with one covariance object per dimension. If ``Sigma`` is a single
            covariance object, these are instead the covariance objects for the
            remaining dimensions of a single Kronecker product.
        dense_mat (JAXArray, optional): An optional :math:`(N, N)` matrix added to
            the covariance matrix after the Kronecker terms are built. Rows and
            columns follow the row-major (``"C"``) flattening of the data array.
        use_stored_values (bool, optional): Currently unused.

    See Also:
        :class:`luas.SeparableKernel`, :class:`luas.LuasKernel`,
        :class:`luas.ExtendedLuasKernel` and the other Kronecker kernels for exact
        optimised alternatives.
    """
    def __init__(
        self,
        Sigma,
        *K_list,
        dense_mat = None,
        use_stored_values = False,
    ):
        # Function used to build the covariance matrix K
        if isinstance(Sigma, CovType) and K_list != ():
            # Case of a single Kronecker term (optimised version is SeparableKernel)
            self.Sigma = (Sigma,) + K_list
            self.K_list = ()
        elif isinstance(Sigma, CovType) and K_list == ():
            # Case of a one dimensional GP, implemented with General as this term expects a tuple for regression variables
            raise Exception("GeneralKernel is a Kronecker kernel and cannot be run as a 1D GP")
        else:
            self.Sigma = Sigma
            self.K_list = K_list

        self.opt_name = "GeneralKernel"
        self.dense_mat = dense_mat
        
        self.dim = len(self.Sigma)

        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            raise NotImplementedError("Use of pre-calculated stored values not yet implemented.")
        else:
            self.decompose = self.decompose_no_stored_values
    
    
    def evaluate(self, X1, X2, full = True, idx = None, **kwargs):

        if idx is None:
            idx = (None,)*self.dim

        Sigma = self.Sigma[0].evaluate(X1[0], X2[0], full = full, row_idx = idx[0], col_idx = idx[0], **kwargs)
        
        for d in range(1, self.dim):
            Sigma = jnp.kron(Sigma, self.Sigma[d].evaluate(X1[d], X2[d], full = full, row_idx = idx[d], col_idx = idx[d], **kwargs))
        
        for i in range(len(self.K_list)):
            K = self.K_list[i][0].evaluate(X1[0], X2[0], full = full, row_idx = idx[0], col_idx = idx[0], **kwargs)
            for d in range(1, self.dim):
                K = jnp.kron(K, self.K_list[i][d].evaluate(X1[d], X2[d], row_idx = idx[d], col_idx = idx[d], full = full, **kwargs))
            
            Sigma += K

        if self.dense_mat is not None:
            Sigma += self.dense_mat
        
        return Sigma
    
    def decompose_no_stored_values(
        self,
        X,
        stored_values: Optional[PyTree] = {},
        **kwargs,
    ) -> PyTree:
        
        # Simply builds the covariance matrix and decomposes it into a Cholesky factor L
        # and precomputes the log determinant of K for log likelihood calculations
        K = self.evaluate(X, X, **kwargs)
        self.factor = JLA.cholesky(K, lower = True)
        self.logdet = 2*jnp.log(jnp.diag(self.factor)).sum()
        
        return self, {"logdet":self.logdet}

    def matrix_inv_sqrt(self, R, transpose = 0):
        
        R_shape = R.shape
        r = R.ravel("C")
        r_prime = JLA.solve_triangular(self.factor, r, lower = True, trans = transpose)
        return r_prime.reshape(R_shape)

    def matrix_sqrt(self, R, transpose = 0):
        
        R_shape = R.shape
        r = R.ravel("C")

        if transpose:
            r_prime = self.factor.T @ r
        else:
            r_prime = self.factor @ r
        return r_prime.reshape(R_shape)
    
    def dot_solve(self, R):
        
        r = R.ravel("C")
        r_prime = JLA.solve_triangular(self.factor, r, lower = True, trans = 0)
        return jnp.square(r_prime).sum()

    def matmul(
        self,
        X1,
        X2,
        R: JAXArray,
        **kwargs,
    ) -> JAXArray:
        r"""Calculates the product of the covariance matrix with a vector, represented by a JAXArray of shape ``(N_l, N_t)``.
        Useful for testing for numerical stability.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrix ``K``. Will be
                unaffected if additional mean function parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            R (JAXArray): JAXArray of shape ``(N_l, N_t)`` representing the vector to multiply on the right by
                the covariance matrix ``K``.
                
        Returns:
            JAXArray: The result of multiplying the covariance matrix ``K`` by the vector ``R``.
        
        """

        R_prime = tensor_mult(self.Sigma, X1, X2, R, **kwargs)

        for i in range(len(self.K_list)):
            R_prime += tensor_mult(self.K_list[i], X1, X2, R, **kwargs)
        
        return R_prime
    
    # def predict(
    #     self,
    #     hp: PyTree,
    #     x_l: JAXArray,
    #     x_l_pred: JAXArray,
    #     x_t: JAXArray,
    #     x_t_pred: JAXArray,
    #     R: JAXArray,
    #     M_s: JAXArray,
    #     wn: Optional[bool] = True,
    #     return_std_dev: Optional[bool] = True,
    # ) -> Tuple[JAXArray, JAXArray, JAXArray]:
    #     r"""Performs GP regression and computes the GP predictive mean and the GP predictive
    #     uncertainty as the standard devation at each location or else can return the full
    #     covariance matrix. Requires the input kernel function ``K`` to have a ``wn`` keyword
    #     argument that defines the kernel when white noise is included (``wn = True``) and
    #     when white noise isn't included (``wn = False``).
        
    #     Currently assumes the same input hyperparameters for both the observed and predicted
    #     locations. The predicted locations ``x_l_pred`` and ``x_t_pred`` may deviate from
    #     the observed locations ``x_l`` and ``x_t`` however.
        
    #     The GP predictive mean is defined as:
        
    #     .. math::

    #         \mathbb{E}[\vec{y}_*] = \vec{\mu}_* + \mathbf{K}_*^T \mathbf{K}^{-1} \vec{r}
            
    #     And the GP predictive covariance is given by:
        
    #     .. math::
            
    #         Var[\vec{y}_*] = \mathbf{K}_{**} - \mathbf{K}_*^T \mathbf{K}^{-1} \mathbf{K}_*
        
    #     Args:
    #         hp (Pytree): Hyperparameters needed to build the covariance matrix ``K``. Will be
    #             unaffected if additional mean function parameters are also included.
    #         x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
    #             for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
    #             different wavelength/vertical regression variables.
    #         x_l_pred (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
    #             for the prediction locations (which may be the same as the observed locations).
    #             May be of shape ``(N_l_pred,)`` or ``(d_l,N_l_pred)`` for ``d_l`` different
    #             wavelength/vertical regression variables.
    #         x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
    #             observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
    #             time/horizontal regression variables.
    #         x_t_pred (JAXArray): Array containing time/horizontal dimension regression variable(s) for
    #             the prediction locations (which may be the same as the observed locations). May be of
    #             shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different time/horizontal regression variables.
    #         R (JAXArray): Residuals to be fit, equal to the observed data minus the deterministic
    #             mean function. Must have the same shape as the observed data ``(N_l, N_t)``.
    #         M_s (JAXArray): Mean function evaluated at the locations of the predictions ``x_l_pred``, ``x_t_pred``.
    #             Must have shape ``(N_l_pred, N_t_pred)`` where ``N_l_pred`` is the number of wavelength/vertical
    #             dimension predictions and ``N_t_pred`` the number of time/horizontal dimension predictions.
    #         wn (bool, optional): Whether to include white noise in the uncertainty at the predicted locations.
    #             Defaults to True.
    #         return_std_dev (bool, optional): If ``True`` will return the standard deviation of uncertainty at the predicted
    #             locations. Otherwise will return the full predictive covariance matrix. Defaults to True.
        
    #     Returns:
    #         (JAXArray, JAXArray): Returns a tuple of two elements, where the first element is
    #         the GP predictive mean at the prediction locations, the second element is either the
    #         standard deviation of the predictions if ``return_std_dev = True``, otherwise it will be
    #         the full covariance matrix of the predicted values.
        
    #     """
    #     # Get the length of each prediction dimension
    #     N_l_pred = x_l_pred.shape[-1]
    #     N_t_pred = x_t_pred.shape[-1]
        
    #     # Calculate the decomposition of K for the required K^-1 calculations
    #     stored_values = self.decomp_fn(hp, x_l, x_t)
            
    #     # Calculate the covariance between the observed and predicted points
    #     K_s = self.kf(hp, x_l, x_l_pred, x_t, x_t_pred, wn = False)
        
    #     # Calculate the covariance between predicted points with other predicted points
    #     K_ss = self.kf(hp, x_l_pred, x_l_pred, x_t_pred, x_t_pred, wn = wn)

    #     # Flatten residuals vector
    #     r = R.ravel("C")
        
    #     # Use forward and backward substitution to solve K^-1 r using the Cholesky factor
    #     alpha = JLA.solve_triangular(stored_values["L_cho"], r, trans = 1)
    #     K_inv_R = JLA.solve_triangular(stored_values["L_cho"], alpha, trans = 0)
        
    #     # Computes the GP predictive mean
    #     gp_mean = K_s.T @ K_inv_R
    #     gp_mean = M_s + gp_mean.reshape(N_l_pred, N_t_pred)
        
    #     # Prepare to calculate K^-1 K_s in the predictive covariance calculation
    #     L_inv_K_s = JLA.solve_triangular(stored_values["L_cho"], K_s, trans = 1)
        
    #     if return_std_dev:
    #         # Get diagonal of covariance of predicted locations with other predicted locations
    #         pred_err = jnp.diag(K_ss)
            
    #         # Subtract off term related to covariance between predicted and observed locations
    #         # This method efficiently calculates only the diagonal of the term
    #         pred_err -= (L_inv_K_s**2).sum(0)
            
    #         # Convert shape to (N_l_pred, N_t_pred) to match observed data but at predicted locations
    #         pred_err = pred_err.reshape(N_l_pred, N_t_pred)
            
    #         # Convert from variance to std dev
    #         pred_err = jnp.sqrt(pred_err)
    #     else:
    #         # Directly compute the predictive covariance matrix
    #         pred_err = K_ss - L_inv_K_s.T @ L_inv_K_s
        
    #     return gp_mean, pred_err
    
