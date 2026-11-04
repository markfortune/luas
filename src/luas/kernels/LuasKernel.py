import numpy as np
import jax.numpy as jnp
import jax
from copy import deepcopy
from typing import Callable, Tuple, Union, Any, Optional

from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import kron_prod
from luas.stable_gradients import logdetK_calc, r_K_inv_r, K_inv_vec, logdetK_calc_hessianable
from luas.kernels.covtype import Identity, ScaledIdentity, Diagonal, CovType

__all__ = [
    "diag_eigendecomp",
    "decomp_S",
    "decomp_K_tilde",
    "LuasKernel",
]

def diag_eigendecomp(K: JAXArray) -> Tuple[JAXArray, JAXArray]:
    r"""Calculates the eigenvalues and eigenvectors of a matrix ``K`` assuming it is diagonal.
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
    r"""Calculates the eigenvalues and matrix inverse square root of a matrix ``S``. Needs to be performed
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


class LuasKernel(CovType):
    r"""GP optimisation for a covariance matrix that is the sum of two Kronecker
    products, using simultaneous diagonalisation in both dimensions.

    Handles kernel functions of the form

    .. math::

        k(\lambda, \lambda', t, t') = s_\lambda(\lambda, \lambda')\, s_t(t, t')
        + k_\lambda(\lambda, \lambda')\, k_t(t, t'),

    which on a rectangular grid gives the covariance matrix

    .. math::

        \mathbf{K} = \mathbf{\Sigma}_\lambda \otimes \mathbf{\Sigma}_t
        + \mathbf{K}_\lambda \otimes \mathbf{K}_t.

    Dimension 0 is written as wavelength :math:`\lambda` and dimension 1 as time
    :math:`t`, but any 2D rectangular grid can be used. The first term typically
    models white noise and measurement covariance, and the second term a 2D
    correlated noise term (e.g. systematics correlated in both wavelength and time).
    Any valid kernels can be used for all four matrices.

    **Method.** From Rakitsch et al. (2013). In each dimension the pair of matrices
    is simultaneously diagonalised: :math:`\mathbf{K}_\lambda` is transformed by the
    inverse square root of :math:`\mathbf{\Sigma}_\lambda` and the result
    eigendecomposed (and similarly in time), which factorises :math:`\mathbf{K}` as

    .. math::

        \mathbf{K}^{-1} = [\mathbf{W}_\lambda \otimes \mathbf{W}_t]\,
        [\mathbf{I} + \mathbf{\Lambda}_{\tilde{K}_\lambda} \otimes
        \mathbf{\Lambda}_{\tilde{K}_t}]^{-1}\,
        [\mathbf{W}_\lambda^T \otimes \mathbf{W}_t^T],

    so the log-likelihood only needs a diagonal solve after transforming the
    residuals. This implementation computes the inverse square roots of
    :math:`\mathbf{\Sigma}_\lambda` and :math:`\mathbf{\Sigma}_t` by
    eigendecomposition, so four eigendecompositions are performed in total
    (Fortune, Gibson & Pope, in prep. describe using cheaper Cholesky
    decompositions for this step instead).

    **Runtime scaling.** :math:`\mathcal{O}(N_\lambda^3 + N_t^3 + N_\lambda N_t
    (N_\lambda + N_t))` runtime and :math:`\mathcal{O}(N_\lambda^2 + N_t^2 +
    N_\lambda N_t)` memory. The two eigendecompositions dominate, which in practice
    limits this kernel to roughly 1000-4000 points in either dimension. Using
    quasiseparable kernels (``luas.kernels.quasisep``) does not reduce this cost.

    **Selection.** :class:`luas.GP` selects this optimisation automatically when the
    kernel function returns two Kronecker terms, ``(s_l, s_t), (k_l, k_t)``, with no
    low-rank terms, and the kernels along the longer dimension are not all
    quasiseparable/diagonal (so :class:`luas.LuasLasrachKernel` would not scale
    linearly along it). The one exception is when *neither* dimension is
    quasiseparable/diagonal and the shorter dimension has fewer than
    ``max_gen_cholesky_blocks`` points (default 10, set through
    ``kernel_select_kwargs``), in which case :class:`luas.LuasLasrachKernel` is used
    instead. Force it with ``luas.GP(..., use_kernel = luas.LuasKernel)``.

    Warning:
        :math:`\mathbf{\Sigma}_\lambda` and :math:`\mathbf{\Sigma}_t` must be positive
        definite and well-conditioned, and must be the **first** tuple returned by
        the kernel function. :math:`\mathbf{K}_\lambda` and :math:`\mathbf{K}_t` may
        be positive semi-definite.

    Note:
        This is a legacy implementation. It evaluates the four matrices densely
        and eigendecomposes them, rather than using each covariance object's own
        decomposition, and is restricted to exactly two dimensions and a single
        :math:`\mathbf{K}` term. See
        :class:`luas.LuasKernelND` for more than two dimensions, and
        :class:`luas.LuasPlusMultiLowRankKernel` or :class:`luas.ExtendedLuasKernel`
        for adding further low-rank terms.

    Example:
        Measurement covariance plus a 2D Matern kernel term (from the ``luas``
        tutorial):

        .. code-block:: python

            def kf(p, X):
                s_l = luas.kernels.Noise(jnp.exp(p["log_sigma"]))
                s_l += jnp.exp(2*p["log_h_readout"][0]) * luas.kernels.Exp(jnp.exp(p["log_l_readout"][0]))
                s_t = luas.kernels.KroneckerDelta()

                k_l = jnp.exp(2*p["log_h"][0]) * luas.kernels.Matern52(jnp.exp(p["log_l_l"][0]))
                k_t = luas.kernels.Matern32(jnp.exp(p["log_l_t"][0]))

                # The positive definite (s_l, s_t) term must come first
                return (s_l, s_t), (k_l, k_t)

            gp = luas.GP(p, (x_l, x_t), kf = kf, use_kernel = luas.LuasKernel)

    Args:
        Sigma (tuple[CovType, CovType]): The covariance objects
            :math:`(\mathbf{\Sigma}_\lambda, \mathbf{\Sigma}_t)` for dimensions 0 and
            1. Both must be positive definite.
        K (tuple[CovType, CovType]): The covariance objects
            :math:`(\mathbf{K}_\lambda, \mathbf{K}_t)` for dimensions 0 and 1.
        use_stored_values (bool, optional): If ``True`` (the default), each call to
            ``decompose`` checks whether any of the four component matrices are
            unchanged (within ``rtol``/``atol``) from the ``stored_values`` passed in
            and reuses their stored eigendecompositions. This can save time when some
            hyperparameters are held fixed, e.g. blocked Gibbs sampling. If ``False``,
            all four matrices are decomposed on every call.
        **kwargs: Ignored, accepted so that all Kronecker kernels share a common
            constructor interface.

    See Also:
        :class:`luas.LuasLasrachKernel` for the same kernel structure with linear
        scaling in one dimension.
        :class:`luas.SeparableKernel` if only the first Kronecker term is needed.
    """
    
    def __init__(
        self,
        Sigma,
        K,
        *, # force keyword only after this
        use_stored_values: Optional[bool] = True,
        **kwargs,
    ):
        r"""Creates the kernel object, see the class docstring for the arguments."""
        
        self.Kl = lambda x1, x2, full = True, **kwargs: K[0].evaluate(x1, x2, full = full, **kwargs)
        self.Kt = lambda x1, x2, full = True, **kwargs: K[1].evaluate(x1, x2, full = full, **kwargs)
        self.Sl = lambda x1, x2, full = True, **kwargs: Sigma[0].evaluate(x1, x2, full = full, **kwargs)
        self.St = lambda x1, x2, full = True, **kwargs: Sigma[1].evaluate(x1, x2, full = full, **kwargs)

        self.Sigma = Sigma
        self.K_list = [K]
        self.opt_name = "LuasKernel"

        for i in range(2):
            S_i = [self.Sl, self.St][i]
            K_i = [self.Kl, self.Kt][i]
            if isinstance(Sigma[i], (Identity, ScaledIdentity, Diagonal)):
                S_i.decomp = diag_eigendecomp

                if isinstance(K[i], (Identity, ScaledIdentity, Diagonal)):
                    K_i.decomp = diag_eigendecomp
                else:
                    K_i.decomp = jnp.linalg.eigh
            else:
                S_i.decomp = jnp.linalg.eigh
                K_i.decomp = jnp.linalg.eigh

        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            self.decompose = self.decompose_use_stored_values
        else:
            self.decompose = self.decompose_no_stored_values

        # Specify how each of the 4 matrices will be decomposed
        self.Sl_decomp_fn = lambda Sl: decomp_S(Sl, eigen_fn = self.Sl.decomp)
        self.St_decomp_fn = lambda St: decomp_S(St, eigen_fn = self.St.decomp)
        self.Kl_tilde_decomp_fn = lambda Kl, Sl_inv_sqrt: decomp_K_tilde(Kl, Sl_inv_sqrt, eigen_fn = self.Kl.decomp)
        self.Kt_tilde_decomp_fn = lambda Kt, St_inv_sqrt: decomp_K_tilde(Kt, St_inv_sqrt, eigen_fn = self.Kt.decomp)

    
    def decompose_no_stored_values(
        self,
        X: tuple,
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:
        r"""Required calculations for the decomposition of the overall matrix ``K`` where the previously
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

        x_l, x_t = X[0], X[1]
        
        # Calculate each of the four component matrices and decompose them into the required matrices for log likelihood calculations
        stored_values["Sl"] = self.Sl(x_l, x_l)
        stored_values["lam_Sl"], stored_values["Sl_inv_sqrt"] = self.Sl_decomp_fn(stored_values["Sl"])

        stored_values["St"] = self.St(x_t, x_t)
        stored_values["lam_St"], stored_values["St_inv_sqrt"] = self.St_decomp_fn(stored_values["St"])

        # See decomp_K_tilde for how W_l and the eigenvalues of Kl_tilde are calculated
        stored_values["Kl"] = self.Kl(x_l, x_l)
        stored_values["lam_Kl_tilde"], stored_values["W_l"] = self.Kl_tilde_decomp_fn(stored_values["Kl"], stored_values["Sl_inv_sqrt"])
        
        stored_values["Kt"] = self.Kt(x_t, x_t)
        stored_values["lam_Kt_tilde"], stored_values["W_t"] = self.Kt_tilde_decomp_fn(stored_values["Kt"], stored_values["St_inv_sqrt"])

        # D is needed for calculation the log determinant of K
        D = jnp.outer(stored_values["lam_Kl_tilde"], stored_values["lam_Kt_tilde"]) + 1.
        
        # D^-1 is needed for calculating K^-1 r
        stored_values["D_inv"] = jnp.reciprocal(D)

        # Computes the log determinant of K
        lam_S = jnp.outer(stored_values["lam_Sl"], stored_values["lam_St"])
        stored_values["logdet"] = jnp.log(jnp.multiply(D, lam_S)).sum()

        self.stored_values = stored_values
        
        return self, stored_values

    
    def decompose_use_stored_values(
        self,
        X: tuple,
        stored_values: Optional[PyTree] = {},
        rtol: Optional[Scalar] = 1e-12,
        atol: Optional[Scalar] = 1e-12,
    ) -> PyTree:
        r"""Required calculations for the decomposition of the overall matrix ``K`` where the previously
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
        x_l, x_t = X[0], X[1]
        
        # Calculate each of the four component matrices
        Sl = self.Sl(x_l, x_l)
        St = self.St(x_t, x_t)
        Kl = self.Kl(x_l, x_l)
        Kt = self.Kt(x_t, x_t)
        
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
        stored_values["logdet"] = jnp.log(jnp.multiply(D, lam_S)).sum()

        # Store in order to perform checks for the next call of this function
        stored_values["Sl"] = Sl
        stored_values["St"] = St
        stored_values["Kl"] = Kl
        stored_values["Kt"] = Kt

        self.stored_values = stored_values

        return self, stored_values
    
    
    def logL(
        self,
        R: JAXArray,
        stored_values = {},
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
        # stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = stored_values)
        
        # Use functions with custom derivatives to accurately calculate the log
        # likelihood and its gradient
        rKr = r_K_inv_r(R, stored_values)
        logdet = logdetK_calc(stored_values)
        logL = - 0.5 * logdet - 0.5 * R.size * jnp.log(2*jnp.pi) -0.5 * rKr 

        return  logL

    
    def logL_hessianable(
        self,
        R: JAXArray,
        stored_values = {},
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
        
        # Use functions with custom derivatives to accurately calculate the log
        # likelihood, its gradient and hessian
        rKr = r_K_inv_r(R, stored_values)
        logdet = logdetK_calc_hessianable(stored_values)
        logL =  -0.5 * rKr - 0.5 * logdet  - 0.5 * R.size * jnp.log(2*jnp.pi)

        return  logL

    def matrix_inv_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
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
            wn (bool, optional): Whether to include white noise when generating noise. Must have
                a `wn` keyword argument in all kernel functions ``Kl``, ``Kt``, ``Sl``, ``St``.
                
        Returns:
            JAXArray: If ``size = 1`` will generate noise of shape ``(N_l, N_t)``, otherwise if ``size > 1`` then
            generated noise will be of shape ``(N_l, N_t, size)``.
        
        """
        D_inv_sqrt = jnp.sqrt(self.stored_values["D_inv"])
        
        if transpose:
            R_prime = jnp.multiply(D_inv_sqrt, R)
            R_prime = kron_prod(self.stored_values["W_l"], self.stored_values["W_t"], R_prime)
        else:
            R_prime = kron_prod(self.stored_values["W_l"].T, self.stored_values["W_t"].T, R)
            R_prime = jnp.multiply(D_inv_sqrt, R_prime)

        return R_prime


    def matrix_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
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
            wn (bool, optional): Whether to include white noise when generating noise. Must have
                a `wn` keyword argument in all kernel functions ``Kl``, ``Kt``, ``Sl``, ``St``.
                
        Returns:
            JAXArray: If ``size = 1`` will generate noise of shape ``(N_l, N_t)``, otherwise if ``size > 1`` then
            generated noise will be of shape ``(N_l, N_t, size)``.
        
        """

        # Computes the sqrt of the diagonal matrix D
        D_half = jnp.sqrt(jnp.outer(self.stored_values["lam_Kl_tilde"], self.stored_values["lam_Kt_tilde"]) + 1.)

        if transpose:
            R_prime = kron_prod(self.stored_values["Sl"], self.stored_values["St"], R)
            R_prime = kron_prod(self.stored_values["W_l"].T, self.stored_values["W_t"].T, R_prime)
            R_prime = jnp.multiply(D_half, R_prime)
        else:
            R_prime = jnp.multiply(D_half, R)
            R_prime = kron_prod(self.stored_values["W_l"], self.stored_values["W_t"], R_prime)
            R_prime = kron_prod(self.stored_values["Sl"], self.stored_values["St"], R_prime)

        return R_prime


    def evaluate(
        self,
        X1,
        X2,
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
        x_l1, x_t1 = X1[0], X1[1]
        x_l2, x_t2 = X2[0], X2[1]

        # Build 4 component matrices
        Kl = self.Kl(x_l1, x_l2, **kwargs)
        Kt = self.Kt(x_t1, x_t2, **kwargs)
        Sl = self.Sl(x_l1, x_l2, **kwargs)
        St = self.St(x_t1, x_t2, **kwargs)
        
        K = jnp.kron(Kl, Kt) + jnp.kron(Sl, St)
        
        return K
    
    
    
    def inverse(
        self,
        R: JAXArray,
    ) -> JAXArray:
        r"""Calculates the product of the inverse of the covariance matrix with a vector, represented by
        a JAXArray of shape ``(N_l, N_t)``. Useful for testing for numerical stability.
        
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
            R (JAXArray): JAXArray of shape ``(N_l, N_t)`` representing the vector to multiply on the right by
                the inverse of the covariance matrix ``K``.
                
        Returns:
            JAXArray: The result of multiplying the inverse of the covariance matrix ``K`` by the vector ``R``.
        
        """
        
        return K_inv_vec(R, self.stored_values)
    
    
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
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
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
        x_l1, x_t1 = X1[0], X1[1]
        x_l2, x_t2 = X2[0], X2[1]
        hp = {}

        Sl = self.Sl(x_l1, x_l2, **kwargs)
        St = self.St(x_t1, x_t2, **kwargs)
        Kl = self.Kl(x_l1, x_l2, **kwargs)
        Kt = self.Kt(x_t1, x_t2, **kwargs)
        
        return kron_prod(Kl, Kt, R) + kron_prod(Sl, St, R)


    # def predict(
    #     self,
    #     hp: PyTree,
    #     x_l: JAXArray,
    #     x_l_pred: JAXArray,
    #     x_t: JAXArray,
    #     x_t_pred: JAXArray,
    #     R: JAXArray,
    #     M_s: JAXArray,
    #     wn = True,
    #     return_std_dev = True,
    # ) -> Tuple[JAXArray, JAXArray]:
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
        
    #     Note:
    #         The calculation of the full predictive covariance matrix when ``return_std_dev = False``
    #         is still experimental and may come with numerically stability issues. It is also very
    #         memory intensive and may cause code to crash. Future updates to luas may improve this.
        
    #     Args:
    #         hp (Pytree): Hyperparameters needed to build the covariance matrices
    #             ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
    #             parameters are also included.
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
    #     # Calculate the decomposition of K
    #     stored_values = self.decomp_fn(hp, x_l, x_t)
        
    #     # Calculate the covariance between the observed and predicted points
    #     Kl_s = self.Kl(hp, x_l, x_l_pred, wn = False)
    #     Kt_s = self.Kt(hp, x_t, x_t_pred, wn = False)
    #     Sl_s = self.Sl(hp, x_l, x_l_pred, wn = False)
    #     St_s = self.St(hp, x_t, x_t_pred, wn = False)
        
    #     # Calculate the covariance between predicted points with other predicted points
    #     Kl_ss = self.Kl(hp, x_l_pred, x_l_pred, wn = wn)
    #     Kt_ss = self.Kt(hp, x_t_pred, x_t_pred, wn = wn)
    #     Sl_ss = self.Sl(hp, x_l_pred, x_l_pred, wn = wn)
    #     St_ss = self.St(hp, x_t_pred, x_t_pred, wn = wn)

    #     # Calculate K^-1 R
    #     K_inv_R = K_inv_vec(R, stored_values)

    #     # Calculates the GP mean including the deterministic mean function at the prediction locations
    #     gp_mean = M_s + kron_prod(Kl_s.T, Kt_s.T, K_inv_R) + kron_prod(Sl_s.T, St_s.T, K_inv_R)

    #     # Prepare matrices for calculating the predictive covariance
    #     KW_l = Kl_s.T @ stored_values["W_l"]
    #     KW_t = Kt_s.T @ stored_values["W_t"]
    #     SW_l = Sl_s.T @ stored_values["W_l"]
    #     SW_t = St_s.T @ stored_values["W_t"]

    #     if return_std_dev:
    #         # Efficiently solves for the diagonal of the predictive covariance
    #         pred_err = jnp.outer(jnp.diag(Kl_ss), jnp.diag(Kt_ss))
    #         pred_err += jnp.outer(jnp.diag(Sl_ss), jnp.diag(St_ss))

    #         # K_s.T K^-1 K_s term can be broken into these three terms
    #         pred_err -= kron_prod(KW_l**2, KW_t**2, stored_values["D_inv"])
    #         pred_err -= kron_prod(SW_l**2, SW_t**2, stored_values["D_inv"])
    #         pred_err -= 2*kron_prod(KW_l * SW_l, KW_t * SW_t, stored_values["D_inv"])
            
    #         # Take the sqrt of the diagonal to get the std dev
    #         pred_err = jnp.sqrt(pred_err)
            
    #     else:
    #         # Note very memory intensive!
    #         K_W = jnp.kron(KW_l, KW_t) + jnp.kron(SW_l, SW_t)
    #         pred_err = -K_W @ jnp.diag(stored_values["D_inv"].ravel()) @ K_W.T
            
    #         # Add the K_ss term
    #         pred_err += jnp.kron(Kl_ss, Kt_ss) + jnp.kron(Sl_ss, St_ss)
        
    #     return gp_mean, pred_err
    