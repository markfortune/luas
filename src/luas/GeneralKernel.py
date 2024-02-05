import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import Optional, Callable, Tuple, Any, Union

import jax
from jax import grad, value_and_grad, hessian, vmap
import jax.numpy as jnp
import jax.scipy.linalg as JLA
from jax.flatten_util import ravel_pytree

from .luas_types import Kernel, PyTree, JAXArray, Scalar
from .kronecker_fns import make_vec, make_mat
from .jax_convenience_fns import array_to_pytree_2D

__all__ = ["GeneralKernel", "general_cholesky"]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)

    
def general_cholesky(K: JAXArray) -> Tuple[JAXArray, JAXArray]:
    r"""Takes an arbitrary covariance matrix ``K`` and returns the Cholesky decomposition
    as a lower triangular matrix L as well as computing the log determinant of the covariance
    matrix.
    
    The Cholesky factor is related to the original matrix by:
    
    .. math::
    
        K = L L^T
    
    Args:
        K (JAXArray): Covariance matrix to decompose.
        
    Returns:
        (JAXArray, Scalar): The tuple where the first element is the Cholesky factor L of ``K``
        and the second element is the log determinant of ``K``.
            
    """

    L_cho = JLA.cholesky(K)
    logdetK = 2*jnp.log(jnp.diag(L_cho)).sum()

    return L_cho, logdetK


class GeneralKernel(Kernel):
    """Kernel object which solves for the log likelihood for any general kernel function ``K``.
    Can also generate noise from ``K`` and can be used to compute the GP predictive mean and 
    predictive covariance matrix conditioned on observed data.
    
    Note:
        This method scales poorly in runtime and memory and is likely only appropriate for 
        small data sets. If the covariance matrix ``K`` possesses structure which can be exploited
        for matrix decomposition then specifying a ``decomp_fn`` which can more efficiently return
        a Cholesky factor and log determinant of ``K`` could lead to significant runtime savings.
        The :class:`LuasKernel` class should provide significant runtime savings if the covariance matrix
        has kronecker product structure in each dimension except in cases where one of the dimensions
        is very small or a sum of more than two kronecker products is needed.
        
    .. code-block:: python

        >>> from luas import GeneralKernel, kernels
        >>> def K_fn(hp, x_l1, x_l2, x_t1, x_t2, wn = True):
        >>> ... Kl = hp["h"]**2*kernels.squared_exp(x_l1, x_l2, hp["l_l"])
        >>> ... Kt = kernels.squared_exp(x_l1, x_l2, hp["l_t"])
        >>> ... K = jnp.kron(Kl, Kt)
        >>> ... return K
        >>> kernel = GeneralKernel(K = K_fn)
        ... )
    
    Args:
        K (Callable, optional): Function which returns the covariance matrix ``K``.
        decomp_fn (Callable, optional): Function which given the covariance matrix ``K``
            computes the Cholesky decomposition and log determinant of ``K``.
            Defaults to ``luas.GeneralKernel.general_cholesky`` which performs Cholesky decomposition for
            any general covariance matrix.
            
    """
    
    def __init__(
        self,
        K: Optional[Callable] = None,
        decomp_fn = None,
    ):
        # Function used to build the covariance matrix K
        self.K = K
        
        if decomp_fn is None:
            # Defaults to general Cholesky factorisation for finding the Cholesky factor
            # and the log determinant of K
            self.K_decomp_fn = general_cholesky
        else:
            # Alternatively may specify a function which given the constructed covariance matrix K
            # will return the Cholesky factor and log determinant of K (see general_cholesky for an example)
            self.K_decomp_fn = decomp_fn
            
        # alias to maintain consistency with LuasKernel which has a separate fn for calculating the hessian
        self.logL_hessianable = self.logL

    
        
    def decomp_fn(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray, 
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:
        """Builds the full covariance matrix K and uses the decomposition function specified at
        initialisation to return the Cholesky factor and the log determinant of K.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrix ``K``. Will be
                unaffected if additional mean function parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s).
                May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l`` different wavelength/vertical
                regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s).
                May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different time/horizontal
                regression variables.
            stored_values (PyTree): Stored values from the decomposition of the covariance matrix. For
                :class:`GeneralKernel` this consists of the Cholesky factor and the log determinant
                of ``K``.
                
        Returns:
            PyTree: Stored values from the decomposition of the covariance matrix consisting of the
            Cholesky factor and the log determinant of ``K``.
        
        """
        # Simply builds the covariance matrix and decomposes it into a Cholesky factor L
        # and precomputes the log determinant of K for log likelihood calculations
        K = self.K(hp, x_l, x_l, x_t, x_t)
        stored_values["L_cho"], stored_values["logdetK"] = self.K_decomp_fn(K)
        
        return stored_values

        
    def logL(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        R: JAXArray,
        stored_values: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        """Computes the log likelihood using Cholesky factorisation and also returns stored values
        from the matrix decomposition.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrix ``K``. Will be
                unaffected if additional mean function parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s).
                May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l`` different wavelength/vertical
                regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s).
                May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different time/horizontal
                regression variables.
            R (JAXArray): Residuals to be fit calculated from the observed data by subtracting the deterministic
                mean function. Must have the same shape as the observed data (N_l, N_t).
            stored_values (PyTree): Stored values from the decomposition of the covariance matrix. For
                :class:`GeneralKernel` this consists of the Cholesky factor and the log determinant
                of ``K``.
                
        Returns:
            (Scalar, PyTree): A tuple where the first element is the value of the log likelihood.
            The second element is a PyTree which contains stored values from the decomposition of the
            covariance matrix.
        """
        
        # Calculate the Cholesky factor and log determinant of K, stored in stored_values
        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = stored_values)
            
        # Calculation requires r to be a vector of shape (N_l*N_t,)
        # As opposed to (N_l, N_t) which is used for LuasKernel calculations
        r = R.ravel("C")
        
        # Solves for L^-1 r
        alpha = JLA.solve_triangular(stored_values["L_cho"], r, trans = 1)

        # Calculates the log likelihood
        logL = - 0.5 *  jnp.sum(jnp.square(alpha)) - 0.5 * stored_values["logdetK"] - (r.size/2.) * jnp.log(2*jnp.pi)

        return logL, stored_values
    

    def generate_noise(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        size: Optional[int] = 1,
        stored_values: Optional[PyTree] = {},
    ) -> JAXArray:
        """Generate noise with the covariance matrix returned by this kernel using the input
        hyperparameters ``hp``.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrix ``K``. Will be
                unaffected if additional mean function parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s).
                May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l`` different wavelength/vertical
                regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s).
                May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different time/horizontal
                regression variables.
            size (int, optional): The number of different draws of noise to generate. Defaults to 1.
            stored_values (PyTree): Stored values from the decomposition of the covariance matrix. For
                :class:`GeneralKernel` this consists of the Cholesky factor and the log determinant
                of ``K``.
                
        Returns:
            JAXArray: Generate noise of shape ``(N_l, N_t)`` if ``size = 1`` or ``(N_l, N_t, size)``
            if size > 1.
        
        """
        
        # Calculates the Cholesky factorisation of K for generating noise
        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = {})
        
        # Get the length of each dimension
        N_l = x_l.shape[-1]
        N_t = x_t.shape[-1]

        # Generate a random normal vector
        z = np.random.normal(size = (N_l*N_t, size))
        
        # Transform normal vector using the Cholesky factor
        r = jnp.einsum("ij,j...->i...", stored_values["L_cho"], z)

        # Return the right shape based on whether generating more than one sample or not
        if size == 1:
            R = r.reshape((N_l, N_t))
        else:
            R = r.reshape((N_l, N_t, size))
        
        return R
    
    
    def predict(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_l_pred: JAXArray,
        x_t: JAXArray,
        x_t_pred: JAXArray,
        R: JAXArray,
        M_s: JAXArray,
        wn: Optional[bool] = True,
        return_std_dev: Optional[bool] = True,
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
            hp (Pytree): Hyperparameters needed to build the covariance matrix ``K``. Will be
                unaffected if additional mean function parameters are also included.
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
        # Get the length of each prediction dimension
        N_l_pred = x_l_pred.shape[-1]
        N_t_pred = x_t_pred.shape[-1]
        
        # Calculate the decomposition of K for the required K^-1 calculations
        stored_values = self.decomp_fn(hp, x_l, x_t)
            
        # Calculate the covariance between the observed and predicted points
        K_s = self.K(hp, x_l, x_l_pred, x_t, x_t_pred, wn = False)
        
        # Calculate the covariance between predicted points with other predicted points
        K_ss = self.K(hp, x_l_pred, x_l_pred, x_t_pred, x_t_pred, wn = wn)

        # Flatten residuals vector
        r = R.ravel("C")
        
        # Use forward and backward substitution to solve K^-1 r using the Cholesky factor
        alpha = JLA.solve_triangular(stored_values["L_cho"], r, trans = 1)
        K_inv_R = JLA.solve_triangular(stored_values["L_cho"], alpha, trans = 0)
        
        # Computes the GP predictive mean
        gp_mean = K_s.T @ K_inv_R
        gp_mean = M_s + gp_mean.reshape(N_l_pred, N_t_pred)
        
        # Prepare to calculate K^-1 K_s in the predictive covariance calculation
        L_inv_K_s = JLA.solve_triangular(stored_values["L_cho"], K_s, trans = 1)
        
        if return_std_dev:
            # Get diagonal of covariance of predicted locations with other predicted locations
            pred_err = jnp.diag(K_ss)
            
            # Subtract off term related to covariance between predicted and observed locations
            # This method efficiently calculates only the diagonal of the term
            pred_err -= (L_inv_K_s**2).sum(0)
            
            # Convert shape to (N_l_pred, N_t_pred) to match observed data but at predicted locations
            pred_err = pred_err.reshape(N_l_pred, N_t_pred)
            
            # Convert from variance to std dev
            pred_err = jnp.sqrt(pred_err)
        else:
            # Directly compute the predictive covariance matrix
            pred_err = K_ss - L_inv_K_s.T @ L_inv_K_s
        
        return gp_mean, pred_err
    
