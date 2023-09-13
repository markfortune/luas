import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from functools import partial
from typing import Any, Optional, Callable, Union, Dict, Tuple
import jax.config
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import jit, grad, hessian

from .luas_types import Scalar, PyTree, JAXArray, Kernel
from .jax_convenience_fns import order_list, pytree2D_to_array, array_to_pytree2D
from .kronecker_functions import kron_prod, kronecker_inv_vec

__all__ = ["GP"]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)

PyTreeOrArray = Union[JAXArray, PyTree]

class GP(object):
    """GP class using the Kronecker product structure of the covariance matrix to extend GPs to
    2D regular structures relatively cheaply.
    
               --------------------------------
              |                                |
              |                                |
              |                                |
    x_l (N_l) |       Y (N_l x N_t) data       |
    input     |                                |
              |                                |
              |                                |
               --------------------------------  
                        x_t (N_t) input
    
    Must be two input dimensions, and each are treated independently to create a kernel in
    each dimension. These are then combined with the sum of two Kronecker products to produce
    the full covariance matrix, although using eigendecomposition allows us to avoid computing and
    inverting huge matrices.

    Kl and Sl (both MxM) will be the covariance matrices in the wavelength/vertical direction,
    Kt and St (both NxN) will be the covariance matrices in the time/horizontal direction.
    The full covariance is given by K = (Kl KRON Kt + Sl KRON St).
    
    Args:
        initial_params (PyTree): dictionary of starting guesses for mean function parameters and hyperparameters
        mfp_to_fit (list): a list of the names of mean function parameters which are being fit (i.e. not fixed)
        hp_to_fit (list): a list of the names of hyperparameters which are being fit (i.e. not fixed)
        x_l (JAXArray): array containing wavelength dimension
        x_t (JAXArray): array containing time dimension
        Y (JAXArray): 2D array containing the observed data
        kf (Kernel): kernel object which has already been initialised with the desired kernel function
        mf (Callable, optional): mean function, by default returns zeros. Needs to be in the format mf(p, x_l, x_t)
        logPrior (Callable, optional): log prior function, optional as PyMC can also be used for priors but more complex priors 
            can be used with this
        log_params (list, optional): list of variable names which it is desired to fit for the log of the parameter (uses log base 10)
        transform_fn (Callable, optional): function for transforming variables being fit to variable values. Where log parameters are transformed
        transform_args (tuple, optional): arguments to transform_fn. If using default transform_fn then this can be left as None
    
    """
    
    def __init__(
        self,
        initial_params: dict,
        mfp_to_fit: list[str],
        hp_to_fit: list[str],
        x_l: JAXArray,
        x_t: JAXArray,
        Y: JAXArray,
        kf: Kernel,
        mf: Optional[Callable] = None,
        logPrior: Optional[Callable] = None,
        gaussian_prior_dict: Optional[Dict] = None,
        log_params: Optional[list[str]] = None,
        transform_fn: Optional[Callable] = None,
        transform_args: Any = None,
    ):
        # Initialise variables
        self.p_initial = initial_params
        self.p = {k:initial_params[k] for k in (mfp_to_fit + hp_to_fit)}
        self.x_l = x_l
        self.x_t = x_t
        self.Y = Y
        self.mfp = mfp_to_fit
        self.hp = hp_to_fit
        self.kf = kf
        self.transform_fn = transform_fn
        self.N_l = self.Y.shape[0]
        self.N_t = self.Y.shape[1]
        self.cov_dict = None
        self.gaussian_prior_dict = gaussian_prior_dict
        self.storage_dict = {}
        
        if log_params is None:
            print("Note: No log parameters given, defaulting to all hyperparameters")
            self.log_params = self.hp
        else:
            self.log_params = log_params
        
        # Flatten parameter dict into an array to provide log-likelihood functions which
        # take array inputs which is required for some optimisers and samplers including by PyMC
        # Also returns a function to convert an array back into a dictionary
        self.p_arr, self.make_p_dict = ravel_pytree(self.p)
        
        # Mean function returns zeros by default
        if mf is None:
            print("Note: No mean function given")
            self.mf = lambda p, x_l, x_t: jnp.zeros((self.N_l, self.N_t))
        else:
            self.mf = mf
            
        if transform_fn is None:
            print("Note: No parameter transformation function given, defaulting to default_param_transform")
            self.transform_fn = self.default_param_transform
            
        # Use default transform_fn arguments if not specified
        if transform_args is None:
            transform_args = (self.p_initial, self.log_params)
            
        # Log Prior function returns zero by default
        if logPrior is None and self.gaussian_prior_dict is None:
            print("Note: No log prior given, setting to zero")
            self.logPrior = lambda p_untransf, p_transf: 0.
        elif logPrior is None and self.gaussian_prior_dict is not None:
            self.logPrior = self.make_logPrior(self.gaussian_prior_dict)
        elif logPrior is not None and self.gaussian_prior_dict is None:
            self.logPrior = logPrior
        else:
            raise Exception("Error: Both a logPrior and a gaussian prior dictionary " +
                            "to create a logPrior function have been specified")
           
        # Convenient function for transforming untransformed parameter dict into a transformed dict
        self.transform_p = lambda p: self.transform_fn(p, *transform_args)
        self.logPrior_transf = lambda p_untransf: self.logPrior(p_untransf, self.transform_fn(p_untransf, *transform_args))
        self.grad_logPrior_transf = grad(self.logPrior_transf)
        self.hessian_logPrior_transf = hessian(self.logPrior_transf)
        
        
    def default_param_transform(
        self,
        p_vary: PyTree,
        p_fixed: PyTree,
        log_params: list[str],
    ) -> PyTree:

        # Copy to avoid transformation affecting stored values
        p = deepcopy(p_fixed)

        # Update fixed values with values being varied
        p.update(p_vary)

        # Transform log parameters
        for name in log_params:
            p[name] = jnp.power(10, p[name])

        return p

    
    def check_storage_dict(self, storage_dict: PyTree) -> Tuple[PyTree, bool]:
        if storage_dict is None:
            storage_dict = {}
            return_storage_dict = False
        else:
            return_storage_dict = True
        return storage_dict, return_storage_dict
    

    def check_p_type(self, p: PyTreeOrArray) -> Tuple[PyTree, bool]:
        
        p_is_array = not isinstance(p, dict)
        
        if p_is_array:
            # Convert array to a parameter dictionary and then perform as logL as normal
            p = self.make_p_dict(p)
            
        return p, p_is_array
    
    
    def build_storage_dict(self, p: PyTree) -> PyTree:
        
        return self.kf.decomp_fn(p, self.x_l, self.x_t, transform_fn = self.transform_p)
    
    
    @partial(jax.jit, static_argnums=(0,))
    def logL(
        self,
        p: PyTreeOrArray,
        storage_dict: Optional[PyTree] = None,
    ) -> Union[Scalar, Tuple[Scalar, PyTree]]:
        
        storage_dict, return_storage_dict = self.check_storage_dict(storage_dict)
        p_dict, p_is_array = self.check_p_type(p)
        
        # Calculate log-likelihood
        logL, storage_dict = self.kf.logL(p_dict, self.x_l, self.x_t, self.Y, self.mf, storage_dict = storage_dict, transform_fn = self.transform_p)
        
        if return_storage_dict:
            return logL, storage_dict
        else:
            return logL
    
    
    
    @partial(jax.jit, static_argnums=(0,))
    def grad_logL(
        self,
        p: PyTreeOrArray,
        storage_dict: Optional[PyTree] = None,
    ) -> Union[PyTreeOrArray, Tuple[PyTreeOrArray, PyTree]]:
        
        storage_dict, return_storage_dict = self.check_storage_dict(storage_dict)
        p_dict, p_is_array = self.check_p_type(p)
        
        grad_dict, storage_dict = self.kf.grad_logL(p_dict, self.x_l, self.x_t, self.Y, self.mf, storage_dict = storage_dict,
                                      fit_mfp = self.mfp, transform_fn = self.transform_p)
        
        if p_is_array:
            grad_vals = ravel_pytree(grad_dict)[0]
        else:
            grad_vals = grad_dict
        
        if return_storage_dict:
            return grad_vals, storage_dict
        else:
            return grad_vals
    
    
    @partial(jax.jit, static_argnums=(0,))
    def value_and_grad_logL(
        self,
        p: PyTreeOrArray,
        storage_dict: Optional[PyTree] = None,
    ) -> Union[Tuple[Scalar, PyTreeOrArray], Tuple[Scalar, PyTreeOrArray, PyTree]]:
        
        storage_dict, return_storage_dict = self.check_storage_dict(storage_dict)
        p_dict, p_is_array = self.check_p_type(p)
        
        (logL, storage_dict), grad_dict = self.kf.value_and_grad_logL(p_dict, self.x_l, self.x_t, self.Y, self.mf, storage_dict = storage_dict,
                                                      fit_mfp = self.mfp, transform_fn = self.transform_p)
        
        if p_is_array:
            grad_vals = ravel_pytree(grad_dict)[0]
        else:
            grad_vals = grad_dict
        
        if return_storage_dict:
            return logL, grad_vals, storage_dict
        else:
            return logL, grad_vals
        
    
    def hessian_logL(
        self,
        p: PyTreeOrArray,
        large: Optional[bool] = False,
        storage_dict: Optional[PyTree] = None
    ) -> Union[PyTreeOrArray, Tuple[PyTreeOrArray, PyTree]]:
        
        storage_dict, return_storage_dict = self.check_storage_dict(storage_dict)
        p_dict, p_is_array = self.check_p_type(p)
        
        if large:
            hessian_dict, storage_dict = self.kf.large_hessian_logL(p_dict, self.x_l, self.x_t, self.Y, self.mf,
                                                                    storage_dict = storage_dict, fit_mfp = self.mfp, fit_hp = self.hp,
                                                                    transform_fn = self.transform_p)
        else:
            hessian_dict, storage_dict = self.kf.hessian_logL(p_dict, self.x_l, self.x_t, self.Y, self.mf, storage_dict = storage_dict,
                                                              fit_mfp = self.mfp, fit_hp = self.hp, transform_fn = self.transform_p)
        
        if p_is_array:
            hessian_vals = pytree2D_to_array(p_dict, hessian_dict)
        else:
            hessian_vals = hessian_dict
            
        if return_storage_dict:
            return hessian_vals, storage_dict
        else:
            return hessian_vals
        
        
    
    def make_logPrior(self, prior_dict: PyTree) -> Callable:
        
        def logPrior_fn(p_untransf, p_transf):
            
            logPrior = 0.
            for par in prior_dict.keys():
                logPrior += -0.5*(((p_transf[par] - prior_dict[par][0])/prior_dict[par][1])**2).sum() 
            return logPrior
        
        return logPrior_fn

    
    @partial(jax.jit, static_argnums=(0,))
    def logP(
        self,
        p: PyTreeOrArray,
        storage_dict: Optional[PyTree] = None,
    ) -> Union[Scalar, Tuple[Scalar, PyTree]]:
        
        storage_dict, return_storage_dict = self.check_storage_dict(storage_dict)
        p_dict, p_is_array = self.check_p_type(p)
        
        # Calculate log-likelihood
        logL, storage_dict = self.kf.logL(p_dict, self.x_l, self.x_t, self.Y, self.mf, storage_dict = storage_dict, transform_fn = self.transform_p)
        logPrior = self.logPrior_transf(p_dict)
        logP = logPrior + logL
        
        if return_storage_dict:
            return logP, storage_dict
        else:
            return logP
    
    
    @partial(jax.jit, static_argnums=(0,))
    def grad_logP(
        self,
        p: PyTreeOrArray,
        storage_dict: Optional[PyTree] = None,
    ) -> Union[PyTreeOrArray, Tuple[PyTreeOrArray, PyTree]]:
        
        storage_dict, return_storage_dict = self.check_storage_dict(storage_dict)
        p_dict, p_is_array = self.check_p_type(p)
        
        grad_dict, storage_dict = self.kf.grad_logL(p_dict, self.x_l, self.x_t, self.Y, self.mf, storage_dict = storage_dict,
                                      fit_mfp = self.mfp, transform_fn = self.transform_p)
        
        prior_grad_dict = self.grad_logPrior_transf(p_dict)
        
        # Combine
        grad_dict.update({k: prior_grad_dict[k]+grad_dict[k] for k in self.mfp + self.hp})
        
        if p_is_array:
            grad_vals = ravel_pytree(grad_dict)[0]
        else:
            grad_vals = grad_dict
        
        if return_storage_dict:
            return grad_vals, storage_dict
        else:
            return grad_vals
        
        
    @partial(jax.jit, static_argnums=(0,))
    def value_and_grad_logP(
        self,
        p: PyTreeOrArray,
        storage_dict: Optional[PyTree] = None,
    ) -> Union[Tuple[Scalar, PyTreeOrArray], Tuple[Scalar, PyTreeOrArray, PyTree]]:
        
        storage_dict, return_storage_dict = self.check_storage_dict(storage_dict)
        p_dict, p_is_array = self.check_p_type(p)
        
        (logL, storage_dict), grad_dict = self.kf.value_and_grad_logL(p_dict, self.x_l, self.x_t, self.Y, self.mf, storage_dict = storage_dict,
                                                      fit_mfp = self.mfp, transform_fn = self.transform_p)
        logPrior = self.logPrior_transf(p_dict)
        logP = logPrior + logL
        
        prior_grad_dict = self.grad_logPrior_transf(p_dict)
        
        # Combine
        grad_dict.update({k: prior_grad_dict[k]+grad_dict[k] for k in self.mfp + self.hp})
        
        if p_is_array:
            grad_vals = ravel_pytree(grad_dict)[0]
        else:
            grad_vals = grad_dict
        
        if return_storage_dict:
            return logP, grad_vals, storage_dict
        else:
            return logP, grad_vals
    
    
    def hessian_logP(
        self,
        p: PyTreeOrArray,
        large: Optional[bool] = False,
        storage_dict: Optional[PyTree] = None
    ) -> Union[PyTreeOrArray, Tuple[PyTreeOrArray, PyTree]]:
        
        storage_dict, return_storage_dict = self.check_storage_dict(storage_dict)
        p_dict, p_is_array = self.check_p_type(p)
        
        logPrior_hess = self.hessian_logPrior_transf(p_dict)
        
        if large:
            hessian_dict, storage_dict = self.kf.large_hessian_logL(p_dict, self.x_l, self.x_t, self.Y, self.mf,
                                                                    storage_dict = storage_dict, fit_mfp = self.mfp, fit_hp = self.hp,
                                                                    transform_fn = self.transform_p, make_p_dict = self.make_p_dict)
        else:
            hessian_dict, storage_dict = self.kf.hessian_logL(p_dict, self.x_l, self.x_t, self.Y, self.mf, storage_dict = storage_dict,
                                                              fit_mfp = self.mfp, fit_hp = self.hp, transform_fn = self.transform_p)
        
        for i in self.mfp + self.hp:
            for j in self.mfp + self.hp:
                hessian_dict[i][j] += logPrior_hess[i][j]
                
        if p_is_array:
            hessian_vals = pytree2D_to_array(p_dict, hessian_dict)
        else:
            hessian_vals = hessian_dict
            
        if return_storage_dict:
            return hessian_vals, storage_dict
        else:
            return hessian_vals

    
    def hessian_to_covariance_mat(
        self,
        hessian_val: PyTree,
        regularise: Optional[bool] = False,
        regularise_sigma: Optional[Scalar] = 0.1
    ) -> PyTree:
        
        if isinstance(hessian_val, dict):
            hess_mat = pytree2D_to_array(self.p, hessian_val)
        else:
            hess_mat = hessian_val
            
        hess_mat = (hess_mat + hess_mat.T)/2.
        self.cov_mat = jnp.linalg.inv(-hess_mat)
        
        if regularise:
            cov_diag = jnp.diag(self.cov_mat)
            neg_ind = cov_diag < 0.
            regularise_vec = -0.5*(1/regularise_sigma**2)*neg_ind
            hess_mat += jnp.diag(regularise_vec)
            self.cov_mat = jnp.linalg.inv(-hess_mat)
            
#             print(f"Regularised locations where a {regularise_sigma} prior sigma was added: ", jnp.arange(cov_diag.size)[neg_ind], self.cov_order)
            
        self.cov_dict = array_to_pytree2D(self.p, self.cov_mat)
        
        return self.cov_dict
    
    
    def get_cov(self, p_list: list[str]) -> JAXArray:
        
        order_dict = {}

        i = 0
        for param in p_list:
            if isinstance(self.p_initial[param], JAXArray):
                param_size = self.p_initial[param].size
                order_dict[param] = jnp.arange(i, i+param_size)
                i += param_size
            else:
                order_dict[param] = jnp.arange(i, i+1)
                i += 1
        
        cov_mat = jnp.zeros((i, i))

        for (k1, v1) in order_dict.items():
            for (k2, v2) in order_dict.items():
                if v1.size > 1 and v2.size == 1:
                    cov_mat = cov_mat.at[jnp.ix_(v1, v2)].set(self.cov_dict[k1][k2].reshape((v1.size, v2.size)))
                else:
                    cov_mat = cov_mat.at[jnp.ix_(v1, v2)].set(self.cov_dict[k1][k2])

        return cov_mat
    
    
    @partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        p_untransf: PyTreeOrArray,
        storage_dict: Optional[PyTree] = None,
    ) -> Union[Tuple[JAXArray, JAXArray, JAXArray, PyTree], Tuple[JAXArray, JAXArray, JAXArray]]:
        
        storage_dict, return_storage_dict = self.check_storage_dict(storage_dict)
        p_dict, p_is_array = self.check_p_type(p_untransf)
        
        gp_mean, sigma_diag, M = self.kf.predict(p_dict, self.x_l, self.x_l, self.x_t, self.x_t, self.Y, self.mf,
                                                 storage_dict = storage_dict, transform_fn = self.transform_p)
        
        if return_storage_dict:
            return gp_mean, sigma_diag, M, storage_dict
        else:
            return gp_mean, sigma_diag, M
    
    
    def clip_outliers(self, p: PyTreeOrArray, sigma: Scalar) -> JAXArray:
        
        gp_mean, sigma_diag, M = self.predict(p)
        
        R = self.Y - gp_mean
        Z = jnp.abs(R/jnp.sqrt(sigma_diag))
        
        Y_clean = self.Y.copy()
        
        plt.imshow(Z, aspect = 'auto')
        plt.colorbar()
        plt.show()
        
        outliers = Z > sigma
        
        Y_clean = Y_clean.at[outliers].set(gp_mean[outliers])
        
        print("Outliers removed = ", (outliers).sum())
        
        if outliers.sum() > 0:
            plt.title("Locations of Outliers Removed")
            plt.imshow(self.Y, aspect = 'auto')
            y, x = jnp.where(outliers)
            plt.scatter(x, y, color='red', marker='x')
            plt.show()

        return Y_clean
    
    
    @partial(jax.jit, static_argnums=(0,))
    def calc_mf(self, p: PyTreeOrArray) -> JAXArray:
        p, p_is_array = self.check_p_type(p)
        
        return self.mf(self.transform_p(p), self.x_l, self.x_t)
    
    @partial(jax.jit, static_argnums=(0,))
    def calc_residuals(self, p: PyTreeOrArray) -> JAXArray:
        p, p_is_array = self.check_p_type(p)
        
        return self.Y - self.mf(self.transform_p(p), self.x_l, self.x_t)
    
    
    def autocorrelate(
        self,
        p: PyTreeOrArray,
        l_sep = None,
        t_sep = None,
    ) -> JAXArray:
        
        if l_sep is None:
            l_sep = self.N_l-1
        if t_sep is None:
            t_sep = self.N_t-1
            
        gp_mean, sigma_diag, M = self.predict(p)
        res = self.Y - gp_mean

        auto_corr = jax.scipy.signal.correlate2d(res, res)
        
        n_l, n_t = auto_corr.shape
        auto_corr_centre = ((n_l-1)//2, (n_t-1)//2)
        auto_corr /= auto_corr[auto_corr_centre[0], auto_corr_centre[1]]
        auto_corr = auto_corr.at[auto_corr_centre[0], auto_corr_centre[1]].set(0.)
        
        if self.x_l.ndim > 1:
            self.x_l_pred = self.x_l[:, 0]
        else:
            self.x_l_pred = self.x_l
            
        if self.x_t.ndim > 1:
            self.x_t_pred = self.x_t[:, 0]
        else:
            self.x_t_pred = self.x_t
        
        t_step = self.x_t_pred.ptp()/(self.N_t-1)
        l_step = self.x_l_pred.ptp()/(self.N_l-1)

        plt.imshow(auto_corr[auto_corr_centre[0]-l_sep:auto_corr_centre[0]+l_sep+1, auto_corr_centre[1]-t_sep:auto_corr_centre[1]+t_sep+1],
                   aspect = 'auto', interpolation = "none",
                   extent = [t_step*-(t_sep+0.5), t_step*(t_sep+0.5), l_step*(l_sep+0.5), l_step*-(l_sep+0.5)])
        plt.xlabel("$\Delta t$")
        plt.ylabel("$\Delta \lambda$")
        plt.colorbar()
        plt.show()

        return auto_corr
    
    
    def plot(self, p_untransf: PyTreeOrArray, fig: Optional[plt.Figure] = None):
    
        #run prediction
        gp_mean, gp_cov, M = self.predict(p_untransf)
        
        if fig is None: fig = plt.figure(figsize = (20, 5))
        ax = fig.subplots(1, 4)
        
        ax[0].set_title("Data")
        ax[0].imshow(self.Y, aspect = 'auto')
        ax[1].set_title("GP mean (incl. mean function)")
        ax[1].imshow(gp_mean, aspect = 'auto')
        ax[2].set_title("GP mean (excl. mean function)")
        ax[2].imshow(gp_mean - M, aspect = 'auto')
        ax[3].set_title("Residual noise")
        ax[3].imshow(self.Y - gp_mean, aspect = 'auto')

        ax[0].set_ylabel('x_l')
        for i in range(4):
            ax[i].set_xlabel('x_t')
        plt.show()
    