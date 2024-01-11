import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from functools import partial
from jax import jit, grad, hessian
import jax.config
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from .jax_convenience_fns import (
    array_to_pytree_2D,
    pytree_to_array_2D,
    order_list,
    large_hessian_calc,
    transf_from_unbounded_params,
    transf_to_unbounded_params,
)

from typing import Any, Optional, Callable, Union, Dict, Tuple
from .luas_types import Scalar, PyTree, JAXArray, Kernel

__all__ = ["GP"]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


class GP(object):
    """GP class using the Kronecker product structure of the covariance matrix to extend GPs to
    2D regular structures relatively cheaply. 
    
    Must be two input dimensions, and each are treated independently to create a kernel in
    each dimension. These are then combined with the sum of two Kronecker products to produce
    the full covariance matrix, although using eigendecomposition allows us to avoid computing and
    inverting huge matrices.

    Kl and Sl (both MxM) will be the covariance matrices in the wavelength/vertical direction,
    Kt and St (both NxN) will be the covariance matrices in the time/horizontal direction.
    The full covariance is given by K = (Kl KRON Kt + Sl KRON St).
    
    Args:
        kf (Kernel): Kernel object which has already been initialised with the desired kernel function
        x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s).
            May be of shape (M,) or (d,M) for d different wavelength/vertical regression variables.
        x_l (JAXArray): Array containing time/horizontal dimension regression variable(s).
            May be of shape (N,) or (d,N) for d different time/horizontal regression variables.
        mf (Callable, optional): The deterministic mean function, by default returns a JAXArray of zeros.
            Needs to be in the format mf(params, x_l, x_t) and returns a JAXArray of shape (M,N)
            matching the shape of the observed data Y.
        logPrior (Callable, optional): Log prior function, by default returns zero.
            Needs to be in the format logPrior(p) and return a scalar.
    """
    
    def __init__(
        self,
        kf: Kernel,
        x_l: JAXArray,
        x_t: JAXArray,
        mf: Optional[Callable] = None,
        logPrior: Optional[Callable] = None,
    ):
        # Initialise variables
        self.kf = kf
        self.x_l = x_l
        self.x_t = x_t
        self.N_l = self.x_l.shape[-1]
        self.N_t = self.x_t.shape[-1]
        self.storage_dict = {}
        
        
        # Mean function returns zeros by default
        if mf is None:
            print("Note: No mean function given, defaulting to zero")
            self.mf = lambda p, x_l, x_t: jnp.zeros((x_l.shape[-1], x_t.shape[-1]))
        else:
            self.mf = mf
            
        # Log Prior function returns zero by default
        if logPrior is None:
            print("Note: No log prior given, setting to zero")
            self.logPrior = lambda p: 0.
        else:
            self.logPrior = logPrior


    def calc_storage_dict(self, p: PyTree) -> PyTree:
        
        return self.kf.decomp_fn(p, self.x_l, self.x_t, storage_dict = {})

    
    @partial(jax.jit, static_argnums=(0,))
    def logL(
        self,
        p: PyTree,
        Y: JAXArray,
    ) -> Union[Scalar, Tuple[Scalar, PyTree]]:
        
        R = Y - self.mf(p, self.x_l, self.x_t)
        
        return self.kf.logL(p, self.x_l, self.x_t, R, {})[0]

    
    @partial(jax.jit, static_argnums=(0,))
    def logL_stored(
        self,
        p: PyTree,
        Y: JAXArray,
        storage_dict: PyTree,
    ) -> Union[Scalar, Tuple[Scalar, PyTree]]:
        
        R = Y - self.mf(p, self.x_l, self.x_t)
    
        return self.kf.logL(p, self.x_l, self.x_t, R, storage_dict)

    @partial(jax.jit, static_argnums=(0,))
    def logP(
        self,
        p: PyTree,
        Y: JAXArray,
    ) -> Union[Scalar, Tuple[Scalar, PyTree]]:
        
        logPrior = self.logPrior(p)
        logL = self.logL(p, Y)
        logP = logPrior + logL
        
        return logP

    
    @partial(jax.jit, static_argnums=(0,))
    def logP_stored(
        self,
        p: PyTree,
        Y: JAXArray,
        storage_dict: PyTree,
    ) -> Union[Scalar, Tuple[Scalar, PyTree]]:

        logPrior = self.logPrior(p)
        logL, storage_dict = self.logL_stored(p, Y, storage_dict)
        logP = logPrior + logL
        
        return logP, storage_dict

    
    @partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        p: PyTree,
        Y: JAXArray,
        x_l_pred = None,
        x_t_pred = None,
    ) -> Union[Tuple[JAXArray, JAXArray, JAXArray, PyTree], Tuple[JAXArray, JAXArray, JAXArray]]:

        if x_l_pred is None:
            x_l_pred = self.x_l
        if x_t_pred is None:
            x_t_pred = self.x_t

        # Generate mean function and compute residuals
        R = Y - self.mf(p, self.x_l, self.x_t)
        M_pred = self.mf(p, x_l_pred, x_t_pred)
            
        gp_mean, sigma_diag = self.kf.predict(p, self.x_l, x_l_pred, self.x_t, x_t_pred, R, M_pred,
                                                 storage_dict = {})
        
        return gp_mean, sigma_diag, M_pred
    
    
    def sigma_clip(self, p: PyTree, Y:JAXArray, sigma: Scalar, plot = True) -> JAXArray:
        
        gp_mean, sigma_diag, M = self.predict(p, Y)
        
        res = Y - gp_mean
        Z = jnp.abs(res/jnp.sqrt(sigma_diag))
        
        Y_clean = jnp.array(Y.copy())

        if plot:
            plt.title("Std. Dev. of Residuals")
            plt.imshow(Z, aspect = 'auto')
            plt.colorbar()
            plt.show()
            
        outliers = Z > sigma
        
        Y_clean = Y_clean.at[outliers].set(gp_mean[outliers])
        
        print("Outliers removed = ", (outliers).sum())
        
        if outliers.sum() > 0 and plot:
            plt.title("Locations of Outliers Removed")
            plt.imshow(Y, aspect = 'auto')
            y, x = jnp.where(outliers)
            plt.scatter(x, y, color='red', marker='x')
            plt.show()

        return Y_clean
    
    def plot(self, p: PyTree, Y, fig: Optional[plt.Figure] = None):
    
        #run prediction
        gp_mean, gp_cov, M = self.predict(p, Y)
        
        if fig is None: fig = plt.figure(figsize = (20, 5))
        ax = fig.subplots(1, 4, sharey = True)
        
        ax[0].set_title("Data")
        ax[0].pcolormesh(self.x_t, self.x_l, Y, shading = "nearest")
        ax[1].set_title("Mean function")
        ax[1].pcolormesh(self.x_t, self.x_l, M, shading = "nearest")
        ax[2].set_title("GP mean (excl. mean function)")
        ax[2].pcolormesh(self.x_t, self.x_l, gp_mean - M, shading = "nearest")
        ax[3].set_title("Residual noise")
        ax[3].pcolormesh(self.x_t, self.x_l, Y - gp_mean, shading = "nearest")

        ax[0].set_ylabel('x_l')
        for i in range(4):
            ax[i].set_xlabel('x_t')

        plt.gca().invert_yaxis()
        plt.show()
    

    def plot_K(self, p, x_l_pred = None, x_t_pred = None, plot = True, **kwargs):

        if x_l_pred is None:
            x_l_pred = self.x_l
        if x_t_pred is None:
            x_t_pred = self.x_t
            
        K = self.kf.K(p, self.x_l, x_l_pred, self.x_t, x_t_pred, **kwargs)

        if plot:
            plt.imshow(K)
            plt.show()

        return K

    
    def autocorrelate(
        self,
        p: PyTree,
        Y: JAXArray,
        l_sep = None,
        t_sep = None,
    ) -> JAXArray:
        
        if l_sep is None:
            l_sep = self.N_l-1
        if t_sep is None:
            t_sep = self.N_t-1
            
        gp_mean, sigma_diag, M = self.predict(p, Y)
        res = Y - gp_mean

        auto_corr = jax.scipy.signal.correlate2d(res, res)
        
        n_l, n_t = auto_corr.shape
        auto_corr_centre = ((n_l-1)//2, (n_t-1)//2)
        auto_corr /= auto_corr[auto_corr_centre[0], auto_corr_centre[1]]
        auto_corr = auto_corr.at[auto_corr_centre[0], auto_corr_centre[1]].set(0.)
        
        if self.x_l.ndim > 1:
            x_l_pred = self.x_l[:, 0]
        else:
            x_l_pred = self.x_l
            
        if self.x_t.ndim > 1:
            x_t_pred = self.x_t[:, 0]
        else:
            x_t_pred = self.x_t
        
        t_step = x_t_pred.ptp()/(self.N_t-1)
        l_step = x_l_pred.ptp()/(self.N_l-1)

        plt.imshow(auto_corr[auto_corr_centre[0]-l_sep:auto_corr_centre[0]+l_sep+1, auto_corr_centre[1]-t_sep:auto_corr_centre[1]+t_sep+1],
                   aspect = 'auto', interpolation = "none",
                   extent = [t_step*-(t_sep+0.5), t_step*(t_sep+0.5), l_step*(l_sep+0.5), l_step*-(l_sep+0.5)])
        plt.xlabel("$\Delta t$")
        plt.ylabel("$\Delta \lambda$")
        plt.colorbar()
        plt.show()

        return auto_corr
    
    
    @partial(jax.jit, static_argnums=(0,))
    def logL_hessianable(
        self,
        p: PyTree,
        Y: JAXArray,
    ) -> Union[Scalar, Tuple[Scalar, PyTree]]:
        
        R = Y - self.mf(p, self.x_l, self.x_t)
        
        return self.kf.logL_hessianable(p, self.x_l, self.x_t, R, {})[0]

    
    @partial(jax.jit, static_argnums=(0,))
    def logP_hessianable(
        self,
        p: PyTree,
        Y: JAXArray,
    ) -> Union[Scalar, Tuple[Scalar, PyTree]]:
        
        logPrior = self.logPrior(p)
        logL = self.logL_hessianable(p, Y)
        logP = logPrior + logL
        
        return logP

    
    def hessian_calc(
        self,
        p: PyTree,
        Y: JAXArray,
        fn = None,
        return_array = False,
        large = False,
    ) -> Union[Scalar, Tuple[Scalar, PyTree]]:

        if fn is None:
            fn = self.logP_hessianable
        
        if large:
            p_arr, make_p_dict = ravel_pytree(p)
            logL_wrapper = lambda p_arr: fn(make_p_dict(p_arr), Y)
            hessian_arr = large_hessian_calc(logL_wrapper, p_arr)

            if return_array:
                return hessian_arr
            else:
                return array_to_pytree_2D(p, hessian_arr)

        elif return_array:
            p_arr, make_p_dict = ravel_pytree(p)
            logL_wrapper = lambda p_arr: fn(make_p_dict(p_arr), Y)
            return jax.hessian(logL_wrapper)(p_arr)

        else:
            return jax.hessian(fn)(p, Y)
        
        
    
    def laplace_approx(
        self,
        p: PyTree,
        Y: PyTree,
        regularise: Optional[bool] = True,
        regularise_const: Optional[Scalar] = 100.,
        fn = None,
        return_array = False,
        large = False,
        hessian_array = None,
    ) -> PyTree:

        if fn is None:
            fn = self.logP_hessianable

        if hessian_array is None:
            hessian_array = self.hessian_calc(p, Y, fn = fn, return_array = True, large = large)
        
        hessian_array = (hessian_array + hessian_array.T)/2.
        cov_mat = jnp.linalg.inv(-hessian_array)
        
        if regularise:
            cov_diag = jnp.diag(cov_mat)
            neg_ind = cov_diag < 0.
            num_neg_diag_vals = neg_ind.sum()
            
            regularise_vec = -regularise_const*neg_ind
            hessian_array += jnp.diag(regularise_vec)
            cov_mat = jnp.linalg.inv(-hessian_array)

            p_arr, make_p_dict = ravel_pytree(p)
            regularised_values = make_p_dict(neg_ind)
            
            for par in p.keys():
                if not jnp.any(regularised_values[par]):
                    del regularised_values[par]
            
            cov_diag = jnp.diag(cov_mat)
            neg_ind = cov_diag < 0.
            num_neg_diag_vals_remaining = neg_ind.sum()
                    
            print(f"Initial number of negative values on diagonal of covariance matrix = {num_neg_diag_vals}\nCorresponding to parameters: {regularised_values}.\nRemaining number of negative values = {num_neg_diag_vals_remaining}.")

        ordered_param_list = order_list(list(p.keys()))
        
        if return_array:
            return cov_mat, ordered_param_list
        else:
            return array_to_pytree_2D(p, cov_mat)
    
    
    def varying_params_wrapper(self, p, vars = None, fixed_vars = None):
        if vars is not None and fixed_vars is None:
            p_fit = {par:np.array(p[par]) for par in vars}
        if vars is None and fixed_vars is not None:
            p_fit = {par:np.array(p[par]) for par in p.keys() if par not in fixed_vars}
        elif vars is None and fixed_vars is None:
            p_fit = {par:np.array(p[par]) for par in p.keys()}
        elif vars is not None and fixed_vars is not None:
            raise Exception("Both vars and fixed_vars cannot be defined!")

        p_fit_arr, make_p_fit_dict = ravel_pytree(p_fit)
        
        p_fixed = deepcopy(p)
        def make_p(p_arr):
            p_fit = make_p_fit_dict(p_arr)
            p_fixed.update(p_fit)
            return p_fixed

        return p_fit, make_p
    
    
    def laplace_approx_with_bounds(self, p, Y, param_bounds, vars = None, fixed_vars = None, fn = None, large = False, **kwargs):

        if fn is None:
            fn = self.logP_hessianable
        
        p_fit, make_p = self.varying_params_wrapper(p, vars = vars, fixed_vars = fixed_vars)
        p_fixed = deepcopy(p)
        
        p_transf = transf_to_unbounded_params(p_fit, param_bounds)
        def transf_back_to_p(p_transf):
            p_fit = transf_from_unbounded_params(p_transf, param_bounds)
            p_fixed.update(p_fit)
            return p_fixed

        pymc_logP_hessianable = lambda p_transf, Y: fn(transf_back_to_p(p_transf), Y)

        hessian_dict = self.hessian_calc(p_transf, Y, fn = pymc_logP_hessianable, return_array = False, large = large)
        
        for par in p_transf.keys():
            if par in param_bounds.keys():
                exp_min_x = jnp.exp(-p_transf[par])
                hessian_dict[par][par] += jnp.diag(-2*exp_min_x/(1+exp_min_x)**2)

        hessian_array = pytree_to_array_2D(p_transf, hessian_dict)
        
        cov_mat = self.laplace_approx(p_transf, Y, hessian_array = hessian_array, **kwargs)
            
        return cov_mat
