import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import Optional, Callable, Tuple, Any
import jax
from jax import grad, value_and_grad, hessian, vmap
import jax.numpy as jnp
import jax.scipy.linalg as JLA
from jax.flatten_util import ravel_pytree

from .luas_types import Kernel, PyTree, JAXArray, Scalar
from .kronecker_functions import kron_prod, make_vec, make_mat
from .jax_convenience_fns import array_to_pytree2D

__all__ = ["GeneralKernel"]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


class GeneralKernel(Kernel):
    def __init__(
        self,
        K: Optional[Callable] = None,
        Kl_fns: Optional[list[Callable]] = None,
        Kt_fns: Optional[list[Callable]] = None
    ):
        
        if K is None:
            self.K = self.build_kronecker_K(Kl_fns, Kt_fns)
        else:
            self.K = K
        self.decomp_fn = self.cholesky_decomp_no_stored_results
        
        self.grad_logL = grad(self.logL, has_aux = True)
        self.value_and_grad_logL = value_and_grad(self.logL, has_aux = True)
        self.hessian_logL = hessian(self.logL, has_aux = True)
    
    
    def build_kronecker_K(
        self,
        Kl_fns: Optional[list[Callable]] = None,
        Kt_fns: Optional[list[Callable]] = None,
    ) -> Callable:
    
        def K_kron(
            hp: PyTree,
            x_l1: JAXArray,
            x_l2: JAXArray,
            x_t1: JAXArray,
            x_t2: JAXArray, 
            wn: Optional[bool] = True,
        ) -> JAXArray:
        
            K = jnp.zeros((x_l1.shape[-1]*x_t1.shape[-1], x_l2.shape[-1]*x_t2.shape[-1]))
            for i in range(len(Kl_fns)):
                Kl = Kl_fns[i](hp, x_l1, x_l2, wn = wn)
                Kt = Kt_fns[i](hp, x_t1, x_t2, wn = wn)
                K += jnp.kron(Kl, Kt)

            return K
        
        return K_kron
            
        
    def cholesky_decomp_no_stored_results(
        self,
        hp_untransf: PyTree,
        x_l: JAXArray,
        x_t: JAXArray, 
        storage_dict: Optional[PyTree] = {}, 
        transform_fn: Optional[Callable] = None
    ) -> PyTree:
        
        if transform_fn is not None:
            hp = transform_fn(hp_untransf)
        else:
            hp = hp_untransf
    
        K = self.K(hp, x_l, x_l, x_t, x_t)
        
        storage_dict["L_cho"] = JLA.cholesky(K)
        storage_dict["logdetK"] = 2*jnp.log(jnp.diag(storage_dict["L_cho"])).sum()
        storage_dict["hp"] = deepcopy(hp)
        
        return storage_dict

        
    def logL(
        self,
        p_untransf: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        Y: JAXArray,
        mf: Callable,
        storage_dict: Optional[PyTree] = {},
        transform_fn: Optional[Callable] = None,
        fit_mfp: Optional[list[str]] = None,
        fit_hp: Optional[list[str]] = None
    ) -> Tuple[Scalar, PyTree]:
        
        storage_dict = self.decomp_fn(p_untransf, x_l, x_t, storage_dict = storage_dict, transform_fn = transform_fn)
        
        # Computes the log-likelihood
        if transform_fn is not None:
            p = transform_fn(p_untransf)
        else:
            p = p_untransf
            
        # Generate mean function and compute residuals
        M = mf(p, x_l, x_t)
        R = Y - M
        r = make_vec(R)
        alpha = JLA.solve_triangular(storage_dict["L_cho"], r, trans = 1)

        logL = - 0.5 *  jnp.sum(jnp.square(alpha)) - 0.5 * storage_dict["logdetK"] - (r.size/2.) * jnp.log(2*jnp.pi)

        return logL, storage_dict
    
    
    def hessian_wrapper_logL(
        self,
        p_arr: JAXArray,
        x_l: JAXArray,
        x_t: JAXArray,
        Y: JAXArray,
        storage_dict: PyTree,
        mf: Callable,
        i: int,
        make_p_dict: Callable,
        transform_fn = None
    ) -> JAXArray:
        
        p = make_p_dict(p_arr)

        grad_dict = self.grad_logL(p, x_l, x_t, Y, mf, storage_dict = storage_dict, transform_fn = transform_fn)
        
        grad_arr = ravel_pytree(grad_dict)[0]
        
        return grad_arr[i]
    
    
    def large_hessian_logL(
        self,
        p_untransf: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        Y: JAXArray,
        mf: Callable,
        storage_dict: Optional[PyTree] = {},
        transform_fn: Optional[Callable] = None,
        block_size: Optional[int] = 5,
        **kwargs: Any,
    ) -> Tuple[PyTree, PyTree]:
        
        p_arr, make_p_dict = ravel_pytree(p_untransf)
        n_par = p_arr.size
        
        hessian_wrapper_logL_lambda = lambda p_arr, ind: self.hessian_wrapper_logL(p_arr, x_l, x_t, Y, storage_dict, mf, ind,
                                                                                make_p_dict, transform_fn = transform_fn)
        hessian_vmap = vmap(grad(hessian_wrapper_logL_lambda), in_axes=(None, 0), out_axes=0)
        
    
        hessian_array = jnp.zeros((n_par, n_par))
        for i in tqdm(range(0, n_par - (n_par % block_size), block_size)):
            ind = jnp.arange(i, i+5)
            hessian_array = hessian_array.at[ind, :].set(hessian_vmap(p_arr, ind))

        if n_par % block_size != 0:
            ind = jnp.arange(n_par - (n_par % block_size), n_par)
            hessian_array = hessian_array.at[ind, :].set(hessian_vmap(p_arr, ind))
        
        hessian_dict = array_to_pytree2D(p_untransf, hessian_array)
        
        return hessian_dict, storage_dict
    

    def generate_noise(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        storage_dict: Optional[PyTree] = {},
        size: Optional[int] = 1
    ) -> JAXArray:
        
        storage_dict = self.decomp_fn(hp, x_l, x_t, storage_dict = {})
        
        z = np.random.normal(size = (storage_dict["L_cho"].shape[0], size))
        r = jnp.einsum("ij,j...->i...", storage_dict["L_cho"], z)
        R = make_mat(r, x_l.shape[-1], x_t.shape[-1])
                     
        return R
    
    
    def predict(
        self,
        p_untransf: PyTree,
        x_l: JAXArray,
        x_l_s: JAXArray,
        x_t: JAXArray,
        x_t_s: JAXArray,
        Y: JAXArray,
        mf: Callable,
        storage_dict: Optional[PyTree] = {},
        transform_fn: Optional[Callable] = None
    ) -> Tuple[JAXArray, JAXArray, JAXArray]:
        
        storage_dict = self.decomp_fn(p_untransf, x_l, x_t, storage_dict = storage_dict, transform_fn = transform_fn)
        
        if transform_fn is not None:
            p = transform_fn(p_untransf)
        else:
            p = p_untransf
            
        K_s = self.K(p, x_l, x_l_s, x_t, x_t_s, wn = False)
        K_ss = self.K(p, x_l_s, x_l_s, x_t_s, x_t_s, wn = True)

        # Generate mean function and compute residuals
        M = mf(p, x_l, x_t)
        R = Y - M
        r = make_vec(R)
        alpha = JLA.solve_triangular(storage_dict["L_cho"], r, trans = 1)
        K_inv_R = JLA.solve_triangular(storage_dict["L_cho"], alpha, trans = 0)
        K_s_K_inv_R = K_s @ K_inv_R
        
        gp_mean = M + make_mat(K_s_K_inv_R, x_l_s.shape[-1], x_t_s.shape[-1])

        sigma_diag = jnp.diag(K_ss)
        
        K_s_alpha = JLA.solve_triangular(storage_dict["L_cho"], K_s, trans = 1)
        sigma_diag -= jnp.diag(K_s_alpha.T @ K_s_alpha)
        sigma_diag = make_mat(sigma_diag, x_l_s.shape[-1], x_t_s.shape[-1])
        
        return gp_mean, sigma_diag, M
    