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
from .kronecker_fns import make_vec, make_mat
from .jax_convenience_fns import array_to_pytree_2D

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
            **kwargs,
        ) -> JAXArray:
        
            K = jnp.zeros((x_l1.shape[-1]*x_t1.shape[-1], x_l2.shape[-1]*x_t2.shape[-1]))
            for i in range(len(Kl_fns)):
                Kl = Kl_fns[i](hp, x_l1, x_l2, **kwargs)
                Kt = Kt_fns[i](hp, x_t1, x_t2, **kwargs)
                K += jnp.kron(Kl, Kt)

            return K
        
        return K_kron
            
        
    def cholesky_decomp_no_stored_results(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray, 
        storage_dict: Optional[PyTree] = {},
    ) -> PyTree:
    
        K = self.K(hp, x_l, x_l, x_t, x_t)
        
        storage_dict["L_cho"] = JLA.cholesky(K)
        storage_dict["logdetK"] = 2*jnp.log(jnp.diag(storage_dict["L_cho"])).sum()
        
        return storage_dict

        
    def logL(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        R: JAXArray,
        storage_dict: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        
        storage_dict = self.decomp_fn(hp, x_l, x_t, storage_dict = storage_dict)
            
        r = make_vec(R)
        alpha = JLA.solve_triangular(storage_dict["L_cho"], r, trans = 1)

        logL = - 0.5 *  jnp.sum(jnp.square(alpha)) - 0.5 * storage_dict["logdetK"] - (r.size/2.) * jnp.log(2*jnp.pi)

        return logL, storage_dict


    def logL_hessianable(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        R: JAXArray,
        storage_dict: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        
        storage_dict = self.decomp_fn(hp, x_l, x_t, storage_dict = storage_dict)
            
        r = make_vec(R)
        alpha = JLA.solve_triangular(storage_dict["L_cho"], r, trans = 1)

        logL = - 0.5 *  jnp.sum(jnp.square(alpha)) - 0.5 * storage_dict["logdetK"] - (r.size/2.) * jnp.log(2*jnp.pi)

        return logL, storage_dict


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
        hp: PyTree,
        x_l: JAXArray,
        x_l_s: JAXArray,
        x_t: JAXArray,
        x_t_s: JAXArray,
        R: JAXArray,
        M_s: JAXArray,
        storage_dict: Optional[PyTree] = {},
        wn = True,
    ) -> Tuple[JAXArray, JAXArray, JAXArray]:
        
        storage_dict = self.decomp_fn(hp, x_l, x_t, storage_dict = storage_dict)
            
        K_s = self.K(hp, x_l, x_l_s, x_t, x_t_s, wn = False)
        K_ss = self.K(hp, x_l_s, x_l_s, x_t_s, x_t_s, wn = wn)

        # Generate mean function and compute residuals
        r = make_vec(R)
        alpha = JLA.solve_triangular(storage_dict["L_cho"], r, trans = 1)
        K_inv_R = JLA.solve_triangular(storage_dict["L_cho"], alpha, trans = 0)
        K_s_K_inv_R = K_s @ K_inv_R

        gp_mean = M_s + make_mat(K_s_K_inv_R, x_l_s.shape[-1], x_t_s.shape[-1])

        sigma_diag = jnp.diag(K_ss)
        
        K_s_alpha = JLA.solve_triangular(storage_dict["L_cho"], K_s, trans = 1)
        sigma_diag -= jnp.diag(K_s_alpha.T @ K_s_alpha)
        sigma_diag = make_mat(sigma_diag, x_l_s.shape[-1], x_t_s.shape[-1])
        
        return gp_mean, sigma_diag
    