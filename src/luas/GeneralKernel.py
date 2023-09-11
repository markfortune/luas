import numpy as np
from copy import deepcopy
from tqdm import tqdm
import jax
from jax import grad, value_and_grad, hessian, vmap
import jax.numpy as jnp
import jax.scipy.linalg as JLA
from jax.flatten_util import ravel_pytree

from .kronecker_functions import kron_prod, make_vec, make_mat
from .jax_convenience_fns import array_to_pytree2D

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)

class GeneralKernel(object):
    def __init__(self, K = None, Kl_fns = None, Kt_fns = None):
        
        if K is None:
            self.K = self.build_kronecker_K(Kl_fns, Kt_fns)
        else:
            self.K = K
        self.eigen_fn = self.eigendecomp_no_checks
        
        self.grad_logL = grad(self.logL, has_aux = True)
        self.value_and_grad_logL = value_and_grad(self.logL, has_aux = True)
        self.hessian_logL = hessian(self.logL, has_aux = True)
    
    
    def build_kronecker_K(self, Kl_fns, Kt_fns):
    
        def K_kron(hp, x_l1, x_l2, x_t1, x_t2, wn = True):
        
            K = jnp.zeros((x_l1.shape[-1]*x_t1.shape[-1], x_l2.shape[-1]*x_t2.shape[-1]))
            for i in range(len(Kl_fns)):
                Kl = Kl_fns[i](hp, x_l1, x_l2, wn = wn)
                Kt = Kt_fns[i](hp, x_t1, x_t2, wn = wn)
                K += jnp.kron(Kl, Kt)

            return K
        
        return K_kron
            
        
    def eigendecomp_no_checks(self, hp_untransf, x_l, x_t, storage_dict = {}, transform_fn = None):
        
        if transform_fn is not None:
            hp = transform_fn(hp_untransf)
        else:
            hp = hp_untransf
    
        K = self.K(hp, x_l, x_l, x_t, x_t)
        
        storage_dict["L_cho"] = JLA.cholesky(K)
        storage_dict["logdetK"] = 2*jnp.log(jnp.diag(storage_dict["L_cho"])).sum()
        storage_dict["hp"] = deepcopy(hp)
        
        return storage_dict

    
    def eigendecomp_general(self, hp_untransf, x_l, x_t, storage_dict = {}, transform_fn = None, rtol=1e-12, atol=1e-12):
        
        if transform_fn is not None:
            hp = transform_fn(hp_untransf)
        else:
            hp = hp_untransf
        
        K_diff = False
        if storage_dict:
            for par in self.K.hp:
                K_diff = jax.lax.cond(jnp.allclose(hp[par], storage_dict["hp"][par], rtol = rtol, atol = atol), lambda hp: K_diff, lambda hp:True, hp)
        else:
            K_diff = True

        storage_dict = jax.lax.cond(K_diff, self.eigendecomp_no_checks, lambda *args, **kwargs: storage_dict, hp, x_l, x_t, storage_dict = {})

        return storage_dict
            
        
    def logL(self, p_untransf, x_l, x_t, Y, mf, storage_dict = {}, transform_fn = None, fit_mfp = None, fit_hp = None):
        
        storage_dict = self.eigen_fn(p_untransf, x_l, x_t, storage_dict = storage_dict, transform_fn = transform_fn)
        
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
    
    
    def hessian_wrapper_logL(self, p_arr, x_l, x_t, Y, storage_dict, mf, i, make_p_dict = None, transform_fn = None):
        
        p = make_p_dict(p_arr)

        grad_dict = self.grad_logL(p, x_l, x_t, Y, mf, storage_dict = storage_dict, transform_fn = transform_fn)
        
        grad_arr = ravel_pytree(grad_dict)[0]
        
        return grad_arr[i]
    
    
    def large_hessian_logL(self, p_untransf, x_l, x_t, Y, mf, storage_dict = {}, fit_mfp = [], fit_hp = [],
                           transform_fn = None, block_size = 5, make_p_dict = None):
        
        p_arr = ravel_pytree(p_untransf)[0]
        n_par = p_arr.size
        
        hessian_wrapper_logL_lambda = lambda p_arr, ind: self.hessian_wrapper_logL(p_arr, x_l, x_t, Y, storage_dict, mf, ind,
                                                                                make_p_dict = make_p_dict, transform_fn = transform_fn)
        hessian_vmap = vmap(grad(hessian_wrapper_logL_lambda), in_axes=(None, 0), out_axes=0)
        
    
        hessian_array = np.zeros((n_par, n_par))
        for i in tqdm(range(0, n_par - (n_par % block_size), block_size)):
            ind = jnp.arange(i, i+5)
            hessian_array[ind, :] = hessian_vmap(p_arr, ind)

        if n_par % block_size != 0:
            ind = jnp.arange(n_par - (n_par % block_size), n_par)
            hessian_array[ind, :] = hessian_vmap(p_arr, ind)
        
        hessian_dict = array_to_pytree2D(p_untransf, hessian_array)
        
        return hessian_dict, storage_dict
    

    def generate_noise(self, hp, x_l, x_t, storage_dict = {}, size = 1):
        
        storage_dict = self.eigen_fn(hp, x_l, x_t, storage_dict = {})
        
        z = np.random.normal(size = (storage_dict["L_cho"].shape[0], size))

        r = jnp.einsum("ij,j...->i...", storage_dict["L_cho"], z)
        
        R = make_mat(r, x_l.shape[-1], x_t.shape[-1])
                     
        return R
    
    
    def predict(self, p_untransf, x_l, x_l_s, x_t, x_t_s, Y, mf, storage_dict = {}, transform_fn = None):
        
        storage_dict = self.eigen_fn(p_untransf, x_l, x_t, storage_dict = storage_dict, transform_fn = transform_fn)
        
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
    