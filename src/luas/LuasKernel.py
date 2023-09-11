import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, value_and_grad, hessian, vmap
from jax.flatten_util import ravel_pytree
from copy import deepcopy
from tqdm import tqdm

from .kronecker_functions import kron_prod, kronecker_inv_vec
from .jax_convenience_fns import array_to_pytree2D

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)

class LuasKernel(object):
    def __init__(self, Kl = None, Kt = None, Sl = None, St = None, use_stored_results = True):
        
        self.Kl = Kl
        self.Kt = Kt
        self.Sl = Sl
        self.St = St
           
        if use_stored_results:
            self.eigen_fn = self.eigendecomp_general
        else:
            self.eigen_fn = self.eigendecomp_no_checks
        
        # JIT compile log-likelihood function
#         self.compute_logL = self.logL_transf, static_argnums=(5,))
        
        # Use JAX's grad and hessian functions on the log-likelihood in combination with JIT compilation
        self.compute_mfp_grad_logL = grad(self.compute_logL, has_aux = True)
        self.compute_mfp_value_and_grad_logL = value_and_grad(self.compute_logL, has_aux = True)
        self.compute_mfp_hessian_logL = hessian(self.compute_logL, has_aux = True)
        self.compute_hp_grad_logL = grad(self.hp_grad_logL)
        self.compute_hessian_logL1 = hessian(self.hp_grad_logL)
        self.compute_hessian_logL2 = hessian(self.compute_hessian_logL_wrapper)
    
            
    def eigendecomp_no_checks(self, hp_untransf, x_l, x_t, transform_fn = None, storage_dict = {}):
    
        if transform_fn is not None:
            hp = transform_fn(hp_untransf)
        else:
            hp = hp_untransf
    
        Sl = self.Sl(hp, x_l, x_l)
        storage_dict["lam_Sl"], storage_dict["Q_L_neg_half_Sl"] = jax.lax.cond(self.Sl.diag, eigendecomp_diag, eigendecomp_general, operand=Sl)


        St = self.St(hp, x_t, x_t)
        storage_dict["lam_St"], storage_dict["Q_L_neg_half_St"] = jax.lax.cond(self.St.diag, eigendecomp_diag, eigendecomp_general, operand=St)


        Kl = self.Kl(hp, x_l, x_l)
        Kl_tilde = storage_dict["Q_L_neg_half_Sl"].T @ Kl @ storage_dict["Q_L_neg_half_Sl"]
        storage_dict["lam_Kl_tilde"], storage_dict["W_l"] = jax.lax.cond(self.Kl.diag and self.Sl.diag, eigendecomp_diag_K_tilde, eigendecomp_general_K_tilde,
                                                       Kl_tilde, storage_dict["Q_L_neg_half_Sl"])

        Kt = self.Kt(hp, x_t, x_t)
        Kt_tilde = storage_dict["Q_L_neg_half_St"].T @ Kt @ storage_dict["Q_L_neg_half_St"]
        storage_dict["lam_Kt_tilde"], storage_dict["W_t"] = jax.lax.cond(self.Kt.diag and self.St.diag, eigendecomp_diag_K_tilde, eigendecomp_general_K_tilde,
                                                       Kt_tilde, storage_dict["Q_L_neg_half_St"])

        D = jnp.outer(storage_dict["lam_Kl_tilde"], storage_dict["lam_Kt_tilde"]) + 1.
        storage_dict["D_inv"] = jnp.reciprocal(D)

        lam_S = jnp.outer(storage_dict["lam_Sl"], storage_dict["lam_St"])
        storage_dict["logdetK"] = jnp.log(jnp.multiply(D, lam_S)).sum()

        storage_dict["hp"] = deepcopy(hp)
        
        return storage_dict

    
    def eigendecomp_general(self, hp_untransf, x_l, x_t, transform_fn = None, storage_dict = {}, rtol=1e-12, atol=1e-12):

        if transform_fn is not None:
            hp = transform_fn(hp_untransf)
        else:
            hp = hp_untransf
            
        if storage_dict:
            Sl_diff = St_diff = False

            for par in self.Sl.hp:
                Sl_diff = jax.lax.cond(jnp.allclose(hp[par], storage_dict["hp"][par], rtol = rtol, atol = atol), lambda hp: Sl_diff, lambda hp:True, hp)

            for par in self.St.hp:
                St_diff = jax.lax.cond(jnp.allclose(hp[par], storage_dict["hp"][par], rtol = rtol, atol = atol), lambda hp: St_diff, lambda hp:True, hp)

            Kl_diff = Sl_diff
            Kt_diff = St_diff

            for par in self.Kl.hp:
                Kl_diff = jax.lax.cond(jnp.allclose(hp[par], storage_dict["hp"][par], rtol = rtol, atol = atol), lambda hp: Kl_diff, lambda hp:True, hp)

            for par in self.Kt.hp:
                Kt_diff = jax.lax.cond(jnp.allclose(hp[par], storage_dict["hp"][par], rtol = rtol, atol = atol), lambda hp: Kt_diff, lambda hp:True, hp)
        else:
            Sl_diff = St_diff = Kl_diff = Kt_diff = True
            storage_dict["lam_Sl"] = jnp.zeros((x_l.size))
            storage_dict["Q_L_neg_half_Sl"] = jnp.zeros((x_l.size, x_l.size))
            storage_dict["lam_St"] = jnp.zeros((x_t.size))
            storage_dict["Q_L_neg_half_St"] = jnp.zeros((x_t.size, x_t.size))
            storage_dict["lam_Kl_tilde"] = jnp.zeros((x_l.size))
            storage_dict["W_l"] = jnp.zeros((x_l.size, x_l.size))
            storage_dict["lam_Kt_tilde"] = jnp.zeros((x_t.size))
            storage_dict["W_t"] = jnp.zeros((x_t.size, x_t.size))


        Sl = jax.lax.cond(Sl_diff, self.Sl, lambda *args: storage_dict["Q_L_neg_half_Sl"], hp, x_l, x_l)
        storage_dict["lam_Sl"], storage_dict["Q_L_neg_half_Sl"] = jax.lax.cond(Sl_diff, eigendecomp_general_test,
                                                                           lambda *args: (storage_dict["lam_Sl"], storage_dict["Q_L_neg_half_Sl"]),
                                                                            Sl, self.Sl.diag)
        St = jax.lax.cond(St_diff, self.St, lambda *args: storage_dict["Q_L_neg_half_St"], hp, x_t, x_t)
        storage_dict["lam_St"], storage_dict["Q_L_neg_half_St"] = jax.lax.cond(St_diff, eigendecomp_general_test,
                                                                           lambda *args: (storage_dict["lam_St"], storage_dict["Q_L_neg_half_St"]),
                                                                            St, self.St.diag)

        Kl = jax.lax.cond(Kl_diff, self.Kl, lambda *args: storage_dict["W_l"], hp, x_l, x_l)
        storage_dict["lam_Kl_tilde"], storage_dict["W_l"] = jax.lax.cond(Kl_diff, eigendecomp_general_K_tilde_test,
                                                                     lambda *args: (storage_dict["lam_Kl_tilde"], storage_dict["W_l"]),
                                                                     Kl, self.Kl.diag and self.Sl.diag, storage_dict["Q_L_neg_half_Sl"])

        Kt = jax.lax.cond(Kt_diff, self.Kt, lambda *args: storage_dict["W_t"], hp, x_t, x_t)
        storage_dict["lam_Kt_tilde"], storage_dict["W_t"] = jax.lax.cond(Kt_diff, eigendecomp_general_K_tilde_test,
                                                                    lambda *args: (storage_dict["lam_Kt_tilde"], storage_dict["W_t"]),
                                                                    Kt, self.Kt.diag and self.St.diag, storage_dict["Q_L_neg_half_St"])

        D = jnp.outer(storage_dict["lam_Kl_tilde"], storage_dict["lam_Kt_tilde"]) + 1.
        storage_dict["D_inv"] = jnp.reciprocal(D)

        lam_S = jnp.outer(storage_dict["lam_Sl"], storage_dict["lam_St"])
        storage_dict["logdetK"] = jnp.log(jnp.multiply(D, lam_S)).sum()

        storage_dict["hp"] = deepcopy(hp)

        return storage_dict
            
        
    def compute_logL(self, p_untransf, x_l, x_t, storage_dict, Y, mf, transform_fn = None, calc_K_inv_R = False):
        # Computes the log-likelihood
        
        if transform_fn is not None:
            p = transform_fn(p_untransf)
        else:
            p = p_untransf
            
        # Generate mean function and compute residuals
        M = mf(p, x_l, x_t)
        R = Y - M

        # Compute r.T K^-1 r
        alpha1 = kron_prod(storage_dict["W_l"].T, storage_dict["W_t"].T, R)
        alpha2 = jnp.multiply(storage_dict["D_inv"], alpha1)
        
        if calc_K_inv_R:
            K_inv_R = kron_prod(storage_dict["W_l"], storage_dict["W_t"], alpha2)
            R_K_inv_R = jnp.multiply(R, K_inv_R).sum()
        else:
            R_K_inv_R = jnp.multiply(alpha1, alpha2).sum()

        # Can make use of stored logdetK from eigendecomposition
        logL = - 0.5*R_K_inv_R- 0.5*storage_dict["logdetK"] - 0.5*Y.size*jnp.log(2*jnp.pi)

        if calc_K_inv_R:
            return logL, K_inv_R
        else:
            return logL
    
    
    def hp_grad_logL(self, p_untransf, x_l, x_t, storage_dict, transform_fn = None, K_inv_R = None, mf = None, Y = None):
        
        if transform_fn is not None:
            p = transform_fn(p_untransf)
        else:
            p = p_untransf
        
        if K_inv_R is None:
            M = mf(p, x_l, x_t)
            R = Y - M

            # Compute r.T K^-1 r
            alpha1 = kron_prod(storage_dict["W_l"].T, storage_dict["W_t"].T, R)
            alpha2 = jnp.multiply(storage_dict["D_inv"], alpha1)
            K_inv_R = kron_prod(storage_dict["W_l"], storage_dict["W_t"], alpha2)
        
        # Build kernels, necessary to do here for JAX to know how to calculate gradients
        Kl = self.Kl(p, x_l, x_l)
        Kt = self.Kt(p, x_t, x_t)
        Sl = self.Sl(p, x_l, x_l)
        St = self.St(p, x_t, x_t)
        
        K_alpha = kron_prod(Kl, Kt, K_inv_R)
        K_alpha += kron_prod(Sl, St, K_inv_R)
        
        # This transformation is used for both the r^T K^-1 r and logdetK derivatives
        W_Kl_W = storage_dict["W_l"].T @ Kl @ storage_dict["W_l"]
        W_Kt_W = storage_dict["W_t"].T @ Kt @ storage_dict["W_t"]
        W_Sl_W = storage_dict["W_l"].T @ Sl @ storage_dict["W_l"]
        W_St_W = storage_dict["W_t"].T @ St @ storage_dict["W_t"]
        
        # Diagonal of these terms is used for logdetK transformation
        Kl_diag = jnp.diag(W_Kl_W)
        Kt_diag = jnp.diag(W_Kt_W)
        Sl_diag = jnp.diag(W_Sl_W)
        St_diag = jnp.diag(W_St_W)
        
        # Computes diagonal of W.T K W for calculation of logdetK
        W_K_W_diag = jnp.outer(Kl_diag, Kt_diag) + jnp.outer(Sl_diag, St_diag)
        logdetK = jnp.multiply(storage_dict["D_inv"], W_K_W_diag).sum()
        
        return + 0.5 * jnp.multiply(K_inv_R, K_alpha).sum() - 0.5 * logdetK
    
    
    def predict(self, p_untransf, x_l, x_l_s, x_t, x_t_s, Y, mf, storage_dict = {}, transform_fn = None):
        
        storage_dict = self.eigen_fn(p_untransf, x_l, x_t, storage_dict = storage_dict, transform_fn = transform_fn)
        
        if transform_fn is not None:
            p = transform_fn(p_untransf)
        else:
            p = p_untransf
            
        Kl_s = self.Kl(p, x_l, x_l_s, wn = False)
        Kt_s = self.Kt(p, x_t, x_t_s, wn = False)
        Sl_s = self.Sl(p, x_l, x_l_s, wn = False)
        St_s = self.St(p, x_t, x_t_s, wn = False)

        
        Kl_ss = self.Kl(p, x_l_s, x_l_s, wn = True)
        Kt_ss = self.Kt(p, x_t_s, x_t_s, wn = True)
        Sl_ss = self.Sl(p, x_l_s, x_l_s, wn = True)
        St_ss = self.St(p, x_t_s, x_t_s, wn = True)

        # Generate mean function and compute residuals
        M = mf(p, x_l, x_t)
        R = Y - M
        alpha = kronecker_inv_vec(R, storage_dict)

        gp_mean = M
        gp_mean += kron_prod(Kl_s.T, Kt_s.T, alpha)
        gp_mean += kron_prod(Sl_s.T, St_s.T, alpha)

        Y_l = Kl_s.T @ storage_dict["W_l"]
        Y_t = Kt_s.T @ storage_dict["W_t"]
        Z_l = Sl_s.T @ storage_dict["W_l"]
        Z_t = St_s.T @ storage_dict["W_t"]

        sigma_diag = jnp.outer(jnp.diag(Kl_ss), jnp.diag(Kt_ss))
        sigma_diag += jnp.outer(jnp.diag(Sl_ss), jnp.diag(St_ss))

        sigma_diag -= kron_prod(Y_l**2, Y_t**2, storage_dict["D_inv"])
        sigma_diag -= kron_prod(Z_l**2, Z_t**2, storage_dict["D_inv"])
        sigma_diag -= 2*kron_prod(Y_l * Z_l, Y_t * Z_t, storage_dict["D_inv"])

        
        return gp_mean, sigma_diag, M
    
    
    def logL(self, p_untransf, x_l, x_t, Y, mf, storage_dict = {}, transform_fn = None, calc_K_inv_R = False):
        
        storage_dict = self.eigen_fn(p_untransf, x_l, x_t, storage_dict = storage_dict, transform_fn = transform_fn)
        
        return self.compute_logL(p_untransf, x_l, x_t, storage_dict, Y, mf, transform_fn = transform_fn, calc_K_inv_R = calc_K_inv_R), storage_dict
        
    
    def grad_logL(self, p_untransf, x_l, x_t, Y, mf, storage_dict = {}, fit_mfp = [], transform_fn = None):
        
        storage_dict = self.eigen_fn(p_untransf, x_l, x_t, storage_dict = storage_dict, transform_fn = transform_fn)
        
        # Calculate dictionary of gradients of log-likelihood
        mfp_grad_dict, K_inv_R = self.compute_mfp_grad_logL(p_untransf, x_l, x_t, storage_dict, Y, mf,
                                                            calc_K_inv_R = True, transform_fn = transform_fn)
        grad_dict = self.compute_hp_grad_logL(p_untransf, x_l, x_t, storage_dict,
                                              K_inv_R = K_inv_R, transform_fn = transform_fn)
        
        # Combine
        grad_dict.update({k: mfp_grad_dict[k] for k in fit_mfp})
        
        return grad_dict, storage_dict
    
    
    def value_and_grad_logL(self, p_untransf, x_l, x_t, Y, mf, storage_dict = {}, fit_mfp = [], transform_fn = None):
        
        storage_dict = self.eigen_fn(p_untransf, x_l, x_t, storage_dict = storage_dict, transform_fn = transform_fn)
        
        # Calculate dictionary of gradients of log-likelihood
        (logL, K_inv_R), mfp_grad_dict = self.compute_mfp_value_and_grad_logL(p_untransf, x_l, x_t, storage_dict, Y, mf,
                                                                              calc_K_inv_R = True, transform_fn = transform_fn)
        grad_dict = self.compute_hp_grad_logL(p_untransf, x_l, x_t, storage_dict,
                                              K_inv_R = K_inv_R, transform_fn = transform_fn)
        
        # Combine
        grad_dict.update({k: mfp_grad_dict[k] for k in fit_mfp})
        
        return (logL, storage_dict), grad_dict
    
   
    def hessian_logL(self, p_untransf, x_l, x_t, Y, mf, storage_dict = {}, fit_mfp = [], fit_hp = [], transform_fn = None):
    
        # Calculate any necessary eigendecompositions
        storage_dict = self.eigen_fn(p_untransf, x_l, x_t, storage_dict = storage_dict, transform_fn = transform_fn)
        
        mfp_hessian_dict, K_inv_R = self.compute_mfp_hessian_logL(p_untransf, x_l, x_t, storage_dict, Y, mf, transform_fn = transform_fn, calc_K_inv_R = True)
        
        # Calculate nested dictionary of hessian of log-likelihood
        hessian_dict = self.compute_hessian_logL1(p_untransf, x_l, x_t, storage_dict,
                                                  mf = mf, Y = Y, transform_fn = transform_fn)
        for i in fit_mfp:
            for j in fit_mfp:
                hessian_dict[i][j] = mfp_hessian_dict[i][j]
                
        if fit_hp:
            hp = {k:p_untransf[k] for k in fit_hp}
            p_cross = {"p1":hp, "p2":deepcopy(hp)}
            hessian_dict2 = self.compute_hessian_logL2(p_cross, p_untransf, x_l, x_t, Y, storage_dict, mf, transform_fn = transform_fn)
        
            for i in fit_hp:
                for j in fit_hp:
                    hessian_dict[i][j] += hessian_dict2["p1"][i]["p2"][j]
                 
        return hessian_dict, storage_dict
    
    
    def compute_hessian_logL_wrapper(self, p_untransf, *args, **kwargs):

        return self.compute_hessian_logL(p_untransf["p1"], p_untransf["p2"], *args, **kwargs)

    
    def compute_hessian_logL(self, p1_untransf, p2_untransf, mfp_untransf, x_l, x_t, Y, storage_dict, mf, transform_fn = None):
        
        if transform_fn is not None:
            p1 = transform_fn(p1_untransf)
            p2 = transform_fn(p2_untransf)
            mfp = transform_fn(mfp_untransf)
        else:
            p1 = p1_untransf
            p2 = p2_untransf
            mfp = mfp_untransf

        # Generate mean function and compute residuals
        M = mf(mfp, x_l, x_t)
        R = Y - M

        alpha1 = kron_prod(storage_dict["W_l"].T, storage_dict["W_t"].T, R)
        alpha2 = jnp.multiply(storage_dict["D_inv"], alpha1)

        Kl1 = self.Kl(p1, x_l, x_l)
        Kt1 = self.Kt(p1, x_t, x_t)
        Sl1 = self.Sl(p1, x_l, x_l)
        St1 = self.St(p1, x_t, x_t)

        W_Kl_W1 = storage_dict["W_l"].T @ Kl1 @ storage_dict["W_l"]
        W_Kt_W1 = storage_dict["W_t"].T @ Kt1 @ storage_dict["W_t"]
        W_Sl_W1 = storage_dict["W_l"].T @ Sl1 @ storage_dict["W_l"]
        W_St_W1 = storage_dict["W_t"].T @ St1 @ storage_dict["W_t"]

        Kl2 = self.Kl(p2, x_l, x_l)
        Kt2 = self.Kt(p2, x_t, x_t)
        Sl2 = self.Sl(p2, x_l, x_l)
        St2 = self.St(p2, x_t, x_t)

        W_Kl_W2 = storage_dict["W_l"].T @ Kl2 @ storage_dict["W_l"]
        W_Kt_W2 = storage_dict["W_t"].T @ Kt2 @ storage_dict["W_t"]
        W_Sl_W2 = storage_dict["W_l"].T @ Sl2 @ storage_dict["W_l"]
        W_St_W2 = storage_dict["W_t"].T @ St2 @ storage_dict["W_t"]

        K_alpha1 = kron_prod(W_Kl_W1, W_Kt_W1, alpha2)
        K_alpha1 += kron_prod(W_Sl_W1, W_St_W1, alpha2)

        D_K_alpha1 = jnp.multiply(storage_dict["D_inv"], K_alpha1)


        K_alpha2 = kron_prod(W_Kl_W2, W_Kt_W2, alpha2)
        K_alpha2 += kron_prod(W_Sl_W2, W_St_W2, alpha2)

        rKr = jnp.multiply(K_alpha2, D_K_alpha1).sum()

        K_diag = kron_prod(W_Kl_W1 * W_Kl_W2.T, W_Kt_W1 * W_Kt_W2.T, storage_dict["D_inv"])
        K_diag += kron_prod(W_Sl_W1 * W_Sl_W2.T, W_St_W1 * W_St_W2.T, storage_dict["D_inv"])
        K_diag += kron_prod(W_Kl_W1 * W_Sl_W2.T, W_Kt_W1 * W_St_W2.T, storage_dict["D_inv"])
        K_diag += kron_prod(W_Sl_W1 * W_Kl_W2.T, W_St_W1 * W_Kt_W2.T, storage_dict["D_inv"])

        K_logdet = jnp.multiply(storage_dict["D_inv"], K_diag).sum()

        return - rKr + 0.5 * K_logdet
    
    
    def hessian_wrapper_logL(self, p_arr1, p_arr2, x_l, x_t, Y, storage_dict, mf, i, make_p_dict = None, transform_fn = None, fit_mfp = [], fit_hp = []):
        
        p1 = make_p_dict(p_arr1)
        p2 = make_p_dict(p_arr2)
        
        mfp = deepcopy(p2)

        # Calculate dictionary of gradients of log-likelihood
        mfp_grad_dict, K_inv_R = self.compute_mfp_grad_logL(p1, x_l, x_t, storage_dict, Y, mf,
                                                   calc_K_inv_R = True, transform_fn = transform_fn)
        grad_dict = self.compute_hp_grad_logL(p1, x_l, x_t, storage_dict,
                                              mf = mf, Y = Y, transform_fn = transform_fn)
        
        # Combine
        grad_dict.update({k: mfp_grad_dict[k] for k in fit_mfp})
        
        grad_dict2 = grad(self.compute_hessian_logL, argnums = 1)(p1, p2, mfp, x_l, x_t,
                                                                Y, storage_dict, mf,
                                                                transform_fn = transform_fn)
        
        for par in fit_hp:
            grad_dict[par] += grad_dict2[par]
        
        grad_arr = ravel_pytree(grad_dict)[0]
        
        return grad_arr[i]
    
    
    def large_hessian_logL(self, p_untransf, x_l, x_t, Y, mf, storage_dict = {}, fit_mfp = [], fit_hp = [],
                           transform_fn = None, block_size = 5, make_p_dict = None):
        
        p_arr = ravel_pytree(p_untransf)[0]
        n_par = p_arr.size
        
        # Calculate any necessary eigendecompositions
        storage_dict = self.eigen_fn(p_untransf, x_l, x_t, storage_dict = storage_dict, transform_fn = transform_fn)
        
        hessian_wrapper_logL_lambda = lambda p1, ind: self.hessian_wrapper_logL(p1, p_arr, x_l, x_t, Y, storage_dict, mf, ind,
                                                                                make_p_dict = make_p_dict, transform_fn = transform_fn,
                                                                                fit_mfp = fit_mfp, fit_hp = fit_hp)
        hessian_vmap = vmap(grad(hessian_wrapper_logL_lambda), in_axes=(None, 0), out_axes=0)
        
        
        hessian_array = jnp.zeros((n_par, n_par))
        for i in tqdm(range(0, n_par - (n_par % block_size), block_size)):
            ind = jnp.arange(i, i+5)
            hessian_array = hessian_array.at[ind, :].set(hessian_vmap(p_arr, ind))

        if n_par % block_size != 0:
            ind = jnp.arange(n_par - (n_par % block_size), n_par)
            hessian_array = hessian_array.at[ind, :].set(hessian_vmap(p_arr, ind))
        
        hessian_dict = array_to_pytree2D(p_untransf, hessian_array)
        
        for mfp_par in fit_mfp:
            for hp_par in fit_hp:
                hessian_dict[mfp_par][hp_par] = hessian_dict[hp_par][mfp_par].T
        
        return hessian_dict, storage_dict

            
    def generate_noise(self, hp_untransf, x_l, x_t, size = 1, stable_const = 1e-6, transform_fn = None):
        
        if transform_fn is not None:
            hp = transform_fn(hp_untransf)
        else:
            hp = hp_untransf
        
        Kl = self.Kl(hp, x_l, x_l)
        Kt = self.Kt(hp, x_t, x_t)
        Sl = self.Sl(hp, x_l, x_l)
        St = self.St(hp, x_t, x_t)
    
        Lam_Kl, Q_Kl = jnp.linalg.eigh(Kl)
        Lam_Kt, Q_Kt = jnp.linalg.eigh(Kt)

        Lam_mat_K = jnp.outer(Lam_Kl, Lam_Kt) + stable_const**2
        Lam_mat_sqrt_K = jnp.sqrt(Lam_mat_K)

        Lam_Sl, Q_Sl = jnp.linalg.eigh(Sl)
        Lam_St, Q_St = jnp.linalg.eigh(St)

        Lam_mat_S = jnp.outer(Lam_Sl, Lam_St) - stable_const**2
        Lam_mat_sqrt_S = jnp.sqrt(Lam_mat_S)

        z = np.random.normal(size = (Kl.shape[0], Kt.shape[0], 2, size))

        if size == 1:
            Lam_z1 = jnp.multiply(Lam_mat_sqrt_K, z[:, :, 0, 0])
            R = kron_prod(Q_Kl, Q_Kt, Lam_z1)

            Lam_z2 = jnp.multiply(Lam_mat_sqrt_S, z[:, :, 1, 0])
            R += kron_prod(Q_Sl, Q_St, Lam_z2)

        else:
            R = jnp.zeros((Kl.shape[0], Kt.shape[0], size))
            for i in range(size):
                Lam_z1 = jnp.multiply(Lam_mat_sqrt_K, z[:, :, 0, i])
                R[:, :, i] = kron_prod(Q_Kl, Q_Kt, Lam_z1)

                Lam_z2 = jnp.multiply(Lam_mat_sqrt_S, z[:, :, 1, i])
                R[:, :, i] += kron_prod(Q_Sl, Q_St, Lam_z2)

        return R
 

    
def eigendecomp_diag(K):
    lam_K = jnp.diag(K)
    Q_L_neg_half_K = jnp.diag(jnp.sqrt(jnp.reciprocal(lam_K)))
    
    return lam_K, Q_L_neg_half_K


def eigendecomp_general(K):
    lam_K, Q_K = jnp.linalg.eigh(K)
    Q_L_neg_half_K = Q_K @ jnp.diag(jnp.sqrt(jnp.reciprocal(lam_K)))
    
    return lam_K, Q_L_neg_half_K


def eigendecomp_diag_K_tilde(K_tilde, W):
    lam_K_tilde = jnp.diag(K_tilde)
    
    return lam_K_tilde, W


def eigendecomp_general_K_tilde(K_tilde, W):
    lam_K_tilde, Q_K_tilde = jnp.linalg.eigh(K_tilde)
    W_K_tilde = W @ Q_K_tilde
    
    return lam_K_tilde, W_K_tilde


def eigendecomp_general_test(S, diag):
    return jax.lax.cond(diag, eigendecomp_diag, eigendecomp_general, operand=S)


def eigendecomp_general_K_tilde_test(K, diag, Q_L_neg_half_S):
    K_tilde = Q_L_neg_half_S.T @ K @ Q_L_neg_half_S
    
    return jax.lax.cond(diag, eigendecomp_diag_K_tilde, eigendecomp_general_K_tilde,
                        K_tilde, Q_L_neg_half_S)
