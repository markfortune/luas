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
from .jax_convenience_fns import array_to_pytree_2D


__all__ = [
    "LuasKernel",
    "eigendecomp_diag_S",
    "eigendecomp_general_S",
    "eigendecomp_diag_K_tilde",
    "eigendecomp_general_K_tilde",
    "eigendecomp_S",
    "eigendecomp_K_tilde",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


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


class LuasKernel(Kernel):
    def __init__(
        self,
        Kl_fn: Callable,
        Kt_fn: Callable,
        Sl_fn: Callable,
        St_fn: Callable,
        use_stored_results: Optional[bool] = True,
    ):
        
        self.Kl = Kl_fn
        self.Kt = Kt_fn
        self.Sl = Sl_fn
        self.St = St_fn
           
        if use_stored_results:
            self.decomp_fn = self.eigendecomp_use_stored_results
        else:
            self.decomp_fn = self.eigendecomp_no_stored_results

        decomp_dict = {}
        for fn in [self.Sl, self.St, self.Kl, self.Kt]:
            if hasattr(fn, "decomp"):
                if fn.decomp == "diag":
                    fn.decomp = diag_eigendecomp
                    decomp_dict[fn] = "diag"
            else:
                fn.decomp = jnp.linalg.eigh
                decomp_dict[fn] = "general"
                
        if decomp_dict[self.Kl] == "diag" and not decomp_dict[self.Sl] == "diag":
            raise Warning("The transformation of Kl is set to be diagonal but the matrix Sl is not set to diagonal. This may be possible for example if Kl is a scalar times the identity matrix or Kl shares the same eigenvectors as Sl but it is not true if Kl is any general diagonal matrix. Alternatively perhaps Sl is also diagonal and you forgot to add Sl.decomp = 'diag'. Be careful to ensure the transformation of Kl is diagonal or else log likelihood values will be incorrect!")
        if decomp_dict[self.Kt] == "diag" and not decomp_dict[self.St] == "diag":
            raise Warning("The transformation of Kt is set to be diagonal but the matrix St is not set to diagonal. This may be possible for example if Kt is a scalar times the identity matrix or Kt shares the same eigenvectors as St but it is not true if Kt is any general diagonal matrix. Alternatively perhaps St is also diagonal and you forgot to add St.decomp = 'diag'. Be careful to ensure the transformation of Kt is diagonal or else log likelihood values will be incorrect!")

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

        storage_dict = deepcopy(storage_dict)
        
        Sl = self.Sl(hp, x_l, x_l)
        St = self.St(hp, x_t, x_t)
        Kl = self.Kl(hp, x_l, x_l)
        Kt = self.Kt(hp, x_t, x_t)

        N_l = Sl.shape[0]
        N_t = St.shape[0]

        
        if storage_dict: 
            Sl_diff = jax.lax.cond(jnp.allclose(Sl, storage_dict["Sl"], rtol = rtol, atol = atol), lambda hp: False, lambda hp: True, hp)
            St_diff = jax.lax.cond(jnp.allclose(St, storage_dict["St"], rtol = rtol, atol = atol), lambda hp: False, lambda hp: True, hp)
            Kl_diff = jax.lax.cond(jnp.allclose(Kl, storage_dict["Kl"], rtol = rtol, atol = atol), lambda hp: Sl_diff, lambda hp: True, hp)
            Kt_diff = jax.lax.cond(jnp.allclose(Kt, storage_dict["Kt"], rtol = rtol, atol = atol), lambda hp: St_diff, lambda hp: True, hp)
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


        storage_dict["lam_Sl"], storage_dict["Sl_inv_sqrt"] = jax.lax.cond(Sl_diff, self.Sl_decomp_fn,
                                                                           lambda *args: (storage_dict["lam_Sl"], storage_dict["Sl_inv_sqrt"]), Sl)
        
        storage_dict["lam_St"], storage_dict["St_inv_sqrt"] = jax.lax.cond(St_diff, self.St_decomp_fn,
                                                                           lambda *args: (storage_dict["lam_St"], storage_dict["St_inv_sqrt"]), St)

        storage_dict["lam_Kl_tilde"], storage_dict["W_l"] = jax.lax.cond(Kl_diff, self.Kl_tilde_decomp_fn,
                                                                         lambda *args: (storage_dict["lam_Kl_tilde"], storage_dict["W_l"]), Kl, storage_dict["Sl_inv_sqrt"])

        storage_dict["lam_Kt_tilde"], storage_dict["W_t"] = jax.lax.cond(Kt_diff, self.Kt_tilde_decomp_fn,
                                                                         lambda *args: (storage_dict["lam_Kt_tilde"], storage_dict["W_t"]), Kt, storage_dict["St_inv_sqrt"])

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

        storage_dict = self.decomp_fn(hp, x_l, x_t, storage_dict = storage_dict)
        
        rKr = r_K_inv_r(R, storage_dict)
        logdetK = logdetK_calc(storage_dict)
        logL = -0.5 * rKr - 0.5 * logdetK  - 0.5 * R.size * jnp.log(2*jnp.pi)

        return  logL, storage_dict

    
    def logL_hessianable(self, hp, x_l, x_t, R, storage_dict):

        storage_dict = self.decomp_fn(hp, x_l, x_t, storage_dict = storage_dict)
        
        rKr = r_K_inv_r(R, storage_dict)
        logdetK = logdetK_calc_hessianable(storage_dict)
        logL =  -0.5 * rKr - 0.5 * logdetK  - 0.5 * R.size * jnp.log(2*jnp.pi)

        return  logL, storage_dict
        
    
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
        
        Kl_s = self.Kl(hp, x_l, x_l_s, wn = False)
        Kt_s = self.Kt(hp, x_t, x_t_s, wn = False)
        Sl_s = self.Sl(hp, x_l, x_l_s, wn = False)
        St_s = self.St(hp, x_t, x_t_s, wn = False)
        
        Kl_ss = self.Kl(hp, x_l_s, x_l_s, wn = wn)
        Kt_ss = self.Kt(hp, x_t_s, x_t_s, wn = wn)
        Sl_ss = self.Sl(hp, x_l_s, x_l_s, wn = wn)
        St_ss = self.St(hp, x_t_s, x_t_s, wn = wn)

        alpha = K_inv_vec(R, storage_dict)

        gp_mean = M_s + kron_prod(Kl_s.T, Kt_s.T, alpha) + kron_prod(Sl_s.T, St_s.T, alpha)

        Y_l = Kl_s.T @ storage_dict["W_l"]
        Y_t = Kt_s.T @ storage_dict["W_t"]
        Z_l = Sl_s.T @ storage_dict["W_l"]
        Z_t = St_s.T @ storage_dict["W_t"]

        sigma_diag = jnp.outer(jnp.diag(Kl_ss), jnp.diag(Kt_ss))
        sigma_diag += jnp.outer(jnp.diag(Sl_ss), jnp.diag(St_ss))

        sigma_diag -= kron_prod(Y_l**2, Y_t**2, storage_dict["D_inv"])
        sigma_diag -= kron_prod(Z_l**2, Z_t**2, storage_dict["D_inv"])
        sigma_diag -= 2*kron_prod(Y_l * Z_l, Y_t * Z_t, storage_dict["D_inv"])
        
        return gp_mean, sigma_diag


    def generate_noise(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        size: Optional[int] = 1,
        stable_const: Optional[Scalar] = 1e-6,
    ) -> JAXArray:
        
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


    def K(self, hp, x_l, x_l_s, x_t, x_t_s, **kwargs):

        Kl = self.Kl(hp, x_l, x_l_s, **kwargs)
        Kt = self.Kt(hp, x_t, x_t_s, **kwargs)
        Sl = self.Sl(hp, x_l, x_l_s, **kwargs)
        St = self.St(hp, x_t, x_t_s, **kwargs)
        
        K = jnp.kron(Kl, Kt) + jnp.kron(Sl, St)
        
        return K
