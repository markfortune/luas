import numpy as np
import jax
import jax.numpy as jnp
from copy import deepcopy


def make_vec(R):
    return R.ravel("C")

def make_mat(r, N_l, N_t):
    return r.reshape((N_l, N_t))

def kron_prod(A, B, R):
    return A @ R @ B.T


def eigendecomp_rakitsch_general(hp, x_l, x_t, kf, eigen_dict = {}, rtol=1e-12, atol=1e-12):
    # Uses control flow based on parameter values so cannot JIT compile
    
    Kl_diff = Kt_diff = Sl_diff = St_diff = False
    
    if eigen_dict == {}:
        Kl_diff = Kt_diff = Sl_diff = St_diff = True
    else:
        for par in kf.Kl.hp:
            if not jnp.allclose(hp[par], eigen_dict["hp"][par], rtol = rtol, atol = atol):
                Kl_diff = True
        for par in kf.Kt.hp:
            if not jnp.allclose(hp[par], eigen_dict["hp"][par], rtol = rtol, atol = atol):
                Kt_diff = True
        for par in kf.Sl.hp:
            if not jnp.allclose(hp[par], eigen_dict["hp"][par], rtol = rtol, atol = atol):
                Sl_diff = True
        for par in kf.St.hp:
            if not jnp.allclose(hp[par], eigen_dict["hp"][par], rtol = rtol, atol = atol):
                St_diff = True
                
    if (not Kl_diff) and (not Kt_diff) and (not Sl_diff) and (not St_diff):
        return eigen_dict
    
    
    if Sl_diff:
        Sl = kf.Sl(hp, x_l, x_l)
        if kf.Sl.diag:
            eigen_dict["lam_Sl"] = jnp.diag(Sl)
            eigen_dict["Q_L_neg_half_Sl"] = jnp.diag(jnp.sqrt(jnp.reciprocal(eigen_dict["lam_Sl"])))
        else:
            eigen_dict["lam_Sl"], Q_Sl = jnp.linalg.eigh(Sl)
            eigen_dict["Q_L_neg_half_Sl"] = Q_Sl @ jnp.diag(jnp.sqrt(jnp.reciprocal(eigen_dict["lam_Sl"])))
       
    
    if St_diff:
        St = kf.St(hp, x_t, x_t)
        if kf.St.diag:
            eigen_dict["lam_St"] = jnp.diag(St)
            eigen_dict["Q_L_neg_half_St"] = jnp.diag(jnp.sqrt(jnp.reciprocal(eigen_dict["lam_St"])))
        else:
            eigen_dict["lam_St"], Q_St = jnp.linalg.eigh(St)
            eigen_dict["Q_L_neg_half_St"] = Q_St @ jnp.diag(jnp.sqrt(jnp.reciprocal(eigen_dict["lam_St"])))
    
    
    if Kl_diff or Sl_diff:
        Kl = kf.Kl(hp, x_l, x_l)
        Kl_tilde = eigen_dict["Q_L_neg_half_Sl"].T @ Kl @ eigen_dict["Q_L_neg_half_Sl"]
        
        if kf.Kl.diag and kf.Sl.diag:
            eigen_dict["lam_Kl_tilde"] = jnp.diag(Kl_tilde)
            eigen_dict["W_l"] = eigen_dict["Q_L_neg_half_Sl"]
        else:
            eigen_dict["lam_Kl_tilde"], Q_Kl_tilde = jnp.linalg.eigh(Kl_tilde)
            eigen_dict["W_l"] = eigen_dict["Q_L_neg_half_Sl"] @ Q_Kl_tilde
    
    
    if Kt_diff or St_diff:
        Kt = kf.Kt(hp, x_t, x_t)
        Kt_tilde = eigen_dict["Q_L_neg_half_St"].T @ Kt @ eigen_dict["Q_L_neg_half_St"]
        
        if kf.Kt.diag and kf.St.diag:
            eigen_dict["lam_Kt_tilde"] = jnp.diag(Kt_tilde)
            eigen_dict["W_t"] = eigen_dict["Q_L_neg_half_St"]
        else:
            eigen_dict["lam_Kt_tilde"], Q_Kt_tilde = jnp.linalg.eigh(Kt_tilde)
            eigen_dict["W_t"] = eigen_dict["Q_L_neg_half_St"] @ Q_Kt_tilde
    
    D = jnp.outer(eigen_dict["lam_Kl_tilde"], eigen_dict["lam_Kt_tilde"]) + 1.
    eigen_dict["D_inv"] = jnp.reciprocal(D)
    
    lam_S = jnp.outer(eigen_dict["lam_Sl"], eigen_dict["lam_St"])
    eigen_dict["logdetK"] = jnp.log(jnp.multiply(D, lam_S)).sum()
    
    eigen_dict["hp"] = deepcopy(hp)
    
    return eigen_dict


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



def eigendecomp_no_checks_jittable(hp, x_l, x_t, kf, eigen_dict = {}):
    
    Sl = kf.Sl(hp, x_l, x_l)
    lam_Sl, Q_L_neg_half_Sl = jax.lax.cond(kf.Sl.diag, eigendecomp_diag, eigendecomp_general, operand=Sl)


    St = kf.St(hp, x_t, x_t)
    lam_St, Q_L_neg_half_St = jax.lax.cond(kf.St.diag, eigendecomp_diag, eigendecomp_general, operand=St)

    
    Kl = kf.Kl(hp, x_l, x_l)
    Kl_tilde = Q_L_neg_half_Sl.T @ Kl @ Q_L_neg_half_Sl
    lam_Kl_tilde, eigen_dict["W_l"] = jax.lax.cond(kf.Kl.diag and kf.Sl.diag, eigendecomp_diag_K_tilde, eigendecomp_general_K_tilde,
                                                   Kl_tilde, Q_L_neg_half_Sl)
    
    Kt = kf.Kt(hp, x_t, x_t)
    Kt_tilde = Q_L_neg_half_St.T @ Kt @ Q_L_neg_half_St
    lam_Kt_tilde, eigen_dict["W_t"] = jax.lax.cond(kf.Kt.diag and kf.St.diag, eigendecomp_diag_K_tilde, eigendecomp_general_K_tilde,
                                                   Kt_tilde, Q_L_neg_half_St)
    
    D = jnp.outer(lam_Kl_tilde, lam_Kt_tilde) + 1.
    eigen_dict["D_inv"] = jnp.reciprocal(D)
    
    lam_S = jnp.outer(lam_Sl, lam_St)
    eigen_dict["logdetK"] = jnp.log(jnp.multiply(D, lam_S)).sum()
    
    return eigen_dict


def kronecker_inv_vec(R, eigen_dict):
    
    b = kron_prod(eigen_dict["W_l"].T, eigen_dict["W_t"].T, R)
    b = jnp.multiply(eigen_dict["D_inv"], b)
    b = kron_prod(eigen_dict["W_l"], eigen_dict["W_t"], b)

    return b
