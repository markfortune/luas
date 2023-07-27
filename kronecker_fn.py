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


def eigendecomp_grad_cancel_jittable(hp, x_l, x_t, kf, eigen_dict = {}, Y = None, mf = None):
    
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
    
    
    # This transformation is used for both the r^T K^-1 r and logdetK derivatives
    W_Kl_W = eigen_dict["W_l"].T @ Kl @ eigen_dict["W_l"]
    W_Kt_W = eigen_dict["W_t"].T @ Kt @ eigen_dict["W_t"]
    W_Sl_W = eigen_dict["W_l"].T @ Sl @ eigen_dict["W_l"]
    W_St_W = eigen_dict["W_t"].T @ St @ eigen_dict["W_t"]
    
    # Generate mean function and compute residuals
    M = mf(hp, x_l, x_t)
    R = Y - M
    
    # Compute outer part of r^T K^-1 r derivative
    alpha1 = kron_prod(eigen_dict["W_l"].T, eigen_dict["W_t"].T, R)
    alpha2 = jnp.multiply(eigen_dict["D_inv"], alpha1)
    
    K_alpha = kron_prod(W_Kl_W, W_Kt_W, alpha2)
    K_alpha += kron_prod(W_Sl_W, W_St_W, alpha2)

    # Diagonal of these terms is used for logdetK transformation
    Kl_diag = jnp.diag(W_Kl_W)
    Kt_diag = jnp.diag(W_Kt_W)
    Sl_diag = jnp.diag(W_Sl_W)
    St_diag = jnp.diag(W_St_W)
    
    # Computes diagonal of W.T K W for calculation of logdetK
    W_K_W_diag = jnp.outer(Kl_diag, Kt_diag) + jnp.outer(Sl_diag, St_diag)
    eigen_dict["grad_logdetK"] = jnp.multiply(eigen_dict["D_inv"], W_K_W_diag).sum()
    
    return eigen_dict


# def compare_parameters(parameters, hp, eigen_hp, rtol, atol):
#     return jnp.all(jnp.array([jnp.allclose(hp[par], eigen_hp[par], rtol=rtol, atol=atol) for par in parameters]))


# def eigendecomp_calc_S(S, S_diag):
#     return jax.lax.cond(S_diag, eigendecomp_diag, eigendecomp_general, operand=S)


# def eigendecomp_calc_K(K, Q_L_neg_half_S, K_diag):
#     K_tilde = Q_L_neg_half_S.T @ K @ Q_L_neg_half_S
#     return jax.lax.cond(K_diag, eigendecomp_diag_K_tilde, eigendecomp_general_K_tilde,
#                                                    K_tilde, Q_L_neg_half_S)


# def eigendecomp_jittable(hp, x_l, x_t, kf, eigen_dict = {}, rtol=1e-12, atol=1e-12):
    
#     dict_length = len(eigen_dict)  # This needs to be a tensor for JIT compatibility
#     dict_empty = jnp.equal(dict_length, 0)
    
#     eigen_hp = eigen_dict.get("hp", {})
    
#     Kl_same = jax.lax.cond(dict_empty,
#                            lambda _: False,
#                            lambda _: compare_parameters(kf.Kl.hp, hp, eigen_hp, rtol, atol),
#                            operand=None)
#     Kt_same = jax.lax.cond(dict_empty,
#                            lambda _: False,
#                            lambda _: compare_parameters(kf.Kt.hp, hp, eigen_hp, rtol, atol),
#                            operand=None)
#     Sl_same = jax.lax.cond(dict_empty,
#                            lambda _: False,
#                            lambda _: compare_parameters(kf.Sl.hp, hp, eigen_hp, rtol, atol),
#                            operand=None)
#     St_same = jax.lax.cond(dict_empty,
#                            lambda _: False,
#                            lambda _: compare_parameters(kf.St.hp, hp, eigen_hp, rtol, atol),
#                            operand=None)
# #     Kl_diff = Kt_diff = Sl_diff = St_diff = False

#     eigen_dict["lam_Sl"], eigen_dict["Q_L_neg_half_Sl"] = jax.lax.cond(Sl_same, 
#                                                                        lambda a, b: (eigen_dict["lam_Sl"], eigen_dict["Q_L_neg_half_Sl"]), 
#                                                                        eigendecomp_calc_S,
#                                                                        kf.Sl(hp, x_l, x_l), kf.Sl.diag)

#     eigen_dict["lam_St"], eigen_dict["Q_L_neg_half_St"] = jax.lax.cond(St_same,
#                                                                        lambda a, b: (eigen_dict["lam_St"], eigen_dict["Q_L_neg_half_St"]), 
#                                                                        eigendecomp_calc_S,
#                                                                        kf.St(hp, x_t, x_t), kf.St.diag)
#     eigen_dict["lam_Kl_tilde"], eigen_dict["W_l"] = jax.lax.cond(Kl_same,
#                                                                  lambda a, b, c: (eigen_dict["lam_Kl_tilde"], eigen_dict["W_l"]),
#                                                                  eigendecomp_calc_K,
#                                                                  kf.Kl(hp, x_l, x_l), eigen_dict["Q_L_neg_half_Sl"], kf.Kl.diag and kf.Sl.diag)
#     eigen_dict["lam_Kt_tilde"], eigen_dict["W_t"] = jax.lax.cond(Kt_same, 
#                                                                  lambda a, b, c: (eigen_dict["lam_Kt_tilde"], eigen_dict["W_t"]), 
#                                                                  eigendecomp_calc_K,
#                                                                  kf.Kt(hp, x_t, x_t), eigen_dict["Q_L_neg_half_St"], kf.Kt.diag and kf.St.diag)
    
    
#     D = jnp.outer(eigen_dict["lam_Kl_tilde"], eigen_dict["lam_Kt_tilde"]) + 1.
#     eigen_dict["D_inv"] = jnp.reciprocal(D)
    
#     lam_S = jnp.outer(eigen_dict["lam_Sl"], eigen_dict["lam_St"])
#     eigen_dict["logdetK"] = jnp.log(jnp.multiply(D, lam_S)).sum()
    
#     return eigen_dict



def kronecker_inv_vec(R, eigen_dict):
    
    b = kron_prod(eigen_dict["W_l"].T, eigen_dict["W_t"].T, R)
    b = jnp.multiply(eigen_dict["D_inv"], b)
    b = kron_prod(eigen_dict["W_l"], eigen_dict["W_t"], b)

    return b
