from .LuasKernel import LuasKernel
from .GeneralKernel import GeneralKernel
from .kernel_functions import evaluate_kernel, rbf_kernel
import jax.numpy as jnp

def Kl_VLT(hp, x_l1, x_l2, wn = True):
    
    Kl = evaluate_kernel(rbf_kernel, x_l1, x_l2, hp["l_l_CM"])
    
    h_mat = jnp.diag(hp["h_CM"] * jnp.ones_like(x_l1))
    Kl = h_mat @ Kl @ h_mat
    
    Kl += jnp.diag(hp["h_WSS"]**2)
    
    return Kl
Kl_VLT.hp = ["h_CM", "h_WSS", "l_l_CM"]
Kl_VLT.diag = False


def Kl_VLT_l_l_WSS(hp, x_l1, x_l2, wn = True):
    
    Kl = evaluate_kernel(rbf_kernel, x_l1, x_l2, hp["l_l_CM"])
    h_mat = jnp.diag(hp["h_CM"] * jnp.ones_like(x_l1))
    Kl = h_mat @ Kl @ h_mat
    
    Kl_WSS = evaluate_kernel(rbf_kernel, x_l1, x_l2, hp["l_l_WSS"])
    h_mat2 = jnp.diag(hp["h_WSS"] * jnp.ones_like(x_l1))
    Kl += h_mat2 @ Kl_WSS @ h_mat2
    
    return Kl
Kl_VLT_l_l_WSS.hp = ["h_CM", "h_WSS", "l_l_CM", "l_l_WSS"]
Kl_VLT_l_l_WSS.diag = False


def Kt_VLT(hp, x_t1, x_t2, wn = True):
    
    Kt = evaluate_kernel(rbf_kernel, x_t1, x_t2, hp["l_t"])
#     Kt += jnp.diag(jnp.ones(x_t1.size))*1e-5
    
    return Kt
Kt_VLT.hp = ["l_t"]
Kt_VLT.diag = False


def Kl_VLT_no_WSS(hp, x_l1, x_l2, wn = True):
    
    Kl = evaluate_kernel(rbf_kernel, x_l1, x_l2, hp["l_l_CM"])
    
    h_mat = jnp.diag(hp["h_CM"] * jnp.ones_like(x_l1))
    Kl = h_mat @ Kl @ h_mat
    
    return Kl
Kl_VLT_no_WSS.hp = ["h_CM", "l_l_CM"]
Kl_VLT_no_WSS.diag = False


def Wl_VLT(hp, x_l1, x_l2, wn = True):
    
    Kl = evaluate_kernel(rbf_kernel, x_l1, x_l2, hp["l_l_WSS"])
    
    h_mat = jnp.diag(hp["h_WSS"] * jnp.ones_like(x_l1))
    Kl = h_mat @ Kl @ h_mat
    
    return Kl
Wl_VLT.hp = ["h_WSS", "l_l_WSS"]
Wl_VLT.diag = False


def Wt_VLT(hp, x_t1, x_t2, wn = True):
    
    Kt = evaluate_kernel(rbf_kernel, x_t1, x_t2, hp["l_t_WSS"])
#     Kt += jnp.diag(jnp.ones(x_t1.size))*1e-5
    
    return Kt
Wt_VLT.hp = ["l_t_WSS"]
Wt_VLT.diag = False


def Sl_VLT(hp, x_l1, x_l2, wn = True):
    
    Sl = hp["h_HFS"]**2 * evaluate_kernel(rbf_kernel, x_l1, x_l2, hp["l_l_HFS"])
    
    if wn:
        Sl += jnp.diag(jnp.square(hp["sigma"]) * jnp.ones_like(x_l1))
    
    return Sl
Sl_VLT.hp = ["h_HFS", "l_l_HFS", "sigma"]
Sl_VLT.diag = False


def St_VLT(hp, x_t1, x_t2, wn = True):
    return jnp.eye(x_t1.size)
St_VLT.hp = []
St_VLT.diag = True


def build_VLT_kernel():
    VLT_kernel = LuasKernel()
    VLT_kernel.Kl = Kl_VLT_l_l_WSS
    VLT_kernel.Kt = Kt_VLT
    VLT_kernel.Sl = Sl_VLT
    VLT_kernel.St = St_VLT
    
    return VLT_kernel


def Kl_HST(hp, x_l1, x_l2, wn = True):
    
    Kl = evaluate_kernel(rbf_kernel, x_l1, x_l2, hp["l_l_CM"])
    
    h_mat = jnp.diag(hp["h_CM"] * jnp.ones_like(x_l1))
    Kl = h_mat @ Kl @ h_mat
    
    Kl += jnp.diag(hp["h_WSS"]**2)
    
    return Kl
Kl_HST.hp = ["h_CM", "h_WSS", "l_l_CM"]
Kl_HST.diag = False


def Kt_HST(hp, x_t1, x_t2, wn = True):
    
    K_phase = evaluate_kernel(rbf_kernel, x_t1[1, :], x_t2[1, :], hp["l_p"])
    K_orbit_no = evaluate_kernel(rbf_kernel, x_t1[2, :], x_t2[2, :], hp["l_n"])
#     Kt += jnp.diag(jnp.ones(x_t1.size))*1e-5
    
    return K_phase*K_orbit_no
Kt_HST.hp = ["l_p", "l_n"]
Kt_HST.diag = False


def Sl_HST(hp, x_l1, x_l2, wn = True):
    
#     Sl = hp["h_HFS"]**2 * evaluate_kernel(rbf_kernel, x_l1, x_l2, hp["l_l_HFS"])
    
    Sl = jnp.zeros((x_l1.size, x_l2.size))
    if wn:
        Sl = jnp.diag(jnp.square(hp["sigma"]) * jnp.ones_like(x_l1))
    
    return Sl
Sl_HST.hp = ["sigma"]
Sl_HST.diag = False


def St_HST(hp, x_t1, x_t2, wn = True):
    return jnp.eye(x_t1.shape[1])
St_HST.hp = []
St_HST.diag = True



def build_HST_kernel():
    HST_kernel = Kernel()
    HST_kernel.Kl = Kl_HST
    HST_kernel.Kt = Kt_HST
    HST_kernel.Sl = Sl_HST
    HST_kernel.St = St_HST
    
    HST_kernel.St_diag = True
    
    return HST_kernel


def K_VLT(hp, x_l1, x_l2, x_t1, x_t2, wn = True):
    
    Kl = Kl_VLT(hp, x_l1, x_l2, wn = wn)
    Kt = Kt_VLT(hp, x_t1, x_t2, wn = wn)
    Sl = Sl_VLT(hp, x_l1, x_l2, wn = wn)
    St = St_VLT(hp, x_t1, x_t2, wn = wn)
    
    return jnp.kron(Kl, Kt) + jnp.kron(Sl, St)


def build_VLT_GeneralKernel():
    return GeneralKernel(Kl_fns = [Kl_VLT_no_WSS, Sl_VLT, Wl_VLT], Kt_fns = [Kt_VLT, St_VLT, Wt_VLT])