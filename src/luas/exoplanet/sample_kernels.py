from ..LuasKernel import LuasKernel
from ..kernel_functions import evaluate_kernel, squared_exp_kernel
import jax.numpy as jnp
from typing import Optional, Dict
from ..luas_types import JAXArray

def Kl_VLT(
    hp: Dict[str, JAXArray],
    x_l1: JAXArray,
    x_l2: JAXArray,
    wn: Optional[bool] = True
) -> JAXArray:
    
    Kl = evaluate_kernel(squared_exponential_kernel, x_l1, x_l2, hp["l_l_CM"])
    
    h_mat = jnp.diag(hp["h_CM"] * jnp.ones_like(x_l1))
    Kl = h_mat @ Kl @ h_mat
    
    Kl += jnp.diag(hp["h_WSS"]**2)
    
    return Kl
Kl_VLT.hp = ["h_CM", "h_WSS", "l_l_CM"]
Kl_VLT.diag = False


def Kt_VLT(
    hp: Dict[str, JAXArray],
    x_t1: JAXArray,
    x_t2: JAXArray,
    wn: Optional[bool] = True
) -> JAXArray:
    
    Kt = evaluate_kernel(squared_exponential_kernel, x_t1, x_t2, hp["l_t"])
    
    return Kt
Kt_VLT.hp = ["l_t"]
Kt_VLT.diag = False


def Sl_VLT(
    hp: Dict[str, JAXArray],
    x_l1: JAXArray,
    x_l2: JAXArray,
    wn: Optional[bool] = True
) -> JAXArray:
    
    Sl = hp["h_HFS"]**2 * evaluate_kernel(squared_exponential_kernel, x_l1, x_l2, hp["l_l_HFS"])
    
    if wn:
        Sl += jnp.diag(jnp.square(hp["sigma"]) * jnp.ones_like(x_l1))
    
    return Sl
Sl_VLT.hp = ["h_HFS", "l_l_HFS", "sigma"]
Sl_VLT.diag = False


def St_VLT(
    hp: Dict[str, JAXArray],
    x_t1: JAXArray,
    x_t2: JAXArray,
    wn: Optional[bool] = True
) -> JAXArray:
    
    return jnp.eye(x_t1.size)
St_VLT.hp = []
St_VLT.diag = True


def build_VLT_kernel():
    VLT_kernel = LuasKernel()
    VLT_kernel.Kl = Kl_VLT
    VLT_kernel.Kt = Kt_VLT
    VLT_kernel.Sl = Sl_VLT
    VLT_kernel.St = St_VLT
    
    return VLT_kernel
