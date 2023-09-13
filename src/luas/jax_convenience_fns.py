import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from .luas_types import PyTree, JAXArray

__all__ = [
    "order_list",
    "array_to_pytree2D",
    "pytree2D_to_array",
]

def order_list(par_list: list) -> list:
    
    map_dict = {par:i for (i, par) in enumerate(par_list)}
    list_map, f = ravel_pytree(map_dict)
    
    par_ordered = [par_list[par_ind] for par_ind in list_map]
    
    return par_ordered

    
def array_to_pytree2D(p_untransf: PyTree, hessian_array: JAXArray) -> PyTree:

    param_list = list(p_untransf.keys())
    par_order = order_list(param_list)
    cov_order = {}

    i = 0
    for param in par_order:
        if type(p_untransf[param]) in [float, np.float32, np.float64, jnp.float32, jnp.float64]:
            cov_order[param] = jnp.arange(i, i+1)
            i += 1
        else:
            param_size = p_untransf[param].size
            cov_order[param] = jnp.arange(i, i+param_size)
            i += param_size

    hess_dict = {p:{} for p in par_order}

    for (k1, v1) in cov_order.items():
        for (k2, v2) in cov_order.items():
            hess_dict[k1][k2] = hessian_array[jnp.ix_(v1, v2)]

    return hess_dict
    
    
def pytree2D_to_array(p_untransf: PyTree, hessian_dict: PyTree) -> JAXArray:
        
        cov_order = {}
        
        param_list = list(p_untransf.keys())
        par_order = order_list(param_list)
        
        i = 0
        for param in par_order:
            if type(p_untransf[param]) in [float, np.float32, np.float64, jnp.float32, jnp.float64]:
                cov_order[param] = jnp.arange(i, i+1)
                i += 1
            else:
                param_size = p_untransf[param].size
                cov_order[param] = jnp.arange(i, i+param_size)
                i += param_size
        
        hess_mat = jnp.zeros((i, i))

        for (k1, v1) in cov_order.items():
            for (k2, v2) in cov_order.items():
                if v1.size > 1 and v2.size == 1:
                    hess_mat = hess_mat.at[jnp.ix_(v1, v2)].set(hessian_dict[k1][k2].reshape((v1.size, v2.size)))
                else:
                    hess_mat = hess_mat.at[jnp.ix_(v1, v2)].set(hessian_dict[k1][k2])

        hess_mat = 0.5 * (hess_mat + hess_mat.T)
        
        return hess_mat