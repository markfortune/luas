import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from tqdm import tqdm
from typing import Callable, Optional, Tuple
from .luas_types import PyTree, JAXArray

__all__ = [
    "get_corr_mat",
    "order_list",
    "order_dict",
    "array_to_pytree_2D",
    "pytree_to_array_2D",
    "large_hessian_calc",
    "sigmoid",
    "transf_from_unbounded_params",
    "transf_to_unbounded_params",
]

def get_corr_mat(cov_mat: JAXArray) -> JAXArray:
    """Given a covariance matrix will return its corresponding correlation matrix.
    
    Args:
        cov_mat (JAXArray): Covariance matrix to convert
        
    Returns:
        JAXArray: The input covariance matrix converted to a correlation matrix.
    """
    
    d = jnp.diag(jnp.sqrt(jnp.reciprocal(jnp.diag(cov_mat))))
    return d @ cov_mat @ d


def order_list(par_list: list) -> list:
    
    map_dict = {par:i for (i, par) in enumerate(par_list)}
    ind_order, f = ravel_pytree(map_dict)
    
    par_ordered = [par_list[par_ind] for par_ind in ind_order]
    
    return par_ordered

    
def order_dict(par_dict: dict) -> Tuple[list, list]:
    
    key_list = list(par_dict.keys())
    map_dict = {par:i for (i, par) in enumerate(key_list)}
    ind_order, f = ravel_pytree(map_dict)

    par_keys_ordered = [key_list[par_ind] for par_ind in ind_order]
    par_values_ordered = [par_dict[par] for par in par_keys_ordered]
    
    return par_keys_ordered, par_values_ordered


def array_to_pytree_2D(p: PyTree, arr_2D: JAXArray) -> PyTree:
    p_arr, make_p_dict = ravel_pytree(p)
    coord_dict = make_p_dict(jnp.arange(p_arr.size))
    
    pytree_2D = {}
    for k1 in p.keys():
        pytree_2D[k1] = {}
        for k2 in p.keys():
            pytree_2D[k1][k2] = arr_2D[jnp.ix_(coord_dict[k1], coord_dict[k2])]
    
    return pytree_2D


def pytree_to_array_2D(
    p: PyTree,
    pytree_2D: JAXArray,
    param_order: Optional[list] = None,
)-> PyTree:
    
    if param_order is None:
        param_order = order_list(list(p.keys()))
        p_arr, make_p_dict = ravel_pytree(p)
        coord_dict = make_p_dict(jnp.arange(p_arr.size))
        N_par = p_arr.size
    else:
        coord_dict = {}
        N_par = 0
        for par in param_order:
            coord_dict[par] = jnp.arange(N_par, N_par + p[par].size)
            N_par += p[par].size
            
    arr_2D = jnp.zeros((N_par, N_par))
    for k1 in param_order:
        for k2 in param_order:
            arr_2D = arr_2D.at[jnp.ix_(coord_dict[k1], coord_dict[k2])].set(pytree_2D[k1][k2])
    
    return arr_2D


def large_hessian_calc(
    fn: Callable,
    p_arr: JAXArray,
    *args,
    block_size: Optional[int] = 50,
    **kwargs,
) -> JAXArray:
    
    grad_wrapper = lambda p_arr, i: jax.grad(fn)(p_arr, *args, **kwargs)[i]
    
    hessian_wrapper = jax.jit(jax.vmap(jax.grad(grad_wrapper), in_axes = (None, 0)))
    
    N_par = p_arr.size
    hess_arr = jnp.zeros((N_par, N_par))
    for i in tqdm(range(0, N_par, block_size)):
        rows = jnp.arange(i, i+block_size)
        hess_arr = hess_arr.at[rows, :].set(hessian_wrapper(p_arr, rows))

    if N_par % block_size > 0:
        rows = jnp.arange(i+block_size, N_par)
        hess_arr = hess_arr.at[rows, :].set(hessian_wrapper(p_arr, rows))

    return hess_arr


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def transf_to_unbounded(p, bounds):
    return jnp.log(p - bounds[0]) - jnp.log(bounds[1] - p)


def transf_from_unbounded(x, bounds):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (bounds[1] - bounds[0]) + bounds[0]


def transf_to_unbounded_params(self, p, param_bounds):
        p_pymc = deepcopy(p)

        for par in param_bounds.keys():
            if par in p_pymc.keys():
                p_pymc[par] = transf_to_unbounded(p[par], param_bounds[par])

        return p_pymc


def transf_from_unbounded_params(self, p_pymc, param_bounds):
    p = deepcopy(p_pymc)

    for par in param_bounds.keys():
        if par in p_pymc.keys():
            p[par] = transf_from_unbounded(p_pymc[par], param_bounds[par])

    return p
