import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from tqdm import tqdm
from copy import deepcopy
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
    """Orders a list in the same way jax.flatten_util.ravel_pytree will order dictionary keys
    to form an array.
    
    Args:
        par_list (list): List to sort
        
    Returns:
        list: The list sorted to match jax.flatten_util.ravel_pytree.
    """
    
    # Create a dictionary with values ordered 0 to len(par_list) - 1
    map_dict = {par:i for (i, par) in enumerate(par_list)}
    
    # Run ravel_pytree on this dict to get the order it sorts in
    ind_order, f = ravel_pytree(map_dict)
    
    # Create a new list which sorts the old list in the way ravel_pytree flattens to arrays.
    par_ordered = [par_list[par_ind] for par_ind in ind_order]
    
    return par_ordered

    
def order_dict(par_dict: dict) -> Tuple[list, list]:
    """Takes a PyTree/dict and returns two lists ordered to match how jax.flatten_util.ravel_pytree
    would sort them, one list for the keys in the input dictionary and another list for the values.
    This function is useful for determining how to sort a dictionary of PyMC tensor variables into an array
    as ravel_pytree will not work on a PyTree containing PyMC tensor variables.
    
    Args:
        par_dict (PyTree): Dictionary of inputs to have keys and values sorted.
        
    Returns:
        list: List of keys sorted to match jax.flatten_util.ravel_pytree.
        list: List of values sorted to match jax.flatten_util.ravel_pytree.
    """
    
    key_list = list(par_dict.keys())
    map_dict = {par:i for (i, par) in enumerate(key_list)}
    ind_order, f = ravel_pytree(map_dict)

    par_keys_ordered = [key_list[par_ind] for par_ind in ind_order]
    par_values_ordered = [par_dict[par] for par in par_keys_ordered]
    
    return par_keys_ordered, par_values_ordered


def array_to_pytree_2D(p: PyTree, arr_2D: JAXArray) -> PyTree:
    """Takes a 2D array (e.g. a covariance matrix) where the rows and columns are sorted according to jax.flatten_util.ravel_pytree(p)
    for some input PyTree of parameters p and returns a nested PyTree of the array sorted into the parameter values.
    
    Args:
        p (PyTree): Parameters which describe the order of the array according to how jax.flatten_util.ravel_pytree
            will flatten into an array.
        arr_2D (JAXArray): A 2D array where the rows and columns are both sorted in the order in which jax.flatten_util.ravel_pytree
            flattens the parameter PyTree p.
        
    Returns:
        PyTree: A nested PyTree where the input 2D array has been rearranged to its corresponding parameters.
    """
    
    # First get the function which can convert an array into a PyTree like p in the right order
    p_arr, make_p_dict = ravel_pytree(p)
    
    # Use this function to sort the numbers from 0 to N for N parameters
    # This will be used to sort the array into a nested PyTree in the right order
    coord_dict = make_p_dict(jnp.arange(p_arr.size))
    
     
    pytree_2D = {}
    # Loop through the parameters lying along each row
    for k1 in p.keys():
        pytree_2D[k1] = {}
        
        # Loop through the parameters lying along each column
        for k2 in p.keys():
            
            # Select the array elements corresponding to the row and column parameters
            pytree_2D[k1][k2] = arr_2D[jnp.ix_(coord_dict[k1], coord_dict[k2])]
    
    return pytree_2D


def pytree_to_array_2D(
    p: PyTree,
    pytree_2D: JAXArray,
    param_order: Optional[list] = None,
)-> PyTree:
    """Inverse of array_to_pytree_2D, takes a nested PyTree (e.g. describing a covariance matrix) where the keys
    correspond to the row and column of a 2D array with a value defined for each parameter with every other parameter.
    Sorts the nested PyTree into this 2D array sorted according to param_order, defaults to the order
    jax.flatten_util.ravel_pytree will sort dictionary keys into when forming an array.
    
    Args:
        p (PyTree): Parameters which describe the order of the array according to how jax.flatten_util.ravel_pytree
            will flatten into an array.
        pytree_2D (PyTree): A nested PyTree describing a 2D array keyed by the pair of parameters along each row and column.
        
    Returns:
        JAXArray: A 2D array which the nested PyTree has been sorted into.
            
    """
    
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
    """A sigmoid function. When dealing with parameters bounded between limits [a, b], PyMC and NumPyro
    vary a parameter between (-jnp.inf, jnp.inf) and use a sigmoid transform followed by an affine transform to
    map this interval to the interval (a, b).
    
    Needed by GP.laplace_approx_with_bounds to correctly calculate the Laplace approximation with respect to
    the transformed parameters being varied by PyMC and NumPyro.
    
    Args:
        x (JAXArray): A variable in the unbounded transformed space used by PyMC and NumPyro
        
    Returns:
        JAXArray: The sigmoid function applied to x, constraining it to the interval (0, 1)
    
    """
    
    return 1 / (1 + jnp.exp(-x))

def transf_to_unbounded(p, bounds):
    return jnp.log(p - bounds[0]) - jnp.log(bounds[1] - p)


def transf_from_unbounded(x, bounds):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (bounds[1] - bounds[0]) + bounds[0]


def transf_to_unbounded_params(p, param_bounds):
        p_pymc = deepcopy(p)

        for par in param_bounds.keys():
            if par in p_pymc.keys():
                p_pymc[par] = transf_to_unbounded(p[par], param_bounds[par])

        return p_pymc


def transf_from_unbounded_params(p_pymc, param_bounds):
    p = deepcopy(p_pymc)

    for par in param_bounds.keys():
        if par in p_pymc.keys():
            p[par] = transf_from_unbounded(p_pymc[par], param_bounds[par])

    return p
