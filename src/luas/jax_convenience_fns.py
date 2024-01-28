import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from tqdm import tqdm
from copy import deepcopy
from typing import Callable, Optional, Tuple, Union
from .luas_types import PyTree, JAXArray

__all__ = [
    "get_corr_mat",
    "order_list",
    "order_dict",
    "array_to_pytree_2D",
    "pytree_to_array_2D",
    "varying_params_wrapper",
    "large_hessian_calc",
    "sigmoid",
    "transf_from_unbounded_params",
    "transf_to_unbounded_params",
]

def get_corr_mat(cov_mat: JAXArray, zero_diag: Optional[bool] = False) -> JAXArray:
    """Given a covariance matrix will return its corresponding correlation matrix.
    
    Args:
        cov_mat (JAXArray): Covariance matrix to convert
        zero_diag (bool, optional): Option to set diagonal to zeros to help visualise
            off-diagonal correlations.
        
    Returns:
        JAXArray: The input covariance matrix converted to a correlation matrix.
        
    """
    
    d = jnp.diag(jnp.sqrt(jnp.reciprocal(jnp.diag(cov_mat))))
    corr_mat = d @ cov_mat @ d
    
    if zero_diag:
        corr_mat = corr_mat*(1 - jnp.eye(corr_mat.shape[0]))
    return corr_mat


def order_list(par_list: list) -> list:
    """Orders a list in the same way ``jax.flatten_util.ravel_pytree`` will order dictionary keys
    to form an array.
    
    Args:
        par_list (:obj:`list` of :obj:`str`): List to sort
        
    Returns:
        :obj:`list` of :obj:`str`: The list sorted to match jax.flatten_util.ravel_pytree.
        
    """
    
    # Create a dictionary with values ordered 0 to len(par_list) - 1
    map_dict = {par:i for (i, par) in enumerate(par_list)}
    
    # Run ravel_pytree on this dict to get the order it sorts in
    ind_order, f = ravel_pytree(map_dict)
    
    # Create a new list which sorts the old list in the way ravel_pytree flattens to arrays.
    par_ordered = [par_list[par_ind] for par_ind in ind_order]
    
    return par_ordered

    
def order_dict(par_dict: dict) -> Tuple[list, list]:
    """Takes a PyTree/``dict`` and returns two lists ordered to match how ``jax.flatten_util.ravel_pytree``
    would sort them, one list for the keys in the input dictionary and another list for the values.
    This function is useful for determining how to sort a dictionary of PyMC tensor variables into an array
    as ravel_pytree will not work on a PyTree containing PyMC tensor variables.
    
    Args:
        par_dict (PyTree): Dictionary of inputs to have keys and values sorted.
        
    Returns:
        :obj:`list` of :obj:`str`: List of keys sorted to match ``jax.flatten_util.ravel_pytree``.
        :obj:`list` of :obj:`str`: List of values sorted to match ``jax.flatten_util.ravel_pytree``.
        
    """
    
    key_list = list(par_dict.keys())
    map_dict = {par:i for (i, par) in enumerate(key_list)}
    ind_order, f = ravel_pytree(map_dict)

    par_keys_ordered = [key_list[par_ind] for par_ind in ind_order]
    par_values_ordered = [par_dict[par] for par in par_keys_ordered]
    
    return par_keys_ordered, par_values_ordered


def array_to_pytree_2D(p: PyTree, arr_2D: JAXArray) -> PyTree:
    """Takes a 2D JAXArray (e.g. a covariance matrix) where the rows and columns are sorted according to ``jax.flatten_util.ravel_pytree``
    would sort the input PyTree of parameters ``p`` and returns a nested PyTree of the array sorted into the parameter values.
    
    Args:
        p (PyTree): Parameters used for log likelihood calculations, used to describe the order of the array
            according to how ``jax.flatten_util.ravel_pytree`` will flatten into an array.
        arr_2D (JAXArray): A 2D array where the rows and columns are both sorted in the order in which ``jax.flatten_util.ravel_pytree``
            flattens the parameter PyTree p.
        
    Returns:
        PyTree: A nested PyTree where the input 2D array ``arr_2D`` has been rearranged to its corresponding parameters.
        
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
        p (PyTree): Parameters used for log likelihood calculations, used to describe the order of the array
            according to how jax.flatten_util.ravel_pytree will flatten into an array.
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

    

def varying_params_wrapper(
    p: PyTree,
    vars: Optional[list] = None,
    fixed_vars: Optional[list] = None,
    to_numpy: Optional[bool] = True
) ->  Tuple[PyTree, Callable]:
    """Often useful to take a PyTree of parameters and return a subset of the (key, value)
    pairs which are to be varied. E.g. PyMC requires start values at initialisation of
    optimisation/inference to only include parameters which are to be fit for.
    
    Also returns a function which can take the subset of parameters being varied and
    return the full set of parameters with the fixed parameters added back in.
    
    Note: By default will return parameter values in NumPy arrays as this is required for inference
    with PyMC. This can be turned off by setting to_numpy to False.

    Args:
        p (PyTree): The full set of parameters used for likelihood calculations
            potentially including both mean function parameters and hyperparameters.
        vars (:obj:`list` of :obj:`str`, optional): The list of keys names corresponding to
            the parameters being varied which we want to include in the output
            parameter PyTree. If specified in addition to fixed_vars will raise an Exception.
        fixed_vars (:obj:`list` of :obj:`str`, optional): Alternative to vars, may specify instead
            the parameters being kept fixed which will be excluded from the output parameter
            PyTree.  If specified in addition to vars will raise an Exception.
        to_numpy (bool, optional): Converts parameter values from input parameter PyTree to
            NumPy arrays as this is required for inference with PyMC. Defaults to True.
    
    Returns:
        PyTree: The input PyTree ``p`` but containing only the (key, value) pairs of
            parameters to be varied.
        Callable: A function which takes the output parameter PyTree containing only the
            parameters being varied and adds back in the fixed parameters without overwritting
            the parameters being varied.
    
    """
    if to_numpy:
        to_array = np.array
    else:
        to_array = lambda p: p
    
    # PyMC requires input parameter arrays in NumPy
    if vars is not None and fixed_vars is None:
        p_vary = {par:to_array(p[par]) for par in vars}
    if vars is None and fixed_vars is not None:
        p_vary = {par:to_array(p[par]) for par in p.keys() if par not in fixed_vars}
    elif vars is None and fixed_vars is None:
        p_vary = {par:to_array(p[par]) for par in p.keys()}
    elif vars is not None and fixed_vars is not None:
        raise Exception("Both vars and fixed_vars cannot be defined!")

    p_fixed = deepcopy(p)
    def make_p(p_vary):
        p_fixed.update(p_vary)
        return p_fixed

    return p_vary, make_p
    

def large_hessian_calc(
    fn: Callable,
    p: PyTree,
    *args,
    block_size: Optional[int] = 50,
    return_array: Optional[bool] = True,
    **kwargs,
) -> Union[JAXArray, PyTree]:
    """Breaks up the calculation of the hessian of a large matrix into groups of rows to reduce
    the memory cost. Useful for large data sets when ``jax.hessian`` applied to a log likelihood function
    can cause the code to crash.
    
    This function should work for any arbitrary function however for which ``jax`` can calculate second-order
    derivatives and for functions where the first argument is a PyTree which the derivative is to be calculated
    with respect to.
    
    Args:
        fn (Callable): Function to calculate the hessian of. Should be of the form ``f(p, *args, **kwargs)``
            where ``p`` is a PyTree.
        p (PyTree): Parameters to calculate the derivative with respect to. All parameters within ``p`` will
            have the derivative be taken with respect to.
        block_size (int, optional): The number of groups of rows to calculate the second derivatives for at once.
            Large numbers will have a higher memory cost but may result in a shorter runtime.
            Defaults to 50.
        return_array (bool, optional): Whether to return the hessian as an array of shape ``(N_{par}, N_{par})``
            for ``N_{par}`` total parameters in ``p`` or as a nested PyTree where the hessian values for parameters
            named par1 and par2 would be given by hessian_pytree[par1][par2] and hessian_pytree[par2][par1].
            Defaults to True.
            
    Returns:
        JAXArray or PyTree: Depending on whether return_array is set to True or False will either return a JAXArray
            or PyTree giving the hessian with respect to each parameter in ``p``. If returning a JAXArray then the
            parameters will be ordered in the same order that ``jax.flatten_util.ravel_pytree`` will order the input
            PyTree ``p``.
    
    """
    
    p_arr, make_p_dict = ravel_pytree(p)
    fn_arr_wrapper = lambda p_arr: fn(make_p_dict(p_arr), *args, **kwargs)
    grad_wrapper = lambda p_arr, i: jax.grad(fn_arr_wrapper)(p_arr)[i]
    hessian_wrapper = jax.jit(jax.vmap(jax.grad(grad_wrapper), in_axes = (None, 0)))
    
    N_par = p_arr.size
    hess_arr = jnp.zeros((N_par, N_par))
    for i in tqdm(range(0, N_par, block_size)):
        rows = jnp.arange(i, i+block_size)
        hess_arr = hess_arr.at[rows, :].set(hessian_wrapper(p_arr, rows))

    if N_par % block_size > 0:
        rows = jnp.arange(i+block_size, N_par)
        hess_arr = hess_arr.at[rows, :].set(hessian_wrapper(p_arr, rows))

    if return_array:
        return hess_arr
    else:
        return array_to_pytree_2D(p, hess_arr)


def sigmoid(x):
    """A sigmoid function. When dealing with parameters bounded between limits [a, b], PyMC and NumPyro
    vary a parameter between (-jnp.inf, jnp.inf) and use a sigmoid transform followed by an affine transform to
    map this interval to the interval (a, b).
    
    Args:
        x (JAXArray): A variable in the unbounded transformed space used by PyMC and NumPyro
        
    Returns:
        JAXArray: The sigmoid function applied to x, constraining it to the interval (0, 1)
    
    """
    
    return 1 / (1 + jnp.exp(-x))


def transf_to_unbounded(par, bounds):
    """Transformation used by PyMC and NumPyro to convert a parameter which lies
    within the interval (bounds[0], bounds[1]) to the interval (-jnp.inf, jnp.inf).
    
    Args:
        p (JAXArray): A parameter value which lies in the interval (bounds[0], bounds[1])
        bounds (:obj:`list` of JAXArray): A list containing the lower bound of p as its first
            element and the upper bound of p as its second element
        
    Returns:
        JAXArray: The unbounded transformed value of p used by PyMC and NumPyro when sampling.
    
    """
    
    return jnp.log(par - bounds[0]) - jnp.log(bounds[1] - par)


def transf_from_unbounded(x, bounds):
    """Inverse of transformation transf_to_unbounded. Converts an unbounded parameter lying
    within the interval (-jnp.inf, jnp.inf) to the bounded interval (bounds[0], bounds[1]).
    
    Args:
        x (JAXArray): A parameter value which lies in the interval (-jnp.inf, jnp.inf).
        bounds (:obj:`list` of JAXArray): A list containing the lower bound of the parameter
            as its first element and the upper bound of the parameter as its second element.
        
    Returns:
        JAXArray: The bounded parameter used for log likelihood calculations.
    
    """
    
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (bounds[1] - bounds[0]) + bounds[0]


def transf_to_unbounded_params(p, param_bounds):
    """Replicates the transformation used by PyMC and NumPyro to convert a PyTree of parameters -
    some or all of which lie within the bounds given by param_bounds - to unbounded parameters
    using the transformation given in transf_to_unbounded.
    
    Used for calculating the Laplace approximation of the posterior with respect to the transformed
    parameters used by PyMC/NumPyro for sampling.
    
    Examples:
        For a single parameter p["d"] which lies between bounds (a, b), param_bounds should be
        of the form: param_bounds = {"d":[a, b]} where a and b have the same shape as p["d"].
    
    Args:
        p (PyTree): All parameters used for log likelihood calculations potentially including
            additional unbounded parameters.
        param_bounds (PyTree): Contains any bounds for the parameters in p.
        
    Returns:
        JAXArray: All parameters in p with the unbounded transformed values for any parameter in
            param_bounds. Should match the transformed parameters being sampled by PyMC/NumPyro
    
    """
    # Copy any parameters which do not lie within bounds specified in param_bounds
    p_pymc = deepcopy(p)

    for par in param_bounds.keys():
        if par in p_pymc.keys():
            p_pymc[par] = transf_to_unbounded(p[par], param_bounds[par])

    return p_pymc


def transf_from_unbounded_params(p_pymc, param_bounds):
    """Inverse of the transformation in transf_to_unbounded_params. Converts parameters being
    sampled by PyMC and NumPyro to the parameters which lie between bounds described in param_bounds
    which are used for log likelihood calculations.
    
    Used for calculating the Laplace approximation of the posterior with respect to the transformed
    parameters used by PyMC/NumPyro for sampling.
    
    Examples:
        For a single parameter p["d"] which should lie between bounds (a, b), param_bounds should be
        of the form: param_bounds = {"d":[a, b]} where a and b have the same shape as p["d"].
    
    Args:
        p_pymc (PyTree): All parameters used for sampling by PyMC/NumPyro to convert for log likelihood
            calculations.
        param_bounds (PyTree): Contains any bounds for the parameters in p.
        
    Returns:
        JAXArray: All parameters used for log likelihood calculations potentially including
            additional unbounded parameters.
    
    """
    
    # Copy any parameters which do not lie within bounds specified in param_bounds
    p = deepcopy(p_pymc)

    for par in param_bounds.keys():
        if par in p_pymc.keys():
            p[par] = transf_from_unbounded(p_pymc[par], param_bounds[par])

    return p
