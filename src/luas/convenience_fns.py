import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from tqdm import tqdm
from copy import deepcopy
from typing import Callable, Optional, Tuple, Union
from .luas_types import PyTree, JAXArray, is_scalar
import warnings

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
    "bin_data_1D",
    "bin_data_2D",
    "interp_params",
    "stack_params",
    "stack_obs_and_pred_points",
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
    
    # Get inverse sqrt of diagonal elements
    d = jnp.diag(jnp.sqrt(jnp.reciprocal(jnp.diag(cov_mat))))
    
    # Divides each covariance element by the standard deviation of each pair of parameters
    corr_mat = d @ cov_mat @ d
    
    # Setting the diagonal to zero can be useful for visualising small correlation
    if zero_diag:
        corr_mat = corr_mat*(1 - jnp.eye(corr_mat.shape[0]))
        
    return corr_mat


def check_params_in_bounds(p, param_bounds, bound_tol = 1e-9):
    """Checks that parameters lie inside their bounds, nudging any which lie exactly on a bound
    just inside it.

    ``PyMC`` and ``NumPyro`` sample bounded parameters in an unbounded space (see
    :func:`transf_to_unbounded`), where a value exactly on a bound maps to infinity. Any values which
    lie on (or within ``bound_tol`` of) a bound are moved inside by ``bound_tol`` with a warning, and a
    ``ValueError`` is raised for values further outside the bounds.

    Example:
        ``param_bounds`` should be of the form ``param_bounds[par] = [lower_bound, upper_bound]``,
        where the bounds are scalars or arrays with the same shape as ``p[par]``.

    Args:
        p (PyTree): Parameters to check. Parameters without bounds in ``param_bounds`` are not checked.
        param_bounds (PyTree): Lower and upper bounds for any of the parameters in ``p``.
        bound_tol (Scalar, optional): How far inside a bound to move values lying on it.
            Defaults to ``1e-9``.

    Returns:
        PyTree: A copy of ``p`` with any values lying on a bound moved inside it.

    Raises:
        ValueError: If any parameter lies outside its bounds by more than ``bound_tol``.

    """
    p_new = deepcopy(p)

    # Check if values are all within bounds
    for par in p.keys():
        if par in param_bounds.keys():
            # Allow bounds to be floats/scalars as well as arrays, broadcasting them to the shape of the parameter
            lower = np.broadcast_to(np.asarray(param_bounds[par][0]), np.shape(p[par]))
            upper = np.broadcast_to(np.asarray(param_bounds[par][1]), np.shape(p[par]))
            param_bounds_par = [lower, upper]

            ind_below = p[par] <= lower
            ind_above = p[par] >= upper

            # Check params are above lower bounds
            if jnp.any(ind_below):
                if jnp.all((p[par][ind_below] + bound_tol) > lower[ind_below]):
                    # Do small adjustment to correct numerical jitter errors
                    warnings.warn(f"Input parameter {par} at indices {np.nonzero(ind_below.ravel())} at lower prior bound and shifted inside by +{bound_tol}", UserWarning)
                    p_new[par][ind_below] += bound_tol
                else:
                    raise ValueError(f"Parameter {par} is below param_bounds at indices {np.nonzero(ind_below.ravel())[0]}. Value: {p[par]}, Bounds: {param_bounds_par}")
            
            # Check params are below upper bounds
            if jnp.any(ind_above):
                if jnp.all((p[par][ind_above] - bound_tol) < upper[ind_above]):
                    warnings.warn(f"Input parameter {par} at indices {np.arange(p[par].size)[ind_above]} at upper prior bound and shifted inside by -{bound_tol}", UserWarning)
                    p_new[par][ind_above] -= bound_tol
                else:
                    raise ValueError(f"Parameter {par} is out of param_bounds at indices {np.nonzero(ind_above.ravel())[0]}. Value: {p[par]}, Bounds: {param_bounds_par}")
    return p_new


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
    """Takes a PyTree and returns two lists ordered to match how ``jax.flatten_util.ravel_pytree``
    would sort the keys, one list for the keys in the input dictionary and another list for the values.
    This function is useful for determining how to sort a dictionary of ``PyMC`` tensor variables into an array
    as ``jax.flatten_util.ravel_pytree`` will not work on a PyTree containing ``PyMC`` tensor variables.
    
    Args:
        par_dict (PyTree): PyTree to sort by keys
        
    Returns:
        (:obj:`list` of :obj:`str`, :obj:`list` of :obj:`str`): Returns a tuple of two lists, the first a list of keys
        sorted to match ``jax.flatten_util.ravel_pytree`` and the second a list of values also sorted to match
        ``jax.flatten_util.ravel_pytree``.
        
    """
    
    # Since the values may not be JAX variables, create a similar dict with integer values
    key_list = list(par_dict.keys())
    map_dict = {par:i for (i, par) in enumerate(key_list)}
    ind_order, f = ravel_pytree(map_dict)

    # Use the order the integers are sorted into to order the list of keys and list of values
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

    coord_dict = {k: jnp.ravel(v) for k, v in coord_dict.items()}
     
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
    """Inverse of ``array_to_pytree_2D``, takes a nested PyTree (e.g. describing a covariance matrix) where the keys
    correspond to the row and column of a 2D array with a value defined for each parameter with every other parameter.
    Sorts the nested PyTree into this 2D array sorted according to param_order, defaults to the order
    ``jax.flatten_util.ravel_pytree`` will sort dictionary keys into when forming an array.
    
    Args:
        p (PyTree): Parameters used for log likelihood calculations, used to describe the order of the array
            according to how ``jax.flatten_util.ravel_pytree`` will flatten into an array.
        pytree_2D (PyTree): A nested PyTree describing a 2D array keyed by the pair of parameters along each row and column.
        param_order (:obj:`list` of :obj:`str`, optional): The order of the parameters along the rows and
            columns of the output array. Defaults to the order ``jax.flatten_util.ravel_pytree`` uses.

    Returns:
        JAXArray: A 2D array of shape ``(N_par, N_par)`` which the nested PyTree has been sorted into,
        where ``N_par`` is the total number of parameter values in ``p``.

    """
    
    # First must solve for the order in the array to place each parameter into
    if param_order is None:
        # Defaults to sorting how ravel_pytree will sort
        param_order = order_list(list(p.keys()))
        p_arr, make_p_dict = ravel_pytree(p)
        coord_dict = make_p_dict(jnp.arange(p_arr.size))

        coord_dict = {k: jnp.ravel(v) for k, v in coord_dict.items()}

        # Get the total number of parameters
        N_par = p_arr.size
    else:
        # If a specified order has been given must solve for the locations in an array
        # each parameter should go to
        coord_dict = {}
        N_par = 0
        for par in param_order:
            coord_dict[par] = jnp.arange(N_par, N_par + np.size(p[par]))
            N_par += np.size(p[par])
            
    # Loop through each element in the nested PyTree and use the array locations already found
    arr_2D = jnp.zeros((N_par, N_par))
    for k1 in param_order:
        for k2 in param_order:
            # Reshape as e.g. jax.hessian returns shape (N,) for a vector-scalar pair but (N, 1) is needed here
            block = jnp.reshape(pytree_2D[k1][k2], (coord_dict[k1].size, coord_dict[k2].size))
            arr_2D = arr_2D.at[jnp.ix_(coord_dict[k1], coord_dict[k2])].set(block)
    
    return arr_2D

    

def varying_params_wrapper(
    p: PyTree,
    vars: Optional[list] = None,
    fixed_vars: Optional[list] = None,
    to_numpy: Optional[bool] = True
) ->  Tuple[PyTree, Callable]:
    """Often useful to take a PyTree of parameters and return a subset of the (key, value)
    pairs which are to be varied. E.g. ``PyMC`` requires start values at initialisation of
    optimisation/inference to only include parameters which are to be fit for.
    
    Also returns a function which can take the subset of parameters being varied and
    return the full set of parameters with the fixed parameters added back in.
    
    Note: By default will return parameter values in NumPy arrays as this is required for inference
    with ``PyMC``. This can be turned off by setting ``to_numpy = False``.

    Args:
        p (PyTree): The full set of parameters used for likelihood calculations
            potentially including both mean function parameters and hyperparameters.
        vars (:obj:`list` of :obj:`str`, optional): The list of keys names corresponding to
            the parameters being varied which we want to include in the output
            parameter PyTree. If specified in addition to fixed_vars will raise an ``Exception``.
        fixed_vars (:obj:`list` of :obj:`str`, optional): Alternative to vars, may specify instead
            the parameters being kept fixed which will be excluded from the output parameter
            PyTree.  If specified in addition to vars will raise an Exception.
        to_numpy (bool, optional): Converts parameter values from input parameter PyTree to
            ``NumPy`` arrays as this is required for inference with ``PyMC``. Defaults to ``True``.
    
    Returns:
        (PyTree, Callable): Returns a tuple where the first element is the input PyTree ``p``
        but containing only the ``(key, value)`` pairs of parameters to be varied, the second element
        is a function which takes the output parameter PyTree containing only the parameters being
        varied and adds back in the fixed parameters without overwritting the parameters being varied.
    
    """
    
    if to_numpy:
        # PyMC cannot deal with values being JAXArrays so must convert to NumPy
        to_array = np.array
    else:
        # Do nothing to parameter values if to_numpy = False
        to_array = lambda p: p
    
    if vars is not None and fixed_vars is None:
        # If just vars specified
        p_vary = {par:to_array(p[par]) for par in vars}
        
    if vars is None and fixed_vars is not None:
        # If just fixed_vars specified

        for par in fixed_vars:
            # Check all specified fixed_vars are actually in p
            assert par in p.keys()
            
        p_vary = {par:to_array(p[par]) for par in p.keys() if par not in fixed_vars}
        
    elif vars is None and fixed_vars is None:
        # If neither vars nor fixed_vars specified just keep all parameters
        p_vary = {par:to_array(p[par]) for par in p.keys()}
        
    elif vars is not None and fixed_vars is not None:
        # If both vars and fixed vars specified raise an Exception
        raise Exception("Both vars and fixed_vars cannot be defined!")

    # Create a function which takes the subset of parameters in p_vary and
    # adds back in the fixed parameters
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
    jit: Optional[bool] = True,
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
        *args: Extra positional arguments passed to ``fn``.
        block_size (int, optional): The number of rows of the hessian to calculate at once.
            Large numbers will have a higher memory cost but may result in a shorter runtime.
            Defaults to 50.
        return_array (bool, optional): Whether to return the hessian as an array of shape ``(N_{par}, N_{par})``
            for ``N_{par}`` total parameters in ``p`` or as a nested PyTree where the hessian values for parameters
            named par1 and par2 would be given by ``hessian_pytree[par1][par2]`` and ``hessian_pytree[par2][par1]``.
            Defaults to True.
        jit (bool, optional): Whether to JIT compile the hessian function, can speed up the calculation assuming
            the function can be JIT compiled. Defaults to ``True``.
        **kwargs: Extra keyword arguments passed to ``fn``.

    Returns:
        JAXArray or PyTree: Depending on whether ``return_array`` is set to ``True`` or ``False`` will either return a JAXArray
        or PyTree giving the hessian with respect to each parameter in ``p``. If returning a JAXArray then the
        parameters will be ordered in the same order that ``jax.flatten_util.ravel_pytree`` will order the input
        PyTree ``p``.
    
    """
    
    # To break calculation into rows must first write a wrapper function which takes as input an array of parameters
    p_arr, make_p_dict = ravel_pytree(p)
    fn_arr_wrapper = lambda p_arr: fn(make_p_dict(p_arr), *args, **kwargs)
    
    # We then create a function which returns the gradient of specific array elements
    grad_wrapper = lambda p_arr, i: jax.grad(fn_arr_wrapper)(p_arr)[i]
    
    # Finally we create a function which takes the derivative of a group of first derivatives
    # vmap allows us to take this gradient wrt an array of gradients as jax.grad only works
    # on functions which return scalars
    hessian_wrapper = jax.vmap(jax.grad(grad_wrapper), in_axes = (None, 0))
    
    if jit:
        hessian_wrapper = jax.jit(hessian_wrapper)
    
    N_par = p_arr.size
    hess_arr = jnp.zeros((N_par, N_par))
    
    # Iterates through blocks of rows solving second derivatives
    for i in tqdm(range(0, N_par, block_size)):
        rows = jnp.arange(i, i+block_size)
        hess_arr = hess_arr.at[rows, :].set(hessian_wrapper(p_arr, rows))

    # If the size of the array is not an even multiple of block_size will calculate the remaining rows
    if N_par % block_size > 0:
        rows = jnp.arange(i+block_size, N_par)
        hess_arr = hess_arr.at[rows, :].set(hessian_wrapper(p_arr, rows))
    
    # Defaults to returning a JAXArray of parameters but can return a nested PyTree
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
    r"""Transformation used by ``PyMC`` and ``NumPyro`` to convert a parameter which lies
    within the interval ``(bounds[0], bounds[1])`` to the interval ``(-jnp.inf, jnp.inf)``.

    For a parameter :math:`p` with lower bound :math:`a` and upper bound :math:`b`:

    .. math::

        x = \log(p - a) - \log(b - p)

    Args:
        par (JAXArray): A parameter value which lies in the interval ``(bounds[0], bounds[1])``.
        bounds (:obj:`list` of JAXArray): A list containing the lower bound of ``par`` as its first
            element and the upper bound of ``par`` as its second element.

    Returns:
        JAXArray: The unbounded transformed value of ``par`` used by ``PyMC`` and ``NumPyro`` when sampling.

    """
    
    return jnp.log(par - bounds[0]) - jnp.log(bounds[1] - par)


def transf_from_unbounded(x, bounds):
    r"""Inverse of the transformation :func:`transf_to_unbounded`. Converts an unbounded parameter lying
    within the interval ``(-jnp.inf, jnp.inf)`` to the bounded interval ``(bounds[0], bounds[1])``.

    For a lower bound :math:`a` and upper bound :math:`b`:

    .. math::

        p = a + (b - a)\,\mathrm{sigmoid}(x)
        
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
    """Replicates the transformation used by ``PyMC`` and ``NumPyro`` to convert a PyTree of parameters -
    some or all of which lie within the bounds given by ``param_bounds`` - to unbounded parameters
    using the transformation given in ``transf_to_unbounded``.
    
    Used for calculating the Laplace approximation of the posterior with respect to the transformed
    parameters used by ``PyMC`` and ``NumPyro`` for sampling.
    
    Examples:
        For a single parameter ``p["d"]`` which lies between bounds ``(a, b)``, ``param_bounds`` should be
        of the form: ``param_bounds = {"d":[a, b]}`` where ``a`` and ``b`` are scalars or have the same shape
        as ``p["d"]``.
    
    Args:
        p (PyTree): All parameters used for log likelihood calculations potentially including
            additional unbounded parameters.
        param_bounds (PyTree): Contains any bounds for the parameters in ``p``.
        
    Returns:
        PyTree: All parameters in ``p`` with the unbounded transformed values for any parameter in
        ``param_bounds``. Should match the transformed parameters being sampled by ``PyMC`` and ``NumPyro``.
    
    """
    # Copy any parameters which do not lie within bounds specified in param_bounds
    p_unbounded = deepcopy(p)

    # Loop through each parameter in the bounds
    for par in param_bounds.keys():
        # Check if the parameter is actually being varied
        if par in p_unbounded.keys():
            # Transforms to unbounded space
            p_unbounded[par] = transf_to_unbounded(p[par], param_bounds[par])

    return p_unbounded


def transf_from_unbounded_params(p_unbounded, param_bounds):
    """Inverse of the transformation in transf_to_unbounded_params. Converts parameters being
    sampled by ``PyMC`` and ``NumPyro`` to the parameters which lie between bounds described in
    ``param_bounds`` which are used for log likelihood calculations.
    
    Used for calculating the Laplace approximation of the posterior with respect to the transformed
    parameters used by ``PyMC`` and ``NumPyro`` for sampling.
    
    Examples:
        For a single parameter ``p["d"]`` which should lie between bounds ``(a, b)``, ``param_bounds`` should be
        of the form: ``param_bounds = {"d":[a, b]}`` where ``a`` and ``b`` are scalars or have the same shape
        as ``p["d"]``.
    
    Args:
        p_unbounded (PyTree): All parameters used for sampling by ``PyMC`` and ``NumPyro`` to convert
            for log likelihood calculations which lie in an transformed unbounded space.
        param_bounds (PyTree): Contains any bounds for the parameters in ``p_unbounded``.

    Returns:
        PyTree: All parameters in ``p_unbounded``, with any parameters in ``param_bounds`` transformed
        back to their bounded values used for log likelihood calculations.
    
    """
    
    # Copy any parameters which do not lie within bounds specified in param_bounds
    p_bounded = deepcopy(p_unbounded)

    # Loop through each parameter in the bounds
    for par in param_bounds.keys():
        # Check if the parameter is actually being varied
        if par in p_bounded.keys():
            # Transforms back to bounded space
            p_bounded[par] = transf_from_unbounded(p_unbounded[par], param_bounds[par])

    return p_bounded


def bin_data_1D(t, y, bin_size):
    """Bins a 1D time-series by averaging consecutive groups of ``bin_size`` points.

    Both the input locations ``t`` and the data ``y`` are binned in the same way so they
    stay aligned. If the number of points ``N`` is not a multiple of ``bin_size`` then the
    final ``N % bin_size`` points are dropped before binning.

    Args:
        t (JAXArray): Input locations (e.g. times) of shape ``(N,)``.
        y (JAXArray): Data values of shape ``(N,)``.
        bin_size (int): Number of consecutive points to average into each bin. Values of 1
            or less leave the data unbinned.

    Returns:
        (JAXArray, JAXArray): The binned input locations and binned data, each of shape
        ``(N // bin_size,)``. Copies of ``t`` and ``y`` are returned if ``bin_size <= 1``.

    Example:
        >>> t = np.arange(10.)
        >>> y = np.arange(10.)
        >>> t_bin, y_bin = bin_data_1D(t, y, 3)
        >>> y_bin  # last point dropped since 10 % 3 = 1
        array([1., 4., 7.])

    """
    N = y.size
        
    y_bin = y.copy()

    if bin_size > 1:
        bin_mat = np.kron(np.eye(N//bin_size), np.ones(bin_size)/bin_size)
        clip = N % bin_size

        if clip:
            t_bin = bin_mat @ t[:-clip]
            y_bin = bin_mat @ y_bin[:-clip]
        else:
            t_bin = bin_mat @ t
            y_bin = bin_mat @ y_bin
    else:
        t_bin = t.copy()
            
    return t_bin, y_bin


def bin_data_2D(x_l, x_t, Y, l_bin, t_bin):
    """Bins a 2D data set (e.g. a spectroscopic time-series) along one or both dimensions
    by averaging consecutive groups of points.

    The column (time) dimension is binned first by averaging groups of ``t_bin`` columns,
    then the row (wavelength) dimension is binned by averaging groups of ``l_bin`` rows.
    The input locations along each dimension are binned in the same way so they stay aligned
    with the binned data. Any leftover points which don't fill a complete bin (i.e. the final
    ``N_t % t_bin`` columns and ``N_l % l_bin`` rows) are dropped.

    Used by :meth:`luas.GP.plot_lightcurves` to bin light curves for plotting.

    Args:
        x_l (JAXArray): Input locations along the row dimension (e.g. wavelength) of shape ``(N_l,)``.
        x_t (JAXArray): Input locations along the column dimension (e.g. time) of shape ``(N_t,)``.
        Y (JAXArray): Data of shape ``(N_l, N_t)``.
        l_bin (int): Number of rows to average into each bin. Values of 1 or less leave
            the row dimension unbinned.
        t_bin (int): Number of columns to average into each bin. Values of 1 or less leave
            the column dimension unbinned.

    Returns:
        (JAXArray, JAXArray, JAXArray): The binned row input locations of shape ``(N_l // l_bin,)``,
        the binned column input locations of shape ``(N_t // t_bin,)`` and the binned data of shape
        ``(N_l // l_bin, N_t // t_bin)``.

    """
    N_l = Y.shape[0]
    N_t = Y.shape[1]
    
    Y_bin = Y.copy()

    if t_bin > 1:
        bin_mat_t = np.kron(np.eye(N_t//t_bin), np.ones(t_bin)/t_bin)
        clip = N_t%t_bin

        if clip:
            x_t_bin = bin_mat_t @ x_t[:-clip]
            Y_bin = (bin_mat_t @ Y_bin[:, :-clip].T).T
        else:
            x_t_bin = bin_mat_t @ x_t
            Y_bin = (bin_mat_t @ Y_bin.T).T
    else:
        x_t_bin = x_t.copy()

    if l_bin > 1:
        bin_mat_l = np.kron(np.eye(N_l//l_bin), np.ones(l_bin)/l_bin)
        clip = N_l % l_bin

        if clip:
            x_l_bin = bin_mat_l @ x_l[:-clip]
            Y_bin = (bin_mat_l @ Y_bin[:-clip, :])
        else:
            x_l_bin = bin_mat_l @ x_l
            Y_bin = (bin_mat_l @ Y_bin)
    else:
        x_l_bin = x_l.copy()
            
    return x_l_bin, x_t_bin, Y_bin


def interp_params(p_obs, p_pred, x, x_pred,
            interp_param = None, interp_row = 0):
    """Linearly interpolates selected parameters from the observed input locations onto the
    prediction locations.

    Used when making GP predictions (see :meth:`luas.GP.predict`) for parameters which take a
    different value at each input location along a dimension (e.g. a wavelength-dependent
    transit depth). The interpolation uses :func:`jax.numpy.interp`, so the observed locations
    must be in increasing order. Values at prediction locations outside the range of the observed
    locations are set to the value at the nearest end point.

    Args:
        p_obs (PyTree): Parameters at the observed locations. Each parameter named in
            ``interp_param`` must have one value per observed location.
        p_pred (PyTree): Parameters at the prediction locations. Any parameters not named in
            ``interp_param`` are returned unchanged.
        x (JAXArray): Observed input locations, either of shape ``(N,)`` or ``(d, N)`` if there
            are multiple input vectors along this dimension.
        x_pred (JAXArray): Prediction input locations, either of shape ``(N_pred,)`` or
            ``(d, N_pred)``.
        interp_param (:obj:`list` of str, optional): Names of the parameters in ``p_obs`` to
            interpolate. If ``None`` (the default) then ``p_pred`` is returned unchanged.
        interp_row (int, optional): If ``x`` and ``x_pred`` have more than one row of input
            locations, the index of the row to interpolate over. Defaults to 0.

    Returns:
        PyTree: A copy of ``p_pred`` where each parameter in ``interp_param`` has been replaced
        by its value interpolated onto ``x_pred``.

    """

    if interp_param is None:
        return p_pred
    else:
        p_interp = deepcopy(p_pred)
        
        if x.ndim > 1:
            x = x[interp_row, :]
            x_pred = x_pred[interp_row, :]
        
        for par in interp_param:
            p_interp[par] = jnp.interp(x_pred, x, p_obs[par])
                
        return p_interp
    

def stack_params(p_obs, p_pred, stack_names = [], sort_fns = []):
    """Joins parameters at the observed and prediction locations into a single PyTree of
    parameters for the full set of stacked input locations.

    For GP predictions, :meth:`luas.GP.predict` builds the covariance matrix over the observed
    and prediction locations together. Parameters which take a separate value at each input
    location must therefore be joined into a single array covering both. For each dimension
    ``d``, every parameter named in ``stack_names[d]`` is concatenated as
    ``[p_obs[par], p_pred[par]]``, then reordered with ``sort_fns[d]`` if a sorting function
    is given, so it matches the ordering of the stacked input locations returned by
    :func:`stack_obs_and_pred_points`.

    Args:
        p_obs (PyTree): Parameters at the observed locations.
        p_pred (PyTree): Parameters at the prediction locations.
        stack_names (:obj:`list` of :obj:`list` of str, optional): For each dimension, a list of
            the names of the parameters to stack along that dimension. If ``None`` then ``p_obs``
            is returned unchanged. Defaults to an empty list (no parameters stacked).
        sort_fns (:obj:`list` of Callable, optional): For each dimension, the sorting function
            returned by :func:`stack_obs_and_pred_points` (or ``None`` for no sorting). Must have
            one entry for each entry in ``stack_names``.

    Returns:
        PyTree: A copy of ``p_obs`` where each stacked parameter has been replaced by the
        (optionally sorted) concatenation of its observed and predicted values. All other
        parameters keep their values from ``p_obs``.

    """

    if stack_names is None:
        return p_obs
    else:
        p_stack = deepcopy(p_obs)

        for d in range(len(stack_names)):
            for par in stack_names[d]:
                p_stack[par] = jnp.concatenate([p_obs[par], p_pred[par]])

                if sort_fns[d] is not None:
                    p_stack[par] = sort_fns[d](p_stack[par])
                
        return p_stack


def stack_obs_and_pred_points(x, x_pred, sort = None):
    """Joins the observed and prediction input locations along one dimension into a single
    array, keeping track of which entries are observed and which are predicted.

    Used by :meth:`luas.GP.predict` to build the full set of input locations the conditional
    GP is evaluated over. Most quasiseparable kernels require their input locations to be in
    increasing order, so the stacked locations can optionally be sorted. A sorting function is
    then also returned so that any parameters defined at each input location can be reordered
    in the same way (see :func:`stack_params`).

    If ``x_pred`` is ``None``, the prediction locations are taken to be the observed locations
    and nothing is stacked, so every point is marked as both observed and predicted.

    Args:
        x (JAXArray): Observed input locations of shape ``(N,)`` or ``(d, N)``.
        x_pred (JAXArray, optional): Prediction input locations of shape ``(N_pred,)`` or
            ``(d, N_pred)``. If ``None`` then taken to be equal to ``x``.
        sort (bool, optional): Whether to sort the stacked input locations into increasing
            order. Sorting currently assumes 1D input locations of shape ``(N,)``.
            Defaults to ``None`` (no sorting).

    Returns:
        (JAXArray, JAXArray, JAXArray, JAXArray, Callable): A tuple of five elements:

        - **x_total** - The stacked (and optionally sorted) input locations, of shape
          ``(N + N_pred,)`` (or ``(N,)`` if ``x_pred`` is ``None``).
        - **x_pred** - The prediction input locations, which is a copy of ``x`` if ``x_pred``
          was ``None``.
        - **obs_points** - Boolean mask over ``x_total`` which is ``True`` at observed locations.
        - **pred_points** - Boolean mask over ``x_total`` which is ``True`` at prediction locations.
        - **sort_fn** - Function ``sort_fn(mat, inverse = False)`` which reorders the first axis of
          an array from the stacked ordering ``[x, x_pred]`` into the sorted ordering (or back
          again if ``inverse = True``). ``None`` if ``sort`` is falsy.

    """
    
    N = x.shape[-1]
    
    if x_pred is not None:
        x_total = jnp.concatenate([x, x_pred], axis = -1)
        obs_points = 1 == jnp.concatenate([jnp.ones(N), jnp.zeros(x_pred.shape[-1])])
        pred_points = ~obs_points
    else:
        x_total = x.copy()
        x_pred = x.copy()
        obs_points = 1 == jnp.ones(N)
        pred_points = 1 == jnp.ones(N)

    
    # Sort stuff if necessary
    if sort:
        sort_ind = x_total.argsort()
        invert_sort = sort_ind.argsort()

        x_total = x_total[sort_ind]
        obs_points = obs_points[sort_ind]
        pred_points = pred_points[sort_ind]

        def sort_fn(mat, sort_axis = 0, inverse = False):
            if inverse:
                return mat[invert_sort]
            else:
                return mat[sort_ind]
    else:
        sort_fn = None

    return x_total, x_pred, obs_points, pred_points, sort_fn
    