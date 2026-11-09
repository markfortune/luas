import warnings

import numpy as np
import jax
from jax.flatten_util import ravel_pytree
from .convenience_fns import order_dict
import pymc as pm
from copy import deepcopy

# pymc is very frustrating and constantly changes with a load of deprecation warnings
# therefore there's a load of checks I've to do just to support versions 5 and 6...
try:
    pymc_version = int(str(pm.__version__).split(".")[0])
except ValueError:
    # pm.__version__ is not a normal version string. This happens when Sphinx
    # builds the docs with PyMC replaced by a placeholder module, so assume v5.
    pymc_version = 6

if pymc_version < 6:
    raise Exception(f"PyMC Version {pymc_version} not currently supported. Supported versions are >=6.")
else:
    # this apparently only costs about 10 us per gradient call but allows PyMC to work in pure Python which is good for installing easily
    # anyway the warning doesn't need to be printed everytime a model is built
    warnings.filterwarnings("ignore", message="Numba will use object mode to run LuasPyMCWrapper", category=UserWarning)

# PyMC uses pytensor as a backend
import pytensor.tensor as tens
from pytensor.graph import Apply, Op
from pytensor.tensor.variable import TensorVariable

import inspect

__all__ = [
    "LuasPyMC",
    "LuasPyMCWrapper",
]

def find_MAP(*args, method='L-BFGS-B', start = None, maxeval = 5000, include_transformed = False, **kwargs):
    """Wrapper for ``pymc.find_MAP`` which suppresses the deprecation warning in PyMC v6."""

    new_find_MAP_params = "initvals" in inspect.signature(pm.find_MAP).parameters  # True only on PyMC >= 6.4
    
    if new_find_MAP_params:
        # stupid renaming of variables for new pymc versions
        map_idata = pm.find_MAP(*args, method = method, initvals=start, maxiter = maxeval, jitter = False, return_inferencedata=True,
                                idata_kwargs = {"include_transformed":include_transformed}, **kwargs)
        # Have to handle an inference object from an optimisation because pymc devs are annoying
        # The MAP point is stored as a single chain with a single draw
        posterior = map_idata["posterior"]
        map_estimate = {}
        for name, val in posterior.data_vars.items():
            val = np.asarray(val.values[0, 0])
            if name in start:
                val = val.reshape(np.shape(start[name]))
            map_estimate[name] = val
    else:
        # beautiful older interface
        map_estimate = pm.find_MAP(*args, method = method, start=start, maxeval = maxeval, jitter = False,
                                include_transformed = include_transformed, **kwargs)

    
    return map_estimate

def _flatten(val):
    """Flattens a parameter (PyMC tensor, or a fixed float/NumPy/JAX array of any shape) into a 1D tensor
    in row-major order, matching how ``jax.flatten_util.ravel_pytree`` flattens each leaf of a PyTree."""
    if not isinstance(val, TensorVariable):
        val = tens.as_tensor_variable(np.asarray(val))
    return val.ravel()


def LuasPyMC(name, gp = None, var_dict = None, Y = None, likelihood_fn = None, jit = True):
    """PyMC extension which can by used with a luas.GP object for log-likelihood calculations.
    
    Args:
        name (str): Name of observable storing likelihood values e.g. "log_like".
        gp (object): The ``GP`` object used for log likelihood calculations.
        var_dict (PyTree): A PyTree of parameter values for calculating the log likelihood.
        Y (JAXArray): Data values being fit.
        likelihood_fn (Callable, optional): Can specify a different log likelihood function other than the default of
            ``GP.logP_stored``. Needs to take the same inputs and give the same order of outputs as ``GP.logP_stored``.
        jit (bool, optional): Whether to jit compile the likelihood function, as ``PyMC`` does not require the
            log likelihood to be jit compiled. Defaults to True.
    """
    
    # Default to using the log posterior method of the gp object which can make use of stored decompositions
    # This can save significant time if blocked Gibbs sampling or if some hyperparameters are being fixed.
    if likelihood_fn is None:
        likelihood_fn = gp.logP_stored

    # PyMC requires an array of parameters while the gp object requires a PyTree/dictionary of parameter inputs
    # We therefore use the dictionary of variables to construct a make_p_dict function which can convert
    # an array of parameters into a PyTree
    
    # First copy the parameters which may include fixed NumPy arrays
    dict_to_build = deepcopy(var_dict)
    
    # Loop through each parameter and if it's a TensorVariable (i.e. a parameter being sampled by PyMC)
    # replace it with a NumPy array of the same shape. This can be any shape (a scalar, 1D or 2D array etc.)
    for par in var_dict.keys():
        if isinstance(var_dict[par], TensorVariable):
            tensor_shape = tuple(s for s in var_dict[par].shape.eval())
            dict_to_build[par] = np.zeros(tensor_shape)

    # Now that we have a PyTree with the right dimensions as the input variables
    # we can use jax's ravel_pytree function to create a function which will convert
    # an array of parameters into a PyTree of parameters in a deterministic order
    test_arr, make_p_dict = ravel_pytree(dict_to_build)

    # This is the likelihood function which can take an array of parameters and send to the
    # likelihood function as a PyTree of input parameters
    logP_fn = lambda p_arr, stored_values: likelihood_fn(make_p_dict(p_arr), Y, stored_values)

    # Generate a likelihood function which will return the value of the log-likelihood
    # as well as the gradients of each parameter. 
    # The default likelihood function gp.logP_stored also returns an auxillary output
    # which is includes the stored decomposition of the covariance matrix.
    value_and_grad_logP_fn = jax.value_and_grad(logP_fn, has_aux = True)
    if jit:
        value_and_grad_logP_fn = jax.jit(value_and_grad_logP_fn)
        
    # Defines the PyMC wrapper object for log-likelihood functions written in JAX
    logP_pymc = LuasPyMCWrapper(value_and_grad_logP_fn)
    
    # Need to sort the variables in the correct order for make_p_dict to construct the right PyTree
    par_keys_ordered, par_values_ordered = order_dict(var_dict)
    
    # Makes the array of parameter inputs PyMC will sample. Each parameter is flattened (in the same
    # row-major order as ravel_pytree) so parameters can be scalars, 1D arrays, 2D arrays etc.
    p_pymc = tens.concatenate([_flatten(val) for val in par_values_ordered])
    
    # Returns the log likelihood using the custom potential provided by PyMC
    return pm.Potential(name, logP_pymc(p_pymc))


def build_pymc_model(gp, p_vary, Y, ordered_param_list, p_fixed = {}, param_bounds = {}):
    with pm.Model() as model:
        var_dict = deepcopy(p_fixed)

        for par in ordered_param_list:
            if par in param_bounds.keys():
                var_dict[par] = pm.Uniform(par, lower=param_bounds[par][0], upper=param_bounds[par][1], shape=p_vary[par].shape)
            else:
                var_dict[par] = pm.Flat(par, shape=p_vary[par].shape)
        LuasPyMC("log_like", gp = gp, var_dict = var_dict, Y = Y)
        
    return model, var_dict


    
class LuasPyMCWrapper(Op):
    """Wrapper for log-likelihood calculations for luas to work with ``PyMC``.
    Taken from an old ``PyMC`` tutorial 'How to wrap a JAX function for use in PyMC'
    (https://www.pymc.io/projects/examples/en/latest/howto/wrapping_jax_function.html).
    
    """
    default_output = 0

    def __init__(self, value_and_grad_logP_fn):
    
        self.stored_values = {} # Stores the decomposition of the covariance matrix
        
        # function which returns the value and the gradients of the log likelihood
        self.v_and_g_logP = value_and_grad_logP_fn
        
        
    def make_node(self, params):
        
        inputs = [tens.as_tensor_variable(params)]
        
        # We now have one output for the function value, and one output for each gradient
        outputs = [tens.dscalar()] + [inp.type() for inp in inputs]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        
        # Returns the value of the log likelihood, stored decomposition of the covariance matrix
        # and the gradients of the log likelihood in a single function call
        (result, self.stored_values), grad_result = self.v_and_g_logP(*inputs, self.stored_values)
        
        outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(grad_result, dtype=node.outputs[1].dtype)

    def pullback(self, inputs, outputs, cotangents):
        # PyTensor >= 3 (PyMC v6)
        return [cotangents[0] * outputs[1]]

    def L_op(self, inputs, outputs, output_gradients):
        # PyTensor < 3 (PyMC v5)
        return self.pullback(inputs, outputs, output_gradients)
    
