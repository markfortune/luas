import numpy as np
import jax
from jax.flatten_util import ravel_pytree
from .jax_convenience_fns import order_dict
import pymc as pm
from copy import deepcopy

pymc_version = int(pm.__version__[0])

if pymc_version == 4:
    # PyMC v4 uses aesara as a backend
    import aesara.tensor as tens
    from aesara.tensor.var import TensorVariable
    from aesara.graph import Apply, Op
    
elif pymc_version == 5:
    # PyMC v5 uses pytensor as a backend
    import pytensor.tensor as tens
    from pytensor.tensor.var import TensorVariable
    from pytensor.graph import Apply, Op
    
else:
    raise Exception(f"PyMC Version {pymc_version} not currently supported. Supported versions are version 4 and version 5.")

__all__ = [
    "LuasPyMC",
]

def LuasPyMC(name, gp = None, var_dict = None, Y = None, likelihood_fn = None, jit = True):
    """PyMC extension which can by used with a luas.GPClass.GP object for log-likelihood calculations.
    
    Args:
        name (str): Name of observable storing likelihood values e.g. "log_like".
        gp (object): The luas.GPClass.GP object used for log likelihood calculations.
        var_dict (PyTree): A dictionary of parameter values for calculating the log likelihood
        Y (JAXArray): Data values being fit.
        likelihood_fn (Callable, optional): Can specify a different log likelihood function other than the default of
            GP.logP_stored. Needs to take the same inputs and give the same order of outputs as GP.logP_stored.
        jit (bool, optional): Whether to jit compile the likelihood function, as PyMC does not require the
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
    # replace it with a NumPy array of the same size
    for par in var_dict.keys():
        if type(var_dict[par]) == TensorVariable:
            dict_to_build[par] = np.zeros(var_dict[par].shape.eval()[0])

    # Now that we have a PyTree with the right dimensions as the input variables
    # we can use jax's ravel_pytree function to create a function which will convert
    # an array of parameters into a PyTree of parameters in a deterministic order
    test_arr, make_p_dict = ravel_pytree(dict_to_build)

    # This is the likelihood function which can take an array of parameters and send to the
    # likelihood function as a PyTree of input parameters
    logP_fn = lambda p_arr, storage_dict: likelihood_fn(make_p_dict(p_arr), Y, storage_dict)

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
    
    # Makes the array of parameter inputs PyMC will sample
    p_pymc = pm.math.concatenate(par_values_ordered)
    
    # Returns the log likelihood using the custom potential provided by PyMC
    return pm.Potential(name, logP_pymc(p_pymc))

    
class LuasPyMCWrapper(Op):
    """Wrapper for log-likelihood calculations used by LuasGP which depending on the version of PyMC 
    uses either Aesara or PyTensor. Taken from the PyMC tutorial "How to wrap a JAX function for use in PyMC"
    (https://www.pymc.io/projects/examples/en/latest/howto/wrapping_jax_function.html).
    
    """
    default_output = 0

    def __init__(self, value_and_grad_logP_fn):
    
        self.storage_dict = {} # Stores the decomposition of the covariance matrix
        
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
        (result, self.storage_dict), grad_result = self.v_and_g_logP(*inputs, self.storage_dict)
        
        outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(grad_result, dtype=node.outputs[1].dtype)

    def grad(self, inputs, output_gradients):
        
        # The `Op` computes its own gradients, so we call it again.
        value = self(*inputs)
        
        # We hid the gradient outputs by setting `default_update=0`, but we
        # can retrieve them anytime by accessing the `Apply` node via `value.owner`
        grad_result = value.owner.outputs[1]

        return [output_gradients[0] * grad_result]