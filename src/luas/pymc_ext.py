import numpy as np
import jax
from jax.flatten_util import ravel_pytree
from .jax_convenience_fns import order_dict
import pymc as pm

pymc_version = int(pm.__version__[0])

if pymc_version == 4:
    import aesara.tensor as tens
    from aesara.graph import Apply, Op
elif pymc_version == 5:
    import pytensor.tensor as tens
    from pytensor.graph import Apply, Op
else:
    raise Exception(f"PyMC Version {pymc_version} not currently supported. Supported versions are version 4 and version 5.")

__all__ = [
    "LuasGP",
    "LuasPyMCWrapper",
]

def LuasGP(name, gp = None, var_dict = None, start = None, Y = None, jit = True):
    """PyMC extension which can by used with a luas.GPClass.GP object for log-likelihood calculations.
    
    """

    p_arr, make_p = ravel_pytree(start)
    logP_fn = lambda p_arr, storage_dict: gp.logP_stored(make_p(p_arr), Y, storage_dict)

    if jit:
        value_and_grad_logP_fn = jax.jit(jax.value_and_grad(logP_fn, has_aux = True))
    else:
        value_and_grad_logP_fn = jax.value_and_grad(logP_fn, has_aux = True)
        
    logP_pymc = LuasPyMCWrapper(value_and_grad_logP_fn)
    
    par_keys_ordered, par_values_ordered = order_dict(var_dict)
    p_pymc = pm.math.concatenate(par_values_ordered)
    
    return pm.Potential(name, logP_pymc(p_pymc))

    
class LuasPyMCWrapper(Op):
    """Wrapper for log-likelihood calculations used by LuasGP which depending on the version of PyMC 
    uses either Aesara or PyTensor. Taken from the PyMC tutorial "How to wrap a JAX function for use in PyMC"
    (https://www.pymc.io/projects/examples/en/latest/howto/wrapping_jax_function.html).
    
    """
    default_output = 0

    def __init__(self, value_and_grad_logP_fn):
    
        self.storage_dict = {}
        self.v_and_g_logP = value_and_grad_logP_fn
        
        
    def make_node(self, params):
        
        inputs = [tens.as_tensor_variable(params)]
        # We now have one output for the function value, and one output for each gradient
        outputs = [tens.dscalar()] + [inp.type() for inp in inputs]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
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