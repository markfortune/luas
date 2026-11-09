import numpyro
import numpyro.distributions as dist
from copy import deepcopy
import jax.numpy as jnp

__all__ = [
    "LuasNumPyro",
]


def build_numpyro_model(gp, p_vary, Y, ordered_param_list, p_fixed = {}, param_bounds = {}):

    def model(Y):
        # Makes of copy of any parameters to be kept fixed during sampling
        var_dict = deepcopy(p_fixed)

        for par in ordered_param_list:
            if par in param_bounds.keys():
                var_dict[par] = numpyro.sample(par, dist.Uniform(low = param_bounds[par][0], high = param_bounds[par][1]))
            else:
                var_dict[par] = numpyro.sample(par, dist.ImproperUniform(dist.constraints.real, (), event_shape = jnp.shape(p_vary[par])))
        
        numpyro.sample("log_like", LuasNumPyro(gp = gp, var_dict = var_dict), obs = Y)

    return model


class LuasNumPyro(dist.Distribution):
    """Custom ``NumPyro`` distribution which allows a ``luas.GP`` object to be used with
    ``NumPyro``.
    
    Args:
        gp (object): The ``GP`` object used for log likelihood calculations.
        var_dict (PyTree): A PyTree of parameter values for calculating the log likelihood
        likelihood_fn (Callable, optional): Can specify a different log likelihood function than the default of
            ``GP.logP`` i.e. ``GP.logP_hessianable`` may be used which is more
            numerically stable if performing hessian/second order derivative calculations.
        
    """

    def __init__(
        self,
        gp=None, 
        var_dict=None,
        likelihood_fn=None,
        validate_args=None,
    ):
        
        self.gp = gp
        self.p = var_dict
        
        # If using NumPyro functionality which makes use of the hessian of the log likelihood
        # (i.e. numpyro.infer.autoguide.AutoLaplaceApproximation) then might be useful to set
        # likelihood_fn = gp.logP_hessianable as the default log likelihood is faster but not
        # as numerically stable for second order derivatives.
        if likelihood_fn is None:
            self.logP_fn = self.gp.logP
        else:
            self.logP_fn = likelihood_fn
        
        super().__init__(
            batch_shape = (),
            event_shape = tuple(self.gp.data_shape),
            validate_args=validate_args,
        )

    @dist.util.validate_sample
    def log_prob(self, Y):
        
        return self.logP_fn(self.p, Y)
    
    @property
    def support(self):
        # The data are real valued, and the whole (N_l, N_t) array is one event
        return dist.constraints.independent(dist.constraints.real, len(self.event_shape))
