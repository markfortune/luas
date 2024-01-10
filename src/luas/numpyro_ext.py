import numpyro.distributions as dist
from numpyro.distributions.util import promote_shapes, validate_sample
from jax import lax

__all__ = [
    "LuasGP",
]

class LuasGP(dist.Distribution):
    """Custom NumPyro distribution which allows a luas.GPClass.GP object to be used with
    NumPyro.
    
    Args:
        gp (object): The luas.GPClass.GP object used for log likelihood calculations.
        p (PyTree): A dictionary of parameter values for calculating the log likelihood
        hessian (bool, optional): Whether to use a slower log likelihood function which is more
            numerically stable if performing hessian/second order derivative calculations.
            Defaults to False.
        
    """

    def __init__(
        self,
        gp=None, 
        p=None,
        validate_args=None,
        hessian=False
    ):
        
        self.gp = gp
        self.p = p
        
        # If using NumPyro functionality which makes use of the hessian of the log likelihood
        # (i.e. numpyro.infer.autoguide.AutoLaplaceApproximation) then hessian might need to be set to true
        # as the default log likelihood is faster but not as numerically stable for second order derivatives.
        if hessian:
            self.logP_fn = self.gp.logP_hessianable
        else:
            self.logP_fn = self.gp.logP
        
        super().__init__(
            batch_shape = (),
            event_shape = (self.gp.N_l, self.gp.N_t),
            validate_args=validate_args,
        )

    @dist.util.validate_sample
    def log_prob(self, Y):
        
        return self.logP_fn(self.p, Y)
    
    
    def support(self, Y):
        
        return dist.constraints.real(Y)
