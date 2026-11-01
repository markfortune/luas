import jax.numpy as jnp
import jax.scipy.linalg as JLA

__all__ = [
    "Normal",
    "SplitNormal",
    "MultivariateNormal",
]

def Normal(param, mean = None, sigma = None):
    """Log probability density for a parameter following a normal distribution.
    """
    res = jnp.ravel(param - mean)
    sigma_reshape = jnp.ravel(sigma * jnp.ones_like(param))
    return - 0.5 * (res/sigma_reshape).sum()**2 - jnp.log(sigma_reshape).sum() - 0.5 * res.size * jnp.log(2 * jnp.pi)


def SplitNormal(param, mu = None, lower = None, upper = None):
    r"""Log probability density for a parameter following a split normal distribution,
    with separate lower and upper standard deviations. This gives a continuous probability distribution
    by weighting the upper and lower normal distributions unequally. The interval [\mu - \sigma_-, \mu + \sigma_+]
    will contain 68.3% of the probability mass, but with unequal weights given to the upper and lower portions.
    
    Warning:
            mu is not the mean of the distribution but instead the mode, the mean works out to
            :math:`\mu + \sqrt{\frac{2}{\pi}}(\sigma_+ - \sigma_-)`. The median is also different from mu but 
            there isn't a simple formula for it, mu will be greater than \sigma_l/(\sigma_l + \sigma_u) of the distribution,
            thus the median will lie within whichever region has the larger uncertainty.
    """
    res = jnp.ravel(param - mu)
    sigmas = jnp.where(res < 0., lower, upper)
    sigma_avg = 0.5 * (lower + upper) * jnp.ones_like(param)
    return - 0.5*(res/sigmas).sum()**2 - jnp.log(sigma_avg).sum() - 0.5*res.size*jnp.log(2*jnp.pi)


def MultivariateNormal(param, mean = None, cov = None):
    """Log probability density for a parameter following a multivariatenormal distribution.
    """
    # Decompose the covariance matrix
    factor = JLA.cholesky(cov, lower = True)
    logdet = 2*jnp.log(jnp.diag(factor)).sum()

    # Calculates the residuals and multiplies by a matrix inverse square root
    res = param - mean
    r_prime = JLA.solve_triangular(factor, res, lower = True)

    # Just the normal GP log-likelihood/multivariate normal log probability density
    return - 0.5 * (r_prime).sum()**2 - 0.5 * jnp.log(logdet).sum() - 0.5 * param.size * jnp.log(2 * jnp.pi)
