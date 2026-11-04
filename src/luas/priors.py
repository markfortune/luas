"""Log prior probability densities, for building the ``logPrior`` function of a :class:`luas.GP`.

Each function returns the log probability density summed over every element of the parameter,
so priors on several parameters can simply be added together::

    def logPrior(p, X):
        logP = luas.priors.Normal(p["u1"], 0.3, 0.1)
        logP += luas.priors.Normal(p["u2"], 0.2, 0.1)
        logP += luas.priors.SplitNormal(p["a"], 8.2, 0.2, 0.3)
        return logP

    gp = luas.GP(p, X, kf = kf, mf = mf, logPrior = logPrior)

All of these functions are written in ``JAX``, so they can be JIT compiled and differentiated
along with the rest of the log posterior.
"""
import jax.numpy as jnp
import jax.scipy.linalg as JLA

__all__ = [
    "Normal",
    "SplitNormal",
    "MultivariateNormal",
]

def Normal(param, mean = None, sigma = None):
    r"""Log probability density of a normal (Gaussian) prior.

    .. math::

        \log p(x) = -\frac{(x - \mu)^2}{2\sigma^2} - \log \sigma - \frac{1}{2}\log 2\pi

    summed over every element of ``param``.

    Example:
        A prior on a limb darkening coefficient of :math:`0.3 \pm 0.1`::

            logP = luas.priors.Normal(p["u1"], 0.3, 0.1)

    Args:
        param (JAXArray | Scalar): The parameter value(s).
        mean (JAXArray | Scalar): The mean :math:`\mu`, either a scalar or an array with the
            same shape as ``param``.
        sigma (JAXArray | Scalar): The standard deviation :math:`\sigma`, either a scalar or an
            array with the same shape as ``param``.

    Returns:
        Scalar: The log probability density, summed over every element of ``param``.

    """
    res = jnp.ravel(param - mean)
    sigma_reshape = jnp.ravel(sigma * jnp.ones_like(param))
    return - 0.5 * jnp.square(res/sigma_reshape).sum() - jnp.log(sigma_reshape).sum() - 0.5 * res.size * jnp.log(2 * jnp.pi)


def SplitNormal(param, mu = None, lower = None, upper = None):
    r"""Log probability density of a split normal prior, which has separate lower and upper standard
    deviations. Useful for asymmetric constraints such as :math:`a = 8.2^{+0.3}_{-0.2}`.

    .. math::

        p(x) = \sqrt{\frac{2}{\pi}} \frac{1}{\sigma_- + \sigma_+}
        \begin{cases}
            \exp\left(-\frac{(x - \mu)^2}{2\sigma_-^2}\right) & x < \mu \\
            \exp\left(-\frac{(x - \mu)^2}{2\sigma_+^2}\right) & x \geq \mu
        \end{cases}

    The two halves are two normal distributions with different widths, weighted so that the density
    is continuous at :math:`\mu`. The interval :math:`[\mu - \sigma_-, \mu + \sigma_+]` contains 68.3%
    of the probability, as for a normal distribution.

    Warning:
        ``mu`` is the mode of the distribution, not the mean or median. The mean is
        :math:`\mu + \sqrt{2/\pi}\,(\sigma_+ - \sigma_-)`. A fraction :math:`\sigma_-/(\sigma_- + \sigma_+)`
        of the probability lies below :math:`\mu`, so the median lies on the side with the larger
        uncertainty.

    Example:
        A prior of :math:`a = 8.2^{+0.3}_{-0.2}`::

            logP = luas.priors.SplitNormal(p["a"], 8.2, 0.2, 0.3)

    Args:
        param (JAXArray | Scalar): The parameter value(s).
        mu (JAXArray | Scalar): The mode :math:`\mu`, either a scalar or an array with the same shape
            as ``param``.
        lower (JAXArray | Scalar): The lower standard deviation :math:`\sigma_-`, either a scalar or an
            array with the same shape as ``param``.
        upper (JAXArray | Scalar): The upper standard deviation :math:`\sigma_+`, either a scalar or an
            array with the same shape as ``param``.

    Returns:
        Scalar: The log probability density, summed over every element of ``param``.

    """
    res = jnp.ravel(param - mu)
    lower = jnp.ravel(lower * jnp.ones_like(param))
    upper = jnp.ravel(upper * jnp.ones_like(param))
    sigmas = jnp.where(res < 0., lower, upper)
    sigma_avg = 0.5 * (lower + upper)
    return - 0.5*jnp.square(res/sigmas).sum() - jnp.log(sigma_avg).sum() - 0.5*res.size*jnp.log(2*jnp.pi)


def MultivariateNormal(param, mean = None, cov = None):
    r"""Log probability density of a multivariate normal prior, for correlated parameters.

    .. math::

        \log p(\vec{x}) = -\frac{1}{2} (\vec{x} - \vec{\mu})^T \mathbf{C}^{-1} (\vec{x} - \vec{\mu})
        - \frac{1}{2} \log |\mathbf{C}| - \frac{N}{2} \log 2\pi

    for :math:`N` parameter values with mean :math:`\vec{\mu}` and covariance matrix :math:`\mathbf{C}`.
    Calculated using a Cholesky decomposition of :math:`\mathbf{C}`.

    Example:
        A joint prior on two limb darkening coefficients, e.g. using the covariance matrix from a
        previous fit::

            u = jnp.array([p["u1"], p["u2"]])
            logP = luas.priors.MultivariateNormal(u, u_mean, u_cov)

    Args:
        param (JAXArray): The parameter values, of shape ``(N,)``.
        mean (JAXArray | Scalar): The mean :math:`\vec{\mu}`, either a scalar or an array of shape ``(N,)``.
        cov (JAXArray): The covariance matrix :math:`\mathbf{C}`, of shape ``(N, N)``. Must be
            positive definite.

    Returns:
        Scalar: The log probability density.

    """
    # Decompose the covariance matrix
    factor = JLA.cholesky(cov, lower = True)
    logdet = 2*jnp.log(jnp.diag(factor)).sum()

    # Calculates the residuals and multiplies by a matrix inverse square root
    res = param - mean
    r_prime = JLA.solve_triangular(factor, res, lower = True)

    # Just the normal GP log-likelihood/multivariate normal log probability density
    return - 0.5 * jnp.square(r_prime).sum() - 0.5 * logdet - 0.5 * param.size * jnp.log(2 * jnp.pi)
