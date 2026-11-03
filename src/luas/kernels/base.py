import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import gammaln
from typing import Callable
from luas.luas_types import JAXArray, Scalar, PyTree, is_scalar
import luas.kernels.covtype as covtype

__all__ = [
    "KroneckerDelta",
    "Noise",
    "evaluate_kernel",
    "distanceL1",
    "distanceL2Sq",
    "Exp",
    "SquaredExp",
    "Matern32",
    "Matern52",
    "MaternHalfInt",
    "RationalQuadratic",
    "ExpSineSquared",
    "Cosine",
    "PoweredExp",
    "Fixed",
    "Custom",
]

from luas.kernels.diagonal import KroneckerDelta, Noise

def evaluate_kernel(kernel_fn: Callable, x: JAXArray, y: JAXArray, *args, axes = None) -> JAXArray:
    r"""Build a pairwise covariance matrix using ``jax.vmap``.

    For a pointwise kernel :math:`k`, this computes :math:`K_{ij}=k(x_i,y_j)`.

    Args:
        kernel_fn (Callable): Pointwise kernel function.
        x (JAXArray): Input coordinates for the first axis.
        y (JAXArray): Input coordinates for the second axis.
        *args: Additional positional parameters forwarded to ``kernel_fn``.
        axes (JAXArray | None, optional): Optional subset of dimensions to evaluate.

    Returns:
        JAXArray: Pairwise covariance matrix.
    """
    if axes is not None:
        K = vmap(lambda x1: vmap(lambda y1: kernel_fn(x1, y1, *args), in_axes = -1)(y[axes, :]), in_axes = -1)(x[axes, :])
    else:
        K = vmap(lambda x1: vmap(lambda y1: kernel_fn(x1, y1, *args), in_axes = -1)(y), in_axes = -1)(x)
    return K
    

def distanceL1(x: JAXArray, y: JAXArray, scale: Scalar) -> JAXArray:
    r"""Evaluates the L1 norm of two input vectors divided by a length scale.
    
    .. math::

        L1(x, y) = \frac{|x - y|}{L}
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        scale (Scalar): Length scale
        
    Returns:
        Scalar: L1 norm between two input vectors
        
    """
    
    return jnp.sum(jnp.abs(x - y)/scale)


def distanceL2Sq(x: JAXArray, y: JAXArray, scale: Scalar) -> JAXArray:
    r"""Evaluates the Squared L2 norm of two input vectors divided by the length scale ``L``.
    
    .. math::

        L2^2(x, y) = \frac{|x - y|^2}{L^2}
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        scale (Scalar): Length scale
        
    Returns:
        Scalar: L2 norm between two input vectors
        
    """
    
    return jnp.sum(jnp.square(x - y)/scale**2)
    


def Exp(scale: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Exponential (Matérn-1/2) kernel factory.

    .. math::

        k(x,y)=\sigma^2\exp\left(-\frac{\|x-y\|}{\mathrm{scale}}\right)

    Args:
        scale (Scalar): Length scale :math:`L`.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: A ``covtype.General`` covariance object.
    """
    
    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(exp_calc, x, y, scale))

def exp_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by ``luas.kernels.exp`` to evaluate the Exponential kernel function. 
    
    """

    delta_t = distanceL1(x, y, L).sum()
    return jnp.exp(-delta_t)

    
def SquaredExp(scale: Scalar, sigma: Scalar = 1., axes: JAXArray | None = None) -> JAXArray:
    r"""Squared exponential (RBF) kernel.

    .. math::

        k(x,y)=\sigma^2\exp\left(-\frac{\|x-y\|^2}{2\,\mathrm{scale}^2}\right)

    Args:
        scale (Scalar): Length scale :math:`L`.
        sigma (Scalar, optional): Kernel amplitude.
        axes (JAXArray | None, optional): Optional axes selector forwarded internally.

    Returns:
        JAXArray: A ``covtype.General`` covariance object.
    """
    
    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(squared_exp_calc, x, y, scale))
ExpSquared = SquaredExp
    
def squared_exp_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by ``luas.kernels.squared_exp`` to evaluate the squared exponential kernel function. 
    
    """

    tau_sq = distanceL2Sq(x, y, L)
    return jnp.exp(-0.5 * tau_sq.sum())


def Matern32(scale: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Matérn-3/2 kernel factory.

    .. math::

        k(x,y)=\sigma^2\left(1+\sqrt{3}r\right)e^{-\sqrt{3}r},\quad r=\frac{\|x-y\|}{\mathrm{scale}}

    Args:
        scale (Scalar): Length scale.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: A ``covtype.General`` covariance object.
    """

    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(matern32_calc, x, y, scale))


def matern32_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by matern32 to evaluate the Matern 3/2 kernel function. 
    
    """

    delta_t = jnp.sqrt(3)*distanceL1(x, y, L).sum()
    return (1+delta_t)*jnp.exp(-delta_t)
    

def Matern52(scale: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Matérn-5/2 kernel factory.

    .. math::

        k(x,y)=\sigma^2\left(1+\sqrt{5}r+\frac{5}{3}r^2\right)e^{-\sqrt{5}r},\quad r=\frac{\|x-y\|}{\mathrm{scale}}

    Args:
        scale (Scalar): Length scale.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: A ``covtype.General`` covariance object.
    """
    
    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(matern52_calc, x, y, scale))

def matern52_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by matern52 to evaluate the Matern 5/2 kernel function. 
    
    """

    delta_t = jnp.sqrt(5)*distanceL1(x, y, L).sum()
    return (1+delta_t+jnp.square(delta_t)/3)*jnp.exp(-delta_t)
    

def MaternHalfInt(scale: Scalar, double_nu: int, sigma: Scalar = 1.) -> JAXArray:
    r"""Half-integer Matérn kernel family.

    Constructs a Matérn covariance with :math:`\nu = (\text{double_nu})/2`,
    where ``double_nu`` is expected to be an odd positive integer for
    half-integer :math:`\nu` values.

    Args:
        scale (Scalar): Length scale parameter.
        double_nu (int): Twice the Matérn smoothness parameter :math:`\nu`.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: A ``covtype.General`` covariance object.
    """
    p = (double_nu - 1)//2
    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(matern_half_int_calc, x, y, scale, p))


def matern_half_int_calc(x: JAXArray, y: JAXArray, scale: Scalar, p: int) -> JAXArray:
    """Evaluate the half-integer Matérn kernel polynomial form.

    Internal helper used by :func:`MaternHalfInt`.
    """

    f = jnp.sqrt(2*p + 1)
    delta_t = f*distanceL1(x, y, scale).sum()

    # const = jax.scipy.special.factorial(p)/jax.scipy.special.factorial(2*p)
    # poly_term = 0
    # for i in jnp.arange(p+1):
        # bin_coeff = jax.scipy.special.factorial(p + i)/(jax.scipy.special.factorial(i)*jax.scipy.special.factorial(p-i))
        # poly_term += bin_coeff * (2*delta_t)**(p-i)

    const = gammaln(p + 1) - gammaln(2*p + 1)

    ind = jnp.arange(p+1)
    poly_term = jnp.exp(gammaln(p + ind + 1) - gammaln(ind + 1) - gammaln(p - ind + 1))
    poly_term *= (2*delta_t)**(p-ind)

    return jnp.exp(const - delta_t) * poly_term.sum()

    
def RationalQuadratic(scale: Scalar, alpha: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Rational quadratic kernel factory.

    .. math::

        k(x,y)=\sigma^2\left(1+\frac{\|x-y\|^2}{2\alpha\,\mathrm{scale}^2}\right)^{-\alpha}

    Args:
        scale (Scalar): Length scale.
        alpha (Scalar): Scale-mixture parameter.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: A ``covtype.General`` covariance object.
    """
    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(rational_quadratic_calc, x, y, scale, alpha))


def rational_quadratic_calc(x: JAXArray, y: JAXArray, L: Scalar, alpha: Scalar) -> JAXArray:
    """Function used by rational_quadratic to evaluate the rational quadratic kernel function. 
    
    """

    tau_sq = distanceL2Sq(x, y, L).sum()
    return (1. + 0.5*tau_sq/alpha)**(-alpha)


def ExpSineSquared(gamma: Scalar, period: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Exponentiated-sine-squared periodic kernel factory.

    .. math::

        k(x,y)=\sigma^2\exp\left[-\gamma\,\sin^2\!\left(\pi\frac{x-y}{\mathrm{period}}\right)\right]

    Args:
        gamma (Scalar): Periodic smoothness parameter.
        period (Scalar): Period.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: A ``covtype.General`` covariance object.
    """
    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(exp_sine_squared_calc, x, y, gamma, period))


def exp_sine_squared_calc(x: JAXArray, y: JAXArray, gamma: Scalar, period: Scalar) -> JAXArray:
    """Function used by exp_sine_squared to evaluate the exponential sine squared kernel function.
    
    """

    sine_sq = jnp.square(jnp.sin(jnp.pi*(x - y)/period))
    return jnp.exp(-gamma * sine_sq).sum()
    

def Cosine(period: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Cosine periodic kernel factory.

    .. math::

        k(x,y)=\sigma^2\cos\left(2\pi\frac{\|x-y\|}{\mathrm{period}}\right)

    Args:
        period (Scalar): Period.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: A ``covtype.General`` covariance object.
    """

    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(cosine_calc, x, y, period))


def cosine_calc(x: JAXArray, y: JAXArray, period: Scalar) -> JAXArray:
    """Function used by cosine to evaluate the cosine kernel function. 
    
    """
        
    delta_t = distanceL1(x, y, period).sum()
    return jnp.cos(2*jnp.pi*delta_t)


def PoweredExp(scale: Scalar, k: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Powered exponential kernel factory.

    Args:
        scale (Scalar): Length scale.
        k (Scalar): Exponent parameter.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: A ``covtype.General`` covariance object.
    """

    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(powered_exp_calc, x, y, scale, k))
    
    
def powered_exp_calc(x: JAXArray, y: JAXArray, L: Scalar, k: Scalar) -> JAXArray:
    """Function used by powered_exp to evaluate the powered exponential kernel function. 
    """

    d = distanceL1(x, y, L)
    zero_dist = d == 0.0
    d_safe = jnp.where(zero_dist, 1.0, d) # avoid numerically unstable gradients at 0^k
    val = jnp.exp(-d_safe ** k)
    return jnp.where(zero_dist, 1.0, val) 



def Constant(sigma: JAXArray) -> JAXArray:
    r"""Constant covariance term represented as an outer-product component.

    This returns a ``covtype.Outer`` object with amplitude equivalent to a
    constant covariance level.

    Args:
        sigma (JAXArray): Constant standard deviation.

    Returns:
        JAXArray: A ``covtype.Outer`` covariance object.
    """
    
    return covtype.Outer(alpha = sigma)


def Amplitude(alpha: JAXArray, sigma: Scalar = 1.) -> JAXArray:
    r"""Rank-1 covariance matrix represented as an outer-product sigma^2 * (alpha @ alpha^T)

    This returns a ``covtype.Outer`` object parameterized by ``sigma * alpha``.
    It is useful when building separable or low-rank covariance terms.

    Args:
        alpha (JAXArray): Feature/basis vector used to construct the outer product.
        sigma (Scalar, optional): Overall amplitude scale.

    Returns:
        JAXArray: A ``covtype.Outer`` covariance object.
    """
    
    return covtype.Outer(alpha = sigma * alpha)


def Custom(kf: Callable, hp: PyTree, params = None, kf_args = (), kf_kwargs = {}) -> JAXArray:
    r"""Wrap a user-provided kernel callable into a ``covtype.General`` object.

    Args:
        kf (Callable): Kernel callable with signature compatible with
            ``kf(hp, x1, x2, *kf_args, **kf_kwargs, **kwargs)``.
        hp (PyTree): Hyperparameter pytree passed to ``kf``.
        params: Optional parameter metadata forwarded to ``covtype.General``.
        kf_args (tuple, optional): Positional args forwarded to ``kf``.
        kf_kwargs (dict, optional): Keyword args forwarded to ``kf``.

    Returns:
        JAXArray: A ``covtype.General`` covariance object.
    """
    custom_kf = lambda _, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs: kf(hp, x1, x2,
                                                                                            *kf_args, **kf_kwargs,
                                                                                            **kwargs)
    return covtype.General(custom_kf, params = params)


def Fixed(cov_mat: JAXArray, params = None, ignore_idx = False, sigma: Scalar = 1.) -> JAXArray:

    def kf(hp, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs):

        if not full and not ignore_idx:
            return sigma**2 * cov_mat[jnp.ix_(row_idx, col_idx)]
        else:
            return sigma**2 * cov_mat
        
    return covtype.General(kf, params = params)