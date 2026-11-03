import numpy as np
import scipy
import jax
import jax.numpy as jnp
from luas.kernels.base import evaluate_kernel, squared_exp_calc, cosine_calc
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
from luas.kernels.covtype import Outer, General, CovType
from luas.kernels.householder import HouseholderProduct
import luas.kernels.tinygp_ext
from luas.stable_gradients import stable_eigh

from luas.kernels.diagonal import KroneckerDelta, Noise


class LowRank(General):
    def __init__(self, kf, hp = {}, V = None, rank = None, params = None):
        self.kf = kf
        self.hp = hp
        self.fixed_rank = rank
        self.params = params
        self.V = V

        if rank is None:
            raise Exception("Must specify rank of LowRank term")

    def eigendecomp(self, x, wn = True, full = True, idx = None, **kwargs):
        if self.V is not None:
            lam = jnp.concatenate([jnp.zeros(x.shape[-1] - self.fixed_rank), (self.V**2).sum(0)])

            # Need to optimise better, matmuls can be much faster!
            V = jnp.zeros((x.shape[-1], x.shape[-1]))
            V = V.at[:, -self.fixed_rank:].set(self.V)
            return lam, V
        else:
            if not full:
                K = self.evaluate(x, x, wn = wn, full = False,
                                row_idx = idx, col_idx = idx, **kwargs)
            else:
                K = self.evaluate(x, x, wn = wn, full = True, **kwargs)

            # ideally there should be a faster way to do this than calculating a full eigen
            return stable_eigh(K)
    

    def rank(self, x):
        return self.fixed_rank

    def scale(self, c):
        return LowRank(kf = lambda hp, x1, x2, **kwargs: c*self.evaluate(x1, x2, **kwargs), rank = self.fixed_rank, use_shared_memory=self.use_shared_memory)
    
    def __add__(self, other):
        if isinstance(other, LowRank):
            return LowRank(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs),
                           rank = self.fixed_rank + other.fixed_rank, use_shared_memory=self.use_shared_memory and other.use_shared_memory)
        if isinstance(other, Outer):
            return LowRank(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs),
                           rank = self.fixed_rank + 1, use_shared_memory=self.use_shared_memory)
        # elif isinstance(other, CovType):
        #     return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))
        elif isinstance(other, (jax.Array, np.ndarray)) or is_scalar(other):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other)
        else:
            raise Exception("Addition not implemented")
        
    def transform_with_inv_sqrt(self, matrix_sqrt_fn, x):
        if self.V is not None:
            return self
        else:
            def kf_transf(hp, x1, x2, **kwargs):
                    K_eval = self.evaluate(x1, x2, **kwargs)
                    K_prime = matrix_sqrt_fn(K_eval, transpose=0)
                    K_tilde = matrix_sqrt_fn(K_prime.T, transpose=0)
                    
                    return K_tilde

            return LowRank(kf_transf, rank = self.fixed_rank, use_shared_memory=self.use_shared_memory)


    def evaluate(self, x1, x2, **kwargs):
        return self.kf(self.hp, x1, x2, **kwargs)
    
    def matmul(self, x1, x2, R, **kwargs):
        return self.evaluate(x1, x2, **kwargs) @ R


    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other.evaluate(x1, x2, **kwargs))
        elif is_scalar(other):
            return self.scale(other)
        elif isinstance(other, jax.Array) or isinstance(other, np.ndarray):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other)
        else:
            raise Exception("Not implemented")



def Constant(sigma: JAXArray, **kwargs) -> JAXArray:
    r"""Low-rank constant covariance term.

    Args:
        sigma (JAXArray): Constant standard deviation amplitude.

    Returns:
        JAXArray: ``Outer`` covariance component.
    """
    
    return Outer(alpha = sigma)


def Amplitude(alpha: JAXArray, **kwargs) -> JAXArray:
    r"""Low-rank linear covariance term.

    Args:
        alpha (JAXArray): Coefficient vector defining the outer-product term.

    Returns:
        JAXArray: ``Outer`` covariance component.
    """
    
    return Outer(alpha = alpha)


def MixingMat(V: JAXArray, **kwargs) -> JAXArray:
    r"""Low-rank linear covariance term.

    Args:
        alpha (JAXArray): Coefficient vector defining the outer-product term.

    Returns:
        JAXArray: ``Outer`` covariance component.
    """
    return LowRank(lambda hp, x, y, **kwargs: V @ V.T, rank = V.shape[1], V = V, **kwargs)


def SquaredExp(scale: Scalar, axes: JAXArray | None = None, rank = None, **kwargs) -> JAXArray:
    r"""Low-rank squared-exponential (RBF) kernel factory.

    .. math::

        k(x,y)=\exp\left(-\frac{\|x-y\|^2}{2\,\mathrm{scale}^2}\right)

    Args:
        scale (Scalar): Length scale.
        axes (JAXArray | None, optional): Optional axes selector.
        rank (optional): Target low-rank approximation rank.
        **kwargs: Extra keyword arguments forwarded to ``LowRank``.

    Returns:
        JAXArray: ``LowRank`` covariance object.
    """
    
    return LowRank(lambda hp, x, y, **kwargs: evaluate_kernel(squared_exp_calc, x, y, scale), rank = rank, **kwargs)
ExpSquared = SquaredExp


def Cosine(period: Scalar, **kwargs) -> JAXArray:
    r"""Low-rank cosine periodic kernel factory.

    .. math::

        k(x,y)=\cos\left(2\pi\frac{\|x-y\|}{\mathrm{period}}\right)

    Args:
        period (Scalar): Period.
        **kwargs: Extra keyword arguments forwarded to ``LowRank``.

    Returns:
        JAXArray: ``LowRank`` covariance object.
    """

    return LowRank(lambda hp, x, y, **kwargs: evaluate_kernel(cosine_calc, x, y, period), rank = 2, **kwargs)


def ExpSineSquaredApprox(gamma: Scalar, period: Scalar, order: int, sigma: Scalar = 1., **kwargs):


    # if order is None:
    #     periodic_scale = jnp.sqrt(2/gamma)
    #     max_order = jax.lax.cond(periodic_scale < 1/6, lambda _: 16, lambda _: (jnp.floor(4. * periodic_scale**-0.8)).astype("int"), periodic_scale)
    #     order = jax.lax.cond(periodic_scale > 1., lambda _: 4, lambda _: max_order, periodic_scale) - 1

    kernel_quasi = luas.kernels.tinygp_ext.ExpSineSquaredApprox(gamma = gamma, period = period, sigma = sigma, order = order)
    rank = 1 + 2 * kernel_quasi.order
    
    return LowRank(lambda hp, x1, x2, **kwargs: kernel_quasi(x1, x2), rank = rank, **kwargs)
