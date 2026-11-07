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
    r"""A dense covariance matrix with a known rank, e.g. a low-rank approximation to a kernel. Behaves like
    ``covtype.General``, but records its rank so the Kronecker kernels can treat it as a sum of that many
    rank-one terms.

    Args:
        kf (Callable): Kernel function ``kf(hp, x1, x2, **kwargs)`` returning the covariance matrix.
        hp (PyTree, optional): Hyperparameters passed to ``kf``. Defaults to an empty dict.
        rank (int): The rank of the matrix. Required.
        params (optional): Metadata about the parameters the object depends on.

    """
    def __init__(self, kf, hp = {}, rank = None, params = None):
        r"""Creates the object, see the class docstring for the arguments."""
        self.kf = kf
        self.hp = hp
        self.fixed_rank = rank
        self.params = params

        if rank is None:
            raise Exception("Must specify rank of LowRank term")

    def eigendecomp(self, x, wn = True, full = True, idx = None, **kwargs):
        r"""Calculates the eigendecomposition. Currently unoptimised as it calculates the full eigenvalues and eigenvectors.

        Returns:
            (JAXArray, JAXArray): The eigenvalues and a matrix whose columns are the eigenvectors.

        """
        if not full:
            K = self.evaluate(x, x, wn = wn, full = False,
                            row_idx = idx, col_idx = idx, **kwargs)
        else:
            K = self.evaluate(x, x, wn = wn, full = True, **kwargs)

        # ideally there should be a faster way to do this than calculating a full eigen
        return stable_eigh(K)
    

    def rank(self, x):
        r"""Returns the rank given when the object was created."""
        return self.fixed_rank

    def scale(self, c):
        r"""Returns a ``LowRank`` object for :math:`c\mathbf{K}`, with the same rank."""
        return LowRank(kf = lambda hp, x1, x2, **kwargs: c*self.evaluate(x1, x2, **kwargs), rank = self.fixed_rank)
    
    def __add__(self, other):
        r"""Adds another ``LowRank`` or ``Outer`` object, giving a ``LowRank`` object whose rank is the sum of their
        ranks. Adding an array or scalar gives a ``General`` object.

        """
        if isinstance(other, LowRank):
            return LowRank(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs),
                           rank = self.fixed_rank + other.fixed_rank)
        if isinstance(other, Outer):
            return LowRank(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs),
                           rank = self.fixed_rank + 1)
        # elif isinstance(other, CovType):
        #     return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))
        elif isinstance(other, (jax.Array, np.ndarray)) or is_scalar(other):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other)
        else:
            return NotImplemented

    def transform_with_inv_sqrt(self, matrix_sqrt_fn, x, **kwargs):
        r"""Whitens the matrix with an inverse matrix square root, returning
        :math:`\mathbf{L}^{-1}\mathbf{K}\mathbf{L}^{-T}` as a ``LowRank`` object with the same rank. Called by
        ``CovType.inv_sqrt_transform``.

        Args:
            matrix_sqrt_fn (Callable): Function ``matrix_sqrt_fn(R, transpose = 0)`` applying :math:`\mathbf{L}^{-1}`.
            x (JAXArray): Not used.

        Returns:
            LowRank: The transformed covariance object.

        """
        def kf_transf(hp, x1, x2, **kwargs):
                K_eval = self.evaluate(x1, x2, **kwargs)
                K_prime = matrix_sqrt_fn(K_eval, transpose=0)
                K_tilde = matrix_sqrt_fn(K_prime.T, transpose=0)
                
                return K_tilde

        return LowRank(kf_transf, rank = self.fixed_rank)


    def evaluate(self, x1, x2, **kwargs):
        r"""Evaluates the kernel function, ``kf(hp, x1, x2, **kwargs)``."""
        return self.kf(self.hp, x1, x2, **kwargs)
    
    def matmul(self, x1, x2, R, **kwargs):
        r"""Multiplies ``R`` by the dense matrix between ``x1`` and ``x2``."""
        return self.evaluate(x1, x2, **kwargs) @ R


    def __mul__(self, other) -> Kernel:
        r"""Returns a ``General`` object for the element-wise product with another covariance object or array, or a ``LowRank`` object scaled by a scalar."""
        if isinstance(other, CovType):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other.evaluate(x1, x2, **kwargs))
        elif is_scalar(other):
            return self.scale(other)
        elif isinstance(other, jax.Array) or isinstance(other, np.ndarray):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other)
        else:
            return NotImplemented



def Constant(sigma: JAXArray, **kwargs) -> JAXArray:
    r"""Rank-one constant covariance term, a matrix with every entry equal to :math:`\sigma^2`.

    Args:
        sigma (JAXArray): The amplitude :math:`\sigma`.
        **kwargs: Not used.

    Returns:
        JAXArray: ``Outer`` covariance component.
    """
    
    return Outer(alpha = sigma)


def Amplitude(alpha: JAXArray, **kwargs) -> JAXArray:
    r"""Rank-one covariance term :math:`\vec{\alpha}\vec{\alpha}^T`, e.g. a fixed shape along this dimension
    with an amplitude given by the kernel along the other dimension.

    Args:
        alpha (JAXArray): The vector :math:`\vec{\alpha}`.
        **kwargs: Not used.

    Returns:
        JAXArray: ``Outer`` covariance component.
    """
    
    return Outer(alpha = alpha)


def MixingMat(V: JAXArray, **kwargs) -> JAXArray:
    r"""Low-rank covariance term :math:`\mathbf{V}\mathbf{V}^T` from a fixed mixing matrix, i.e. a sum of
    rank-one terms :math:`\sum_i \vec{v}_i\vec{v}_i^T` from the columns of :math:`\mathbf{V}`.

    Args:
        V (JAXArray): The mixing matrix, of shape ``(N, rank)``.
        **kwargs: Extra keyword arguments forwarded to ``LowRank``.

    Returns:
        JAXArray: ``LowRank`` covariance object with rank ``V.shape[1]``.
    """
    return LowRank(lambda hp, x, y, **kwargs: V @ V.T, rank = V.shape[1], **kwargs)


def SquaredExp(scale: Scalar, axes: JAXArray | None = None, rank = None, **kwargs) -> JAXArray:
    r"""Low-rank squared-exponential (RBF) kernel factory.

    .. math::

        k(x,y)=\exp\left(-\frac{\|x-y\|^2}{2l^2}\right)

    Args:
        scale (Scalar): The parameter :math:`l`.
        axes (JAXArray | None, optional): Not used.
        rank (int): The rank the matrix is treated as having. Required. The full squared exponential
            matrix is evaluated, which is numerically low rank for long length scales, and only this many of
            its largest eigenvalues are kept by the Kronecker kernels.
        **kwargs: Extra keyword arguments forwarded to ``LowRank``.

    Returns:
        JAXArray: ``LowRank`` covariance object.
    """
    
    return LowRank(lambda hp, x, y, **kwargs: evaluate_kernel(squared_exp_calc, x, y, scale), rank = rank, **kwargs)
ExpSquared = SquaredExp


def Cosine(period: Scalar, **kwargs) -> JAXArray:
    r"""Low-rank cosine periodic kernel factory.

    .. math::

        k(x,y)=\cos\left(2\pi\frac{\|x-y\|}{P}\right)

    Args:
        period (Scalar): The parameter :math:`P`, the period of the kernel.
        **kwargs: Extra keyword arguments forwarded to ``LowRank``.

    Returns:
        JAXArray: ``LowRank`` covariance object.
    """

    return LowRank(lambda hp, x, y, **kwargs: evaluate_kernel(cosine_calc, x, y, period), rank = 2, **kwargs)


def ExpSineSquaredApprox(gamma: Scalar, period: Scalar, order: int, sigma: Scalar = 1., **kwargs):
    r"""Low-rank approximation to the exponential sine squared periodic kernel.

    .. math::

        k(x,y)\approx\sigma^2\exp\left[-\gamma\,\sin^2\!\left(\pi\frac{x-y}{P}\right)\right]

    The exponential sine squared kernel is approximated by a sum of ``order`` cosine terms (plus a
    constant), following Solin & Särkkä (2014). Larger values of ``gamma`` need a higher ``order``
    to be accurate. The rank of the covariance matrix is ``1 + 2 * order``.

    Args:
        gamma (Scalar): The parameter :math:`\gamma`, which sets how smooth the kernel is within each period.
        period (Scalar): The parameter :math:`P`, the period of the kernel.
        order (int): The number of cosine terms in the approximation.
        sigma (Scalar, optional): Kernel amplitude.
        **kwargs: Extra keyword arguments forwarded to ``LowRank``.

    Returns:
        JAXArray: ``LowRank`` covariance object.
    """


    # if order is None:
    #     periodic_scale = jnp.sqrt(2/gamma)
    #     max_order = jax.lax.cond(periodic_scale < 1/6, lambda _: 16, lambda _: (jnp.floor(4. * periodic_scale**-0.8)).astype("int"), periodic_scale)
    #     order = jax.lax.cond(periodic_scale > 1., lambda _: 4, lambda _: max_order, periodic_scale) - 1

    kernel_quasi = luas.kernels.tinygp_ext.ExpSineSquaredApprox(gamma = gamma, period = period, sigma = sigma, order = order)
    rank = 1 + 2 * kernel_quasi.order
    
    return LowRank(lambda hp, x1, x2, **kwargs: kernel_quasi(x1, x2), rank = rank, **kwargs)
