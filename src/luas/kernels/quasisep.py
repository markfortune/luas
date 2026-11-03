import jax.numpy as jnp
import jax
from jax import vmap
import numpy as np
from typing import Callable
import tinygp
import tinygp.kernels.quasisep
import equinox as eqx

from luas.luas_types import JAXArray, Scalar, PyTree, is_scalar
import luas.kernels.covtype as covtype
import luas.kernels.tinygp_ext
from luas.kernels.tinygp_ext import HandleIdx, ScaledKernel

__all__ = [
    "Exp",
    "Matern32",
    "Matern52",
    "MaternHalfInt",
    "Constant",
    "SHO",
    "Cosine",
    "KroneckerDelta",
    "Noise",
    "SquaredExpApprox",
    "ExpSineSquaredApprox",
    "QuasiperiodicApprox",
]

from luas.kernels.diagonal import KroneckerDelta, Noise

def CustomTinygp(kf_tinygp: Callable, params = None) -> JAXArray:
    r"""Wrap a tinygp-compatible kernel in a quasisep covariance object.

    Args:
        kf_tinygp (Callable): tinygp kernel object/callable.
        params: Optional parameter metadata forwarded to the covariance wrapper.

    Returns:
        JAXArray: A ``covtype.GeneralQuasisepPlusNoise`` covariance object.
    """

    return covtype.GeneralQuasisepPlusNoise(HandleIdx(kf_tinygp), params = params)


def Constant(sigma: JAXArray) -> JAXArray:
    r"""Constant covariance component.

    Args:
        const (JAXArray): Constant standard deviation (must be scalar).

    Returns:
        JAXArray: Outer-product covariance representation.
    """
    assert is_scalar(sigma)

    return covtype.Outer(alpha = sigma)


def Amplitude(alpha: JAXArray, sigma: Scalar = 1., use_block: bool = True, const: None = None) -> JAXArray:
    r"""Rank-1 covariance component.

    If ``const`` is provided, returns a quasisep-plus-noise representation using
    a tinygp Amplitude kernel; otherwise returns an outer-product form.

    Args:
        alpha (JAXArray): 1D coefficient vector for the linear term.
        sigma (Scalar, optional): Amplitude scale.
        use_block (bool, optional): Whether to use block implementation where supported.
        const (None, optional): Optional constant offset for the tinygp linear kernel.

    Returns:
        JAXArray: Covariance object for the linear term.
    """
    assert alpha.ndim == 1

    if const is not None:
        tinygp_kf = luas.kernels.tinygp_ext.Amplitude(alpha = sigma * alpha, const = const)
        return covtype.GeneralQuasisepPlusNoise(tinygp_kf, use_block = use_block)
    else:
        return covtype.Outer(alpha * sigma)


def Banded(diag: JAXArray, off_diags: JAXArray, use_block: bool = True) -> covtype.CovType:
    
    symm_qsm = tinygp.noise.Banded(diag, off_diags).to_qsm()

    # Don't use HandleIdx here as need the location in matrix for non-stationary kernels
    return covtype.GeneralQuasisepPlusNoise(None, noise_model = symm_qsm, use_block = use_block)


def ConstantBlocks(endpoints: JAXArray, sigma: Scalar = 1., use_block: bool = True) -> JAXArray:
    r"""Piecewise-constant block kernel.

    Args:
        endpoints (JAXArray): Block boundary locations.
        sigma (Scalar, optional): Kernel amplitude.
        use_block (bool, optional): Whether to use block implementation where supported.

    Returns:
        JAXArray: Quasisep covariance object with constant blocks.
    """

    tinygp_kf = luas.kernels.tinygp_ext.ConstantBlocks(endpoints = endpoints, sigma = sigma)
    return covtype.GeneralQuasisep(HandleIdx(tinygp_kf), use_block = use_block)


def Exp(scale: Scalar, sigma: Scalar = 1., fast_eigen = False) -> JAXArray:
    r"""Exponential (Matérn-1/2) quasisep kernel.

    .. math::

        k(x,y)=\sigma^2\exp\left(-\frac{\|x-y\|}{\mathrm{scale}}\right)

    Args:
        scale (Scalar): Length scale :math:`L`.
        sigma (Scalar, optional): Kernel amplitude.
        fast_eigen (bool, optional): Use optimized eigendecomposition path when available.

    Returns:
        JAXArray: Exponential quasisep covariance object.
    """
    exp_kernel = tinygp.kernels.quasisep.Exp(scale = scale, sigma = sigma)
    return covtype.Exp(HandleIdx(exp_kernel), scale, sigma = sigma, fast_eigen = fast_eigen)


def Matern32(scale: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Matérn-3/2 quasisep kernel factory.

    .. math::

        k(x,y)=\sigma^2\left(1+\sqrt{3}r\right)e^{-\sqrt{3}r},\quad r=\frac{\|x-y\|}{\mathrm{scale}}

    Args:
        scale (Scalar): Length scale.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: Quasisep covariance object.
    """
    
    return covtype.GeneralQuasisep(HandleIdx(tinygp.kernels.quasisep.Matern32(scale = scale, sigma = sigma)))
    

def Matern52(scale: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Matérn-5/2 quasisep kernel factory.

    .. math::

        k(x,y)=\sigma^2\left(1+\sqrt{5}r+\frac{5}{3}r^2\right)e^{-\sqrt{5}r},\quad r=\frac{\|x-y\|}{\mathrm{scale}}

    Args:
        scale (Scalar): Length scale.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: Quasisep covariance object.
    """
    
    return covtype.GeneralQuasisep(HandleIdx(tinygp.kernels.quasisep.Matern52(scale = scale, sigma = sigma)))


def MaternHalfInt(scale: Scalar, double_nu: int, sigma = 1.) -> JAXArray:
    r"""Half-integer Matérn quasisep kernel factory.

    .. math::

        k(x,y)=\sigma^2\,\mathrm{Mat\'ern}_{\nu}(r),\quad \nu=\tfrac{1}{2}\,\mathrm{double\_nu},\; r=\frac{\|x-y\|}{\mathrm{scale}}

    Args:
        scale (Scalar): Length scale.
        double_nu (int): Twice the Matérn smoothness parameter :math:`\nu`.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: Quasisep covariance object.
    """
    tinygp_kf = luas.kernels.tinygp_ext.MaternHalfInt(scale = scale, double_nu = double_nu, sigma = sigma)
    return covtype.GeneralQuasisep(HandleIdx(tinygp_kf))


def SHO(omega: Scalar, quality: Scalar, sigma = 1.) -> JAXArray:
    r"""Simple harmonic oscillator (SHO) quasisep kernel.

    Args:
        omega (Scalar): Angular frequency parameter.
        quality (Scalar): Quality factor.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: SHO quasisep covariance object.
    """
    
    return covtype.GeneralQuasisep(HandleIdx(tinygp.kernels.quasisep.SHO(omega = omega, quality = quality, sigma = sigma)))


def Cosine(scale: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Cosine periodic quasisep kernel factory.

    .. math::

        k(x,y)=\sigma^2\cos\left(2\pi\frac{\|x-y\|}{\mathrm{scale}}\right)

    Args:
        scale (Scalar): Period/scale parameter used by tinygp cosine kernel.
        sigma (Scalar, optional): Kernel amplitude.

    Returns:
        JAXArray: Periodic quasisep covariance object.
    """

    return covtype.PeriodicQuasisep(HandleIdx(tinygp.kernels.quasisep.Cosine(scale = scale, sigma = sigma)))


def ExpSineSquaredApprox(
    period: JAXArray | float,
    gamma: JAXArray | float,
    sigma: Scalar = 1.,
    order: int = 5,
    use_block: bool = True,
):
    # Note defaults to an order of 5 which can be inaccurate
    kf_tinygp = luas.kernels.tinygp_ext.ExpSineSquaredApprox(period=period, gamma=gamma,
                                                             sigma=sigma, order=order,
                                                             use_block = use_block)
    return covtype.PeriodicQuasisep(HandleIdx(kf_tinygp))


def QuasiperiodicApprox(
    period: JAXArray | float,
    gamma: JAXArray | float,
    decay_kernel: covtype.CovType,
    sigma: Scalar = 1.,
    order: int = 5,
    use_block: bool = True,
):
    kf_tinygp = luas.kernels.tinygp_ext.QuasiperiodicApprox(period=period, gamma=gamma, decay_kernel = decay_kernel.tinygp_kf,
                                                            sigma=sigma, order=order)
    return covtype.GeneralQuasisep(HandleIdx(kf_tinygp), use_block = use_block)


def SquaredExpApprox(scale: Scalar, sigma: Scalar = 1., order: int = 6, use_block = True) -> JAXArray:
    """Taylor-spectrum squared-exponential approximation as a quasisep kernel.

    This returns a ``covtype.GeneralQuasisep`` wrapper so it plugs straight into
    existing ``luas`` covariance composition patterns.
    """
    kf_tinygp = luas.kernels.tinygp_ext.SquaredExpApprox(scale=scale, sigma=sigma, order=order,
                                                         use_block = use_block)
    return covtype.GeneralQuasisep(HandleIdx(kf_tinygp))
