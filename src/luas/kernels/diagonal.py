import jax.numpy as jnp
import numpy as np
from typing import Callable
from luas.luas_types import JAXArray, Scalar, PyTree, is_scalar
import luas.kernels.covtype as covtype

__all__ = [
    "Noise",
    "KroneckerDelta"
]

def Noise(sigma: JAXArray | Scalar) -> JAXArray:
    r"""Diagonal white-noise covariance term.

    Returns an identity-like covariance object with diagonal entries
    :math:`\sigma^2` (scalar or element-wise).

    Args:
        sigma (JAXArray | Scalar): Noise standard deviation(s).

    Returns:
        JAXArray: ``covtype.ScaledIdentity`` for scalar ``sigma`` or
        ``covtype.Diagonal`` for array-valued ``sigma``.
    """
    
    if is_scalar(sigma):
        return covtype.ScaledIdentity(wn_diag = sigma**2)
    else:
        return covtype.Diagonal(wn_diag = sigma**2)
 
  
def KroneckerDelta(sigma: JAXArray | Scalar = None) -> JAXArray:
    r"""Diagonal covariance term.

    Returns an identity-like covariance object with diagonal entries
    :math:`\sigma^2` (scalar or element-wise).
    
    Unlike ``Noise``, this term is for terms which are modelled as correlated, but are independent in this particular dimension.
    Swapping out the ``Noise`` term for a ``KroneckerDelta`` term will give the same log-likelihood values but the GP predictive mean
    will be different, as the GP predictive mean will try to fit this term rather than treating it as uncorrelated noise.

    Args:
        diag (JAXArray | Scalar): Noise standard deviation(s).

    Returns:
        JAXArray: ``covtype.ScaledIdentity`` for scalar ``sigma`` or
        ``covtype.Diagonal`` for array-valued ``sigma``.
    """
    if sigma is not None:
        if is_scalar(sigma):
            return covtype.ScaledIdentity(diag = sigma**2)
        else:
            return covtype.Diagonal(diag = sigma**2)
    else:
        return covtype.Identity()
