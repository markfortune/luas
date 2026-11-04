import numpy as np
from typing import Optional, Callable, Tuple, Any, Union

import jax.numpy as jnp

from luas.kernels.covtype import CovType
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
from luas.kronecker_fns import make_vec, make_mat, calc_data_shape

__all__ = ["WhiteNoiseKernel"]


class WhiteNoiseKernel(CovType):
    r"""Diagonal covariance matrix with an independent variance for every data point.

    Models uncorrelated noise whose amplitude does not need to be separable in each
    dimension, i.e.

    .. math::

        k(\lambda, \lambda', t, t') = \sigma^2(\lambda, t)\,\delta_{\lambda\lambda'}\delta_{tt'},
        \qquad \mathbf{K} = \mathrm{diag}\left(\mathrm{vec}(\mathbf{D})\right),

    where :math:`\mathbf{D}` has the same shape as the data. If the variance can be
    written as :math:`\sigma^2(\lambda)\sigma^2(t)`, the same model can instead be
    built with :class:`luas.SeparableKernel` using ``Noise``/``KroneckerDelta``
    kernels, which can then be combined with correlated kernel terms.

    The variance is split into two parts, :math:`\mathbf{D} = \texttt{diag} +
    \texttt{wn\_diag}`. Both contribute to the log-likelihood, but ``wn_diag`` is
    treated as white noise and is dropped when ``wn = False`` (e.g. when
    :class:`luas.GP` computes a GP predictive mean), while ``diag`` is treated as
    part of the signal being fit.

    **Runtime scaling.** :math:`\mathcal{O}(N)` for :math:`N` total data points.

    **Selection.** :class:`luas.GP` uses ``WhiteNoiseKernel(wn_diag = 1.)`` by
    default when no kernel function is given, which gives an ordinary
    (unit-variance) least-squares log-likelihood. For parameter-dependent
    variances, have the kernel function return a ``WhiteNoiseKernel`` directly.

    Warning:
        ``diag`` and ``wn_diag`` should have the same shape as the data. Scalars or
        arrays which only broadcast against the data (e.g. shape ``(N_l, 1)``) are
        accepted by some methods but give an incorrect log determinant.

    Example:
        .. code-block:: python

            def kf(p, X):
                # Separate white noise amplitude for every data point
                return luas.WhiteNoiseKernel(wn_diag = jnp.exp(2*p["log_sigma_2D"]))

            gp = luas.GP(p, (x_l, x_t), kf = kf)

    Args:
        diag (JAXArray, optional): Variance at each data point which is retained
            when ``wn = False``. Defaults to ``0.``.
        wn_diag (JAXArray, optional): White noise variance at each data point,
            excluded when ``wn = False``. Defaults to ``0.``.
        **kwargs: Ignored.

    See Also:
        :class:`luas.SeparableKernel` for separable white noise and measurement
        covariance.
    """

    def __init__(
        self,
        diag: Optional[JAXArray] = 0.,
        wn_diag: Optional[JAXArray] = 0.,
        **kwargs,
    ):
        r"""Creates the kernel object, see the class docstring for the arguments."""
        # Function used to build the covariance matrix K
        self.diag = diag
        self.wn_diag = wn_diag
        self.opt_name = "WhiteNoiseKernel"
        

    def evaluate(
        self,
        X: JAXArray,
        wn = True,
        full = True,
        row_idx = None,
        col_idx = None,
        **kwargs,
    ) -> JAXArray:
        r"""Builds the full diagonal covariance matrix, with diagonal ``diag + wn * wn_diag`` flattened in row-major
        (``"C"``) order.

        Args:
            X (tuple): Input locations along each dimension, used for the shape of the data.
            wn (bool, optional): Whether to include ``wn_diag``. Defaults to ``True``.
            full (bool, optional): Must be ``True``, sub-matrices are not implemented.
            row_idx, col_idx: Not used.

        Returns:
            JAXArray: The diagonal covariance matrix, of shape ``(N, N)``.

        """

        diag = (self.diag + wn * self.wn_diag) * jnp.ones(calc_data_shape(X))

        if not full or is_scalar(diag):
            raise Exception("Not implemented")
        
        diag_mat = jnp.diag((self.diag + wn * self.wn_diag).ravel())

        return diag_mat

    def decompose(
        self,
        X: JAXArray,
        wn = True,
        **kwargs,
    ) -> JAXArray:
        r"""Stores the variance at every data point, ``diag + wn * wn_diag``, and the log determinant (the sum of
        their logs).

        Args:
            X (tuple): Input locations along each dimension, used for the shape of the data.
            wn (bool, optional): Whether to include ``wn_diag``. Defaults to ``True``.

        Returns:
            (WhiteNoiseKernel, dict): Itself and the stored value ``"logdet"``.

        """

        self.D = (self.diag + wn * self.wn_diag) * jnp.ones(calc_data_shape(X))
        self.logdet = jnp.log(self.D).sum()
        
        return self, {"logdet":self.logdet}

    def matrix_sqrt(
        self,
        R: JAXArray,
        **kwargs,
    ) -> JAXArray:
        r"""Multiplies ``R`` element-wise by the standard deviation at each data point. Requires :meth:`decompose` to have been called."""

        return R * jnp.sqrt(self.D)

    def matrix_inv_sqrt(
        self,
        R: JAXArray,
        **kwargs,
    ) -> JAXArray:
        r"""Divides ``R`` element-wise by the standard deviation at each data point. Requires :meth:`decompose` to have been called."""

        return R * jnp.sqrt(1/self.D)

    def inverse(
        self,
        R: JAXArray,
    ) -> JAXArray:
        r"""Divides ``R`` element-wise by the variance at each data point. Requires :meth:`decompose` to have been called."""

        return R * (1/self.D)

    def dot_solve(
        self,
        R: JAXArray,
    ) -> JAXArray:
        r"""Calculates :math:`\vec{r}^T \mathbf{K}^{-1} \vec{r}`, the sum of the squared residuals divided by the
        variance at each data point. Requires :meth:`decompose` to have been called.

        """

        return jnp.sum(jnp.square(R)/self.D)

    def block_dot_solve(
        self,
        v1: JAXArray,
        v2: JAXArray,
        vec_dim: int,
    ) -> JAXArray:
        r"""Calculates the products :math:`\vec{v}_1^T \mathbf{K}^{-1} \vec{v}_2` for blocks of vectors, by contracting
        ``v1 * v2`` along its first axis with the inverse variances along axis ``vec_dim``. Used for the low-rank
        corrections of the Kronecker kernels with low-rank terms.

        Args:
            v1 (JAXArray): The first set of vectors.
            v2 (JAXArray): The second set of vectors, with the same shape as ``v1``.
            vec_dim (int): The axis of the variances to contract over.

        Returns:
            JAXArray: The products.

        """

        vec_prod = v1 * v2

        diag_vals = jnp.tensordot(1/self.D, vec_prod, axes=([vec_dim], [0]))

        return diag_vals

    def matmul(
        self,
        X1: Tuple,
        X2: Tuple,
        R: JAXArray,
        wn = True,
        full = True,
        row_idx = None,
        col_idx = None,
        **kwargs,
    ) -> JAXArray:
        r"""Multiplies ``R`` element-wise by ``diag + wn * wn_diag``.

        Args:
            X1, X2 (tuple): Not used, accepted for the common Kronecker kernel interface.
            R (JAXArray): The data-shaped array to multiply.
            wn (bool, optional): Whether to include ``wn_diag``. Defaults to ``True``.
            full (bool, optional): Must be ``True`` unless the variance is a scalar.

        Returns:
            JAXArray: The product.

        """
        
        diag = self.diag + wn * self.wn_diag

        if not full and not is_scalar(diag):
            raise Exception("Not implemented")

        return R * diag
        
   