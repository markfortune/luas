import numpy as np
import jax.numpy as jnp
from typing import Callable, Tuple, Union, Any, Optional

from luas.kernels.covtype import CovType
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import calc_total_size, calc_data_shape, cyclic_transpose, tensor_mult, vmap_for_tensors

__all__ = [
    "SeparableKernel",
]

class SeparableKernel(CovType):
    r"""GP optimisation for a covariance matrix that is a single Kronecker product,
    i.e. a kernel which is fully separable in every dimension.

    For 2D data this handles kernel functions of the form

    .. math::

        k(\lambda, \lambda', t, t') = s_\lambda(\lambda, \lambda')\, s_t(t, t'),

    which on a rectangular grid gives the covariance matrix

    .. math::

        \mathbf{K} = \mathbf{\Sigma}_\lambda \otimes \mathbf{\Sigma}_t.

    Here and throughout the ``luas`` Kronecker kernels, dimension 0 is written as
    wavelength :math:`\lambda` and dimension 1 as time :math:`t` to match the
    exoplanet use case, but any rectangular grid can be used. Unlike most of the
    other Kronecker kernels, this optimisation works for any number of dimensions,
    i.e. :math:`\mathbf{K} = \mathbf{\Sigma}_1 \otimes \mathbf{\Sigma}_2 \otimes
    \dots \otimes \mathbf{\Sigma}_d`.

    A typical use is modelling white noise plus measurement covariance, e.g.
    wavelength-correlated readout noise which is drawn independently for each
    integration.

    **Method.** Uses the approach of Saatchi (2011), generalised to any matrix
    square root (Fortune, Gibson & Pope, in prep.). Each
    :math:`\mathbf{\Sigma}_d` is decomposed separately using its own ``decompose``
    method (general Cholesky, quasiseparable/``tinygp`` Cholesky, or a trivial
    diagonal decomposition). The residuals are then whitened by applying each
    matrix inverse square root along its own axis, after which the log-likelihood
    reduces to a least-squares term on the whitened residuals plus

    .. math::

        \log|\mathbf{K}| = \sum_d \frac{N}{N_d} \log|\mathbf{\Sigma}_d|,

    where :math:`N = \prod_d N_d` is the total number of data points.

    **Runtime scaling (2D).**

    - Diagonal :math:`\mathbf{\Sigma}_\lambda` and :math:`\mathbf{\Sigma}_t`:
      :math:`\mathcal{O}(N_\lambda N_t)`.
    - Quasiseparable :math:`\mathbf{\Sigma}_\lambda`, :math:`\mathbf{\Sigma}_t`
      (from ``luas.kernels.quasisep``):
      :math:`\mathcal{O}(N_\lambda J_{\Sigma_\lambda}^2 + N_t J_{\Sigma_t}^2
      + N_\lambda N_t (J_{\Sigma_\lambda} + J_{\Sigma_t}))`, i.e. linear in both
      dimensions.
    - Arbitrary :math:`\mathbf{\Sigma}_\lambda` with diagonal
      :math:`\mathbf{\Sigma}_t`: :math:`\mathcal{O}(N_\lambda^3 + N_\lambda^2 N_t)`.

    **Selection.** :class:`luas.GP` selects this optimisation automatically when the
    kernel function ``kf`` returns a single tuple with one kernel per dimension
    (rather than a tuple of tuples).

    Warning:
        Every :math:`\mathbf{\Sigma}_d` must be positive definite and reasonably
        well-conditioned, since each is inverted. Including non-zero white noise
        (``Noise``) or a ``KroneckerDelta`` term in each dimension guarantees this.

    Example:
        White noise varying in wavelength plus exponential readout noise in
        wavelength, independent in time (from the ``luas`` tutorial):

        .. code-block:: python

            import jax.numpy as jnp
            import luas

            def kf(p, X):
                s_l = luas.kernels.quasisep.Noise(jnp.exp(p["log_sigma"]))
                s_l += jnp.exp(2*p["log_h_readout"][0]) * luas.kernels.quasisep.Exp(jnp.exp(p["log_l_readout"][0]))
                s_t = luas.kernels.quasisep.KroneckerDelta()

                # A single tuple, one kernel per dimension, in the same order as X
                return (s_l, s_t)

            gp = luas.GP(p, (x_l, x_t), kf = kf)  # Uses SeparableKernel

    Args:
        *Sigma (CovType): One covariance object per dimension, in the same order as
            the regression variables ``X`` and the axes of the data array. Normally
            built from ``luas.kernels``, ``luas.kernels.quasisep`` or
            ``luas.kernels.diagonal`` kernels. Each must be positive definite.
        use_stored_values (bool, optional): Reusing stored decompositions is not
            currently implemented for this kernel, leave as ``False``.
        data_shape (tuple, optional): Currently unused.
        inv_dims (bool, optional): If ``True``, computes :math:`\vec{r}^T
            \mathbf{K}^{-1} \vec{r}` using a fallback which applies the full inverse
            :math:`\mathbf{\Sigma}_d^{-1}` along any dimension whose covariance
            object does not provide a matrix inverse square root. Defaults to
            ``False``.
        **kwargs: Ignored, accepted so that all Kronecker kernels share a common
            constructor interface.

    See Also:
        :class:`luas.LuasKernel`, :class:`luas.LuasLasrachKernel` for adding a
        second, full-rank Kronecker product term.
        :class:`luas.MultiLowRankKernel` for adding low-rank terms.
        :class:`luas.GeneralKernel` for an unoptimised reference implementation.
    """

    def __init__(
        self,
        *Sigma,
        use_stored_values: Optional[bool] = False,
        data_shape: Optional[Tuple] = None,
        inv_dims: bool = False,
        **kwargs,
    ):
        r"""Creates the kernel object, see the class docstring for the arguments."""

        self.Sigma = Sigma
        self.dim = len(Sigma)
        self.inv_dims = inv_dims
        self.opt_name = "SeparableKernel"

        if inv_dims:
            self.dot_solve = self.dot_solve_w_inv_dims
        
        # Define for consistency with other kernel objects
        self.K_list = []

        self.data_shape = data_shape
           
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            raise NotImplementedError("Use of pre-calculated stored values not yet implemented.")
        else:
            self.decompose = self.decompose_no_stored_values
    
    def evaluate(self, X, **kwargs):
        r"""Builds the full dense covariance matrix by evaluating each Kronecker term and summing them. Only
        practical for small data sets, as the matrix has :math:`N^2` entries for :math:`N` data points.

        Args:
            X (tuple): Input locations along each dimension.
            **kwargs: Passed to each covariance object's ``evaluate`` method, e.g. ``wn``.

        Returns:
            JAXArray: The covariance matrix, of shape ``(N, N)``, ordered as the row-major (``"C"``) flattening
            of the data array.

        """

        dim = len(X)
        Sigma = self.Sigma[0].evaluate(X[0], X[0], **kwargs)
        
        for d in range(1, dim):
            Sigma = jnp.kron(Sigma, self.Sigma[d].evaluate(X[d], X[d], **kwargs))
        
        return Sigma
    
    def decompose_no_stored_values(
        self,
        X: JAXArray,
        stored_values: Optional[PyTree] = {},
        full = True,
        **kwargs,
    ) -> PyTree:
        r"""Decomposes each :math:`\mathbf{\Sigma}_d` separately using its own ``decompose`` method. The log
        determinant of the Kronecker product is :math:`\sum_d (N / N_d) \log|\mathbf{\Sigma}_d|`.

        Args:
            X (tuple): Input locations along each dimension.
            stored_values (PyTree, optional): Dictionary the stored values are added to.
            full (bool, optional): Not used.

        Returns:
            (SeparableKernel, PyTree): Itself and the stored values, including ``"logdet"``.

        """

        total_size = calc_total_size(X)
        gp_dim = len(X)
        
        stored_values["logdet"] =  0.
        sigma_decomp = ()
        for d in range(gp_dim):
            Sigma_d_new, stored_values_d = self.Sigma[d].decompose(X[d])

            if gp_dim > 2:
                Sigma_d_new.matrix_sqrt = vmap_for_tensors(Sigma_d_new.matrix_sqrt)
                Sigma_d_new.matrix_inv_sqrt = vmap_for_tensors(Sigma_d_new.matrix_inv_sqrt)

            sigma_decomp += (Sigma_d_new,)
            stored_values["logdet"] += (total_size/X[d].shape[-1])*stored_values_d["logdet"]

        self.Sigma = sigma_decomp
        self.logdet = stored_values["logdet"]

        return self, stored_values
    
    def dot_solve_w_inv_dims(self, R):
        r"""Calculates :math:`\vec{r}^T \mathbf{K}^{-1} \vec{r}`, applying the full inverse
        :math:`\mathbf{\Sigma}_d^{-1}` along any dimension whose covariance object has no matrix inverse square
        root. Used instead of ``dot_solve`` when ``inv_dims = True``.

        Args:
            R (JAXArray): The data-shaped residuals.

        Returns:
            Scalar: :math:`\vec{r}^T \mathbf{K}^{-1} \vec{r}`.

        """
        R_prime = cyclic_transpose(R, 2)
        R_T = R_prime.copy()

        for d in range(self.dim):
            if hasattr(self.Sigma[d], "matrix_inv_sqrt"):
                R_prime = self.Sigma[d].matrix_inv_sqrt(R_prime, transpose = 0)
                R_T = self.Sigma[d].matrix_inv_sqrt(R_T, transpose = 0)
            else:
                R_prime = self.Sigma[d].inverse(R_prime)
            
            R_prime = cyclic_transpose(R_prime, 1)
            R_T = cyclic_transpose(R_T, 1)

        return (R_T * R_prime).sum()
    
    def matrix_sqrt(self, R, transpose = 0):
        r"""Multiplies ``R`` by the matrix square root :math:`\bigotimes_d \mathbf{L}_d` (or its transpose if
        ``transpose = 1``), by applying each :math:`\mathbf{L}_d` along its own dimension. Requires
        :meth:`decompose` to have been called.

        """

        R_prime = cyclic_transpose(R, 2)

        for d in range(self.dim):
            R_prime = self.Sigma[d].matrix_sqrt(R_prime, transpose = transpose)
            R_prime = cyclic_transpose(R_prime, 1)
            
        R_prime = cyclic_transpose(R_prime, -2)
        
        return R_prime
    
    def matrix_inv_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
    ) -> JAXArray:
        r"""Multiplies ``R`` by :math:`\bigotimes_d \mathbf{L}_d^{-1}` (or its transpose if ``transpose = 1``), by applying
        each :math:`\mathbf{L}_d^{-1}` along its own dimension. Requires :meth:`decompose` to have been called.

        """

        R_prime = cyclic_transpose(R, 2)

        for d in range(self.dim):
            R_prime = self.Sigma[d].matrix_inv_sqrt(R_prime, transpose = transpose)
            R_prime = cyclic_transpose(R_prime, 1)
            
        R_prime = cyclic_transpose(R_prime, -2)
        
        return R_prime

    def eigendecomp(
        self,
        X,
        **kwargs,
    ):
        r"""Calculates the eigendecomposition of the Kronecker product from the eigendecomposition along each
        dimension: the eigenvalues are the Kronecker product of the eigenvalues along each dimension.

        Args:
            X (tuple): Input locations along each dimension.

        Returns:
            (JAXArray, tuple): The eigenvalues, with one axis per dimension, and a tuple of the eigenvectors along
            each dimension.

        """

        dim = len(X)

        all_lam = jnp.ones(1)
        all_lam_shape = ()
        eigen_decomp_mats = ()
        
        for d in jnp.arange(dim):
            lam_d, Q_d = self.Sigma[d].eigendecomp(X[d])

            all_lam = jnp.kron(all_lam.reshape(all_lam_shape + (1,)), lam_d)
            all_lam_shape = all_lam.shape
            eigen_decomp_mats += (Q_d,)

        return all_lam, eigen_decomp_mats
    
    def matmul(self, X1, X2, R, **kwargs):
        r"""Multiplies the data-shaped array ``R`` by the covariance matrix without building it, by applying each
        Kronecker term along each dimension (see :func:`luas.kronecker_fns.tensor_mult`).

        Args:
            X1 (tuple): Input locations along each dimension for the rows.
            X2 (tuple): Input locations along each dimension for the columns, matching the shape of ``R``.
            R (JAXArray): The array to multiply, with one axis per dimension.
            **kwargs: Passed to each covariance object's ``matmul`` method, e.g. ``wn = False``.

        Returns:
            JAXArray: The product, with one axis per dimension.

        """
        
        R_prime = tensor_mult(self.Sigma, X1, X2, R, **kwargs)
        
        return R_prime
