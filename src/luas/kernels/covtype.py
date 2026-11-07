r"""Covariance objects ("CovTypes") which store the covariance matrix along one dimension of the data.

Each class stores a covariance matrix in a particular structure (dense, diagonal, quasiseparable,
rank one, ...) and implements the matrix operations the Kronecker kernel classes need, such as
decompositions, matrix square roots and their inverses, matrix-vector products and
eigendecompositions, in a way that exploits that structure. The kernel functions in
:doc:`luas.kernels <kernels>` return these objects, and adding or multiplying them together picks the
structure of the result, e.g. a quasiseparable kernel plus white noise stays quasiseparable.

Warning:
    This module is internal to ``luas`` and is documented for reference only. These objects are
    created by the kernel functions in ``luas.kernels`` and used by :class:`luas.GP`, and should not be
    created or used directly. Their interface is not stable and may change between versions without
    notice.

Conventions used by the methods in this module:

- ``diag`` and ``wn_diag`` are added to the diagonal of the matrix. ``wn_diag`` is the white noise part,
  which is only included when ``wn = True`` (it is left out when making GP predictions).
- ``full = True`` evaluates the full matrix. Otherwise ``row_idx`` and ``col_idx`` select the rows and
  columns of the sub-matrix to evaluate.
- ``decompose`` must be called before the matrix square root, inverse and log likelihood methods, and
  returns the decomposed object together with a dictionary of stored values including ``"logdet"``.
- The matrix square root :math:`\mathbf{L}` satisfies :math:`\mathbf{K} = \mathbf{L}\mathbf{L}^T`, and
  ``transpose = 1`` applies :math:`\mathbf{L}^T` (or :math:`\mathbf{L}^{-T}`) instead.

"""
import warnings

import numpy as np
from typing import Optional, Callable, Tuple, Any, Union
import scipy

import jax
import jax.numpy as jnp
import jax.scipy.linalg as JLA

import tinygp
from tinygp.solvers.quasisep.core import SymmQSM, StrictLowerTriQSM, DiagQSM
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
import luas.kernels.tinygp_ext
from luas.kernels.tinygp_ext import ScaledKernel, HandleIdx
from luas.kernels.householder import HouseholderTransform, HouseholderandDiagonalTransform
from luas.stable_gradients import stable_eigh


class CovType():
    r"""Base class of the covariance objects, representing the covariance matrix along one dimension of the data.

    Subclasses store the matrix in a particular structure and override these methods to exploit it. The
    methods here either raise an Exception, for operations each subclass must implement, or fall back to
    dense matrix calculations.

    Warning:
        Internal to ``luas`` and not intended to be used directly. The interface is not stable and may change
        without notice.

    The attributes ``diag`` and ``wn_diag`` are the values added to the diagonal of the matrix (``wn_diag`` only
    when ``wn = True``), ``params`` is optional metadata about the parameters the object depends on, and if
    ``use_inv`` is ``True``, :meth:`dot_solve` uses :meth:`inverse` rather than :meth:`matrix_inv_sqrt`.

    """
    K_list = []
    params = None
    use_inv = False
    diag = 0.
    wn_diag = 0.

    def evaluate(self, x1, x2, **kwargs):
        r"""Evaluates the covariance matrix between two sets of input locations.

        Args:
            x1 (JAXArray): Input locations for the rows, of shape ``(N_1,)`` or ``(d, N_1)``.
            x2 (JAXArray): Input locations for the columns, of shape ``(N_2,)`` or ``(d, N_2)``.
            **kwargs: Options such as ``wn``, ``full``, ``row_idx`` and ``col_idx`` (see the module docstring).

        Returns:
            JAXArray: The covariance matrix, of shape ``(N_1, N_2)``.

        Raises:
            Exception: If the subclass doesn't implement it.

        """
        raise Exception("Evaluating covariance matrix not implemented for this CovType/Kernel")
    
    def decompose(self, x, **kwargs):
        r"""Decomposes the covariance matrix at the input locations ``x``, so that its matrix square root, inverse
        and log determinant can be calculated.

        Args:
            x (JAXArray): Input locations, of shape ``(N,)`` or ``(d, N)``.
            **kwargs: Options such as ``wn``, ``full`` and ``idx``.

        Returns:
            (CovType, dict): The decomposed covariance object and a dictionary of stored values, including
            the log determinant ``"logdet"``.

        Raises:
            Exception: If the subclass doesn't implement it.

        """
        raise Exception("Decomposition not implemented for this CovType/Kernel")

    def matrix_inv_sqrt(self, R, transpose=0):
        r"""Multiplies ``R`` by :math:`\mathbf{L}^{-1}` (or :math:`\mathbf{L}^{-T}` if ``transpose = 1``), where
        :math:`\mathbf{K} = \mathbf{L}\mathbf{L}^T`. Requires :meth:`decompose` to have been called.

        Args:
            R (JAXArray): Array to multiply, whose first axis has the size of the matrix.
            transpose (int, optional): Whether to apply the transpose. Defaults to 0.

        Returns:
            JAXArray: The result, with the same shape as ``R``.

        Raises:
            Exception: If the subclass doesn't implement it.

        """
        raise Exception("Matrix inverse square root not implemented for this CovType/Kernel")

    def matrix_sqrt(self, R, transpose=0):
        r"""Multiplies ``R`` by :math:`\mathbf{L}` (or :math:`\mathbf{L}^T` if ``transpose = 1``), where
        :math:`\mathbf{K} = \mathbf{L}\mathbf{L}^T`. Requires :meth:`decompose` to have been called.

        Args:
            R (JAXArray): Array to multiply, whose first axis has the size of the matrix.
            transpose (int, optional): Whether to apply the transpose. Defaults to 0.

        Returns:
            JAXArray: The result, with the same shape as ``R``.

        Raises:
            Exception: If the subclass doesn't implement it.

        """
        raise Exception("Matrix square root not implemented for this CovType/Kernel")

    def scale(self, c):
        r"""Returns a new covariance object for the matrix multiplied by a scalar, :math:`c\mathbf{K}`.

        Args:
            c (Scalar): The scalar to multiply by.

        Returns:
            CovType: The scaled covariance object.

        Raises:
            Exception: If the subclass doesn't implement it.

        """
        raise Exception("Multiplication by a scalar not implemented for this CovType/Kernel")

    def eigendecomp(self, x, wn = True, full = True, idx = None, **kwargs):
        r"""Calculates the eigendecomposition :math:`\mathbf{K} = \mathbf{Q}\,\mathrm{diag}(\vec{\lambda})\,\mathbf{Q}^T`
        of the matrix at the input locations ``x``.

        The default builds the dense matrix and uses ``jnp.linalg.eigh`` with numerically stable gradients
        (see ``luas.stable_gradients.stable_eigh``).

        Args:
            x (JAXArray): Input locations, of shape ``(N,)`` or ``(d, N)``.
            wn (bool, optional): Whether to include white noise. Defaults to ``True``.
            full (bool, optional): Whether to use the full matrix, otherwise the sub-matrix given by ``idx``.
                Defaults to ``True``.
            idx (JAXArray, optional): Row and column indices of the sub-matrix if ``full = False``.

        Returns:
            (JAXArray, JAXArray): The eigenvalues and a matrix whose columns are the eigenvectors.

        """

        if not full:
            K = self.evaluate(x, x, wn = wn, full = False,
                              row_idx = idx, col_idx = idx, **kwargs)
        else:
            K = self.evaluate(x, x, wn = wn, full = True, **kwargs)
        return stable_eigh(K)
    
    def inverse(self, R):
        r"""Multiplies ``R`` by the inverse of the matrix, :math:`\mathbf{K}^{-1}\mathbf{R} = \mathbf{L}^{-T}\mathbf{L}^{-1}\mathbf{R}`.
        Requires :meth:`decompose` to have been called.

        """
        R_prime = self.matrix_inv_sqrt(R, transpose=0)
        K_inv_R = self.matrix_inv_sqrt(R_prime, transpose=1)

        return K_inv_R
    
    def dot_solve(self, R):
        r"""Calculates :math:`\vec{r}^T \mathbf{K}^{-1} \vec{r}`, using :meth:`inverse` if ``use_inv`` is ``True``
        and otherwise the sum of squares of :math:`\mathbf{L}^{-1}\vec{r}`. Requires :meth:`decompose` to have
        been called.

        Args:
            R (JAXArray): The residuals.

        Returns:
            Scalar: :math:`\vec{r}^T \mathbf{K}^{-1} \vec{r}`.

        """
        if self.use_inv:
            # For some optimisations its easier to get the inverse then the matrix inverse sqrt
            K_inv_R = self.inverse(R)
            return (R * K_inv_R).sum()
        else:
            L_inv_R = self.matrix_inv_sqrt(R, transpose = 0)
            return jnp.square(L_inv_R).sum()

    def logL(self, R, **kwargs):
        r"""Calculates the Gaussian log likelihood of the residuals ``R``,

        .. math::

            \log \mathcal{L} = -\frac{1}{2}\vec{r}^T \mathbf{K}^{-1} \vec{r} - \frac{1}{2}\log|\mathbf{K}| - \frac{N}{2}\log 2\pi.

        Requires :meth:`decompose` to have been called, which sets the log determinant.

        Args:
            R (JAXArray): The residuals of the data after subtracting the mean function.
            **kwargs: Not used, accepted so all covariance objects can be called the same way.

        Returns:
            Scalar: The log likelihood.

        """
        return - 0.5 * self.dot_solve(R) - 0.5 * self.logdet - 0.5 * R.size * jnp.log(2*jnp.pi)
    
    def logL_hessianable(self, R, **kwargs):
        r"""Calculates the log likelihood like :meth:`logL`. Kernels whose :meth:`logL` doesn't give accurate
        second derivatives override this with a version that does.

        """
        return - 0.5 * self.dot_solve(R) - 0.5 * self.logdet - 0.5 * R.size * jnp.log(2*jnp.pi)

    def inv_sqrt_transform(self, K, x, **K_kwargs):
        r"""Transforms another covariance object by the inverse matrix square root of this one, returning
        :math:`\tilde{\mathbf{K}} = \mathbf{L}^{-1} \mathbf{K} \mathbf{L}^{-T}` as a new covariance object.

        Used by the Kronecker kernel classes to whiten one Kronecker term by another. The structure of
        ``K`` is kept where possible: an ``Outer`` matrix stays rank one, and dense, diagonal and
        quasiseparable matrices become a ``General`` matrix. Any other object must provide a
        ``transform_with_inv_sqrt`` method. Requires :meth:`decompose` to have been called.

        Args:
            K (CovType): The covariance object to transform.
            x (JAXArray): Input locations to evaluate ``K`` at.
            **K_kwargs: Passed on when decomposing or evaluating ``K``, e.g. ``full`` and ``idx``.

        Returns:
            CovType: The transformed covariance object.

        Raises:
            NotImplementedError: If ``K`` can't be transformed.

        """
        
        if isinstance(K, Outer):
            K, _ = K.decompose(x, **K_kwargs)
            alpha_tilde = self.matrix_inv_sqrt(K.alpha, transpose=0)
            K_tilde = Outer(alpha = alpha_tilde)

        elif isinstance(K, (General, GeneralQuasisep, GeneralQuasisepPlusNoise, Identity,
                            ScaledIdentity, Diagonal, OuterPlusScaledIdentity)):
            # Define a new General covtype for K where it's kernel function transforms it
            K_eval = K.evaluate(x, x, row_idx = K_kwargs.get("idx"), col_idx = K_kwargs.get("idx"), full = K_kwargs.get("full", True))
            K_prime = self.matrix_inv_sqrt(K_eval, transpose=0)
            K_tilde_eval = self.matrix_inv_sqrt(K_prime.T, transpose=0)

            def kf_transf(hp, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs):
                if full:
                    return K_tilde_eval
                else:                   
                    return K_tilde_eval[jnp.ix_(row_idx, col_idx)]

            K_tilde = General(kf_transf)

        else:
            if hasattr(K, "transform_with_inv_sqrt"):
                K_tilde = K.transform_with_inv_sqrt(self.matrix_inv_sqrt, x, **K_kwargs)
            else:
                raise NotImplementedError(f"Don't know how to whiten this CovType/Kernel type {type(K).__name__} to K_tilde = L^-1 K L^-T")
        
        return K_tilde


    def general_transf(self, mat, x, **K_kwargs):
        r"""Returns a ``General`` covariance object for :math:`\mathbf{M}\mathbf{K}\mathbf{M}^T`, where
        :math:`\mathbf{K}` is this matrix evaluated densely at ``x``.

        Args:
            mat (JAXArray): The matrix :math:`\mathbf{M}`.
            x (JAXArray): Input locations to evaluate the matrix at.
            **K_kwargs: Passed to :meth:`evaluate`.

        Returns:
            General: The transformed covariance object.

        """
        K_eval = mat @ self.evaluate(x, x, **K_kwargs) @ mat.T
        def kf_transf(hp, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs):
            if full:
                return K_eval
            else:
                return K_eval[jnp.ix_(row_idx, col_idx)]
            
        return General(kf_transf)


    def matmul(self, x1, x2, R, **kwargs):
        r"""Multiplies ``R`` by the covariance matrix between ``x1`` and ``x2``, :math:`\mathbf{K}(\vec{x}_1, \vec{x}_2)\mathbf{R}`.
        The default builds the dense matrix with :meth:`evaluate`.

        Args:
            x1 (JAXArray): Input locations for the rows.
            x2 (JAXArray): Input locations for the columns, matching the first axis of ``R``.
            R (JAXArray): The array to multiply.
            **kwargs: Passed to :meth:`evaluate`, e.g. ``wn``.

        Returns:
            JAXArray: The product.

        """
        return self.evaluate(x1, x2, **kwargs) @ R
    
    def rank(self, x, **kwargs):
        r"""Returns the rank of the matrix at the input locations ``x``. Full rank (the number of locations) by default."""
        return x.shape[-1]
        
    def __call__(self, x1, x2, **kwargs):
        r"""Evaluates the covariance matrix, the same as :meth:`evaluate`."""
        return self.evaluate(x1, x2, **kwargs)


    def __add__(self, other):
        r"""Adds another covariance object to this one. Subclasses return a new covariance object whose structure
        depends on both terms. Returns ``NotImplemented`` here.

        """
        return NotImplemented

    def __mul__(self, other):
        r"""Multiplies this covariance object by another, or by a scalar. Products of two covariance objects are
        element-wise (the product of the two kernels). Subclasses return a new covariance object whose structure
        depends on both terms. Returns ``NotImplemented`` here.

        """
        return NotImplemented

    def __radd__(self, other):
        r"""Adds this object to ``other`` (``other + self``), the same as ``self + other``."""
        return self.__add__(other)
        
    def __rmul__(self, other):
        r"""Multiplies ``other * self``, the same as ``self * other``."""
        return self.__mul__(other)


class General(CovType):
    r"""Dense covariance matrix defined by a kernel function, decomposed with a Cholesky decomposition.

    The most general structure, costing :math:`\mathcal{O}(N^3)` to decompose. Other covariance objects
    fall back to this when added to or multiplied by a term whose structure they can't keep.

    Args:
        kf (Callable): Kernel function ``kf(hp, x1, x2, **kwargs)`` returning the covariance matrix. If
            sub-matrices are needed it should accept the ``full``, ``row_idx`` and ``col_idx`` keyword arguments.
        hp (PyTree, optional): Hyperparameters passed to ``kf``. Defaults to an empty dict.
        params (optional): Metadata about the parameters the object depends on.

    """
    def __init__(self, kf, hp = {}, params = None):
        r"""Creates the covariance object, see the class docstring for the arguments."""
        self.kf = kf
        self.hp = hp
        self.params = params
        self.opt_name = "General (1D)"
        
    def evaluate(self, x1, x2, **kwargs):
        r"""Evaluates the kernel function, ``kf(hp, x1, x2, **kwargs)``."""
        K = self.kf(self.hp, x1, x2, **kwargs)

        return K

    def decompose(self, x, full = True, idx = None, **kwargs):
        r"""Calculates the Cholesky decomposition :math:`\mathbf{K} = \mathbf{L}\mathbf{L}^T` with :math:`\mathbf{L}`
        lower triangular (matching ``tinygp`` rather than the ``scipy`` default) and the log determinant.

        Args:
            x (JAXArray): Input locations.
            full (bool, optional): Whether to decompose the full matrix, otherwise the sub-matrix given by
                ``idx``. Defaults to ``True``.
            idx (JAXArray, optional): Row and column indices of the sub-matrix if ``full = False``.
            **kwargs: Passed to :meth:`evaluate`, e.g. ``wn``.

        Returns:
            (General, dict): The decomposed object and the stored values ``"logdet"``, ``"factor"`` (the Cholesky
            factor) and ``"hp"``.

        """
        # By default returns the lower triangular Cholesky factor
        # i.e. K = L @ L.T
        # Matches with tinygp default but not scipy or jax.scipy default
        
        self.K = self.evaluate(x, x, full = full, row_idx = idx, col_idx = idx, **kwargs)
        self.factor = JLA.cholesky(self.K, lower=True)
        self.logdet = 2*jnp.log(jnp.diag(self.factor)).sum()
        
        return self, {"logdet":self.logdet, "factor":self.factor, "hp":self.hp}

    def matrix_inv_sqrt(self, R, transpose=0, **kwargs):
        r"""Solves for :math:`\mathbf{L}^{-1}\mathbf{R}` (or :math:`\mathbf{L}^{-T}\mathbf{R}`) with the Cholesky factor."""

        R_prime = jax.scipy.linalg.solve_triangular(self.factor, R, trans=transpose, lower=True)

        return R_prime
    
    def matrix_sqrt(self, R, transpose=0, **kwargs):
        r"""Multiplies by the Cholesky factor, :math:`\mathbf{L}\mathbf{R}` (or :math:`\mathbf{L}^T\mathbf{R}`)."""

        if transpose:
            return self.factor.transpose() @ R
        else:
            return self.factor @ R
        
    def scale(self, c):
        r"""Returns a ``General`` covariance object for :math:`c\mathbf{K}`."""
        return General(kf = lambda hp, x1, x2, **kwargs: c*self.evaluate(x1, x2, **kwargs))
    
    
    def householder_transform(self, x, u):
        r"""Returns a ``General`` covariance object for :math:`\mathbf{H}\mathbf{K}\mathbf{H}`, where
        :math:`\mathbf{H} = \mathbf{I} - 2\vec{u}\vec{u}^T` is the Householder reflection for the unit vector
        :math:`\vec{u}`. Calculated as
        :math:`\mathbf{K} - 2(\vec{u}\vec{w}^T + \vec{w}\vec{u}^T) + 4(\vec{u}\cdot\vec{w})\,\vec{u}\vec{u}^T`
        with :math:`\vec{w} = \mathbf{K}\vec{u}`.

        Args:
            x (JAXArray): Input locations.
            u (JAXArray): The unit Householder vector.

        Returns:
            General: The transformed covariance object.

        """
        w_i = self.matmul(x, x, u, full = True)
        u_dot_w = jnp.dot(u, w_i)

        def new_kf(hp, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs):
            K = self.kf(hp, x1, x2, full = full,
                        row_idx = row_idx, col_idx = col_idx, **kwargs)

            if row_idx is not None and col_idx is not None:
                u_row, u_col = u[row_idx], u[col_idx]
                w_i_row, w_i_col = w_i[row_idx], w_i[col_idx]
            elif full == True:
                u_row, u_col = u.copy(), u.copy()
                w_i_row, w_i_col = w_i.copy(), w_i.copy()
            else:
                raise Exception("Must specify row and col indices or else set full = True to evaluate the full covariance matrix")
            
            K += 4 * u_dot_w * jnp.outer(u_row, u_col)
            K -= 2 * (jnp.outer(u_row, w_i_col) + jnp.outer(w_i_row, u_col))
            return K
        
        return General(new_kf)

    def diagonal(self, x, full = True, idx = None, **kwargs):
        r"""Returns the diagonal of the dense matrix at the input locations ``x``."""
        self.K = self.evaluate(x, x, full = full, row_idx = idx, col_idx = idx, **kwargs)

        return jnp.diag(self.K)
    
    def __add__(self, other):
        r"""Returns a ``General`` covariance object for the sum with another covariance object, an array or a scalar."""
        if isinstance(other, CovType):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))
        elif isinstance(other, (jax.Array, np.ndarray)) or is_scalar(other):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other)
        else:
            return NotImplemented

    def __mul__(self, other) -> Kernel:
        r"""Returns a ``General`` covariance object for the element-wise product with another covariance object or
        an array, or the matrix scaled by a scalar.

        """
        if isinstance(other, CovType):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other.evaluate(x1, x2, **kwargs))
        elif is_scalar(other):
            return self.scale(other)
        elif isinstance(other, jax.Array) or isinstance(other, np.ndarray):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other)
        else:
            return NotImplemented


class Identity(CovType):
    r"""The identity matrix. Its decomposition, square roots and inverse are trivial and its log determinant
    is zero.

    """

    def __init__(self):
        r"""Creates the identity covariance object."""
        self.diag = 1.
        self.wn_diag = 0.
        self.logdet = 0.
        self.params = []
        self.opt_name = "WhiteNoiseKernel (1D)"

    def evaluate(self, x1, x2, row_idx = None, col_idx = None, full = True, **kwargs):
        r"""Returns the identity matrix of size ``x1.shape[-1]``, or the sub-matrix given by ``row_idx`` and ``col_idx`` if ``full = False``."""

        if full:
            mat = jnp.eye(x1.shape[-1])
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if idx_specified:
                mask = row_idx[:, None] == col_idx[None, :]
                mat = jnp.where(mask, 1.0, 0.0)
            else:
                raise Exception("""Cannot evaluate non-stationary diagonal covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                            )
        
        return mat
    
    def decompose(self, x, **kwargs):
        r"""Nothing to decompose. Returns itself and a log determinant of zero."""
        return self, {"logdet":0.}

    def matrix_inv_sqrt(self, R, transpose=0):
        r"""Returns ``R`` unchanged."""
        return R
        
    def matrix_sqrt(self, R, **kwargs):
        r"""Returns ``R`` unchanged."""
        return R
    
    def eigendecomp(self, x, **kwargs):
        r"""Returns eigenvalues of one and the identity matrix as the eigenvectors."""
        return jnp.ones(x.shape[-1]), jnp.eye(x.shape[-1])
    
    def inverse(self, R, **kwargs):
        r"""Returns ``R`` unchanged."""
        return R
        
    def dot_solve(self, R):
        r"""Returns :math:`\vec{r}^T\vec{r}`."""
        return jnp.square(R).sum()
        
    def inv_sqrt_transform(self, K, x, **kwargs):
        r"""Returns ``K`` unchanged, as the matrix square root is the identity."""
        return K
    
    def householder_transform(self, x, u):
        r"""Returns itself, as :math:`\mathbf{H}\mathbf{I}\mathbf{H} = \mathbf{I}`."""
        return self

    def scale(self, c):
        r"""Returns a ``ScaledIdentity`` with diagonal ``c``."""
        return ScaledIdentity(diag = c)

    def matmul(self, x1, x2, R, full = True, **kwargs):
        r"""Returns ``R`` if ``full = True``, otherwise multiplies by the sub-matrix given by ``row_idx`` and ``col_idx``."""
        if full:
            return R
        else:
            mat = self.evaluate(x1, x2, full = False, **kwargs)
            return mat @ R
    
    def __add__(self, other):
        r"""Adds the identity to the diagonal of ``other``, keeping its structure where possible: quasiseparable
        objects become ``GeneralQuasisepPlusNoise``, rank-one objects become ``OuterPlusScaledIdentity``,
        diagonal objects stay diagonal and ``General`` stays ``General``.

        """
        if isinstance(other, (GeneralQuasisep, GeneralQuasisepPlusNoise)):
            K_sum = GeneralQuasisepPlusNoise(other.tinygp_kf, diag = other.diag + 1., wn_diag = other.wn_diag, use_block = other.use_block, noise_model = other.noise_model)

        elif isinstance(other, Outer):
            K_sum = OuterPlusScaledIdentity(other.alpha_init, diag = 1.)

        elif isinstance(other, OuterPlusScaledIdentity):
            K_sum = OuterPlusScaledIdentity(other.alpha_init, diag = 1. + other.diag, wn_diag = other.wn_diag)

        elif isinstance(other, Identity):
            K_sum = ScaledIdentity(diag = 2.)

        elif isinstance(other, (ScaledIdentity, Diagonal)):
            K_sum = type(other)(diag = other.diag + 1., wn_diag = other.wn_diag)

        elif isinstance(other, General):
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))

        else:
            return NotImplemented
            
        return K_sum
    
    def __mul__(self, other) -> Kernel:
        r"""Scales by a scalar. Returns ``NotImplemented`` for anything else."""
        if is_scalar(other):
            return self.scale(other)
        else:
            return NotImplemented


        
class ScaledIdentity(CovType):
    r"""A constant diagonal matrix, :math:`(d + d_{\mathrm{wn}})\mathbf{I}`.

    Args:
        diag (Scalar, optional): The value :math:`d` on the diagonal. Defaults to 0.
        wn_diag (Scalar, optional): The white noise value :math:`d_{\mathrm{wn}}` on the diagonal, only included
            when ``wn = True``. Defaults to 0.
        params (optional): Metadata about the parameters the object depends on.

    """
    def __init__(self, diag = 0., wn_diag = 0., params = None):
        r"""Creates the covariance object, see the class docstring for the arguments."""
        self.diag = diag
        self.wn_diag = wn_diag
        self.params = params
        self.opt_name = "WhiteNoiseKernel (1D)"

        if not is_scalar(self.diag) or not is_scalar(self.wn_diag):
            raise Exception("ScaledIdentity CovType only supports scalar diag and wn_diag values")
        
    def evaluate(self, x1, x2, wn = True, row_idx = None, col_idx = None, full = True, **kwargs):
        r"""Returns the constant diagonal matrix, or the sub-matrix given by ``row_idx`` and ``col_idx`` if ``full = False``."""

        if full:
            mat = (self.diag + wn*self.wn_diag) * jnp.eye(x1.shape[-1])
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if idx_specified:
                mask = row_idx[:, None] == col_idx[None, :]
                d = self.diag + wn*self.wn_diag
                mat = jnp.where(mask, d, 0.0)
            else:
                raise Exception("""Cannot evaluate non-stationary diagonal covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                            )
        return mat

    def decompose(self, x, wn = True, **kwargs):
        r"""Stores the square root of the diagonal value and the log determinant :math:`N\log(d + d_{\mathrm{wn}})`.

        Returns:
            (ScaledIdentity, dict): Itself and the stored value ``"logdet"``.

        """

        self.D = self.diag + wn*self.wn_diag
        self.factor = jnp.sqrt(self.D)
        self.logdet = x.shape[-1]*jnp.log(self.D)
        
        return self, {"logdet":self.logdet}

    def matrix_inv_sqrt(self, R, **kwargs):
        r"""Divides ``R`` by the square root of the diagonal value."""

        return R/self.factor

    def matrix_sqrt(self, R, **kwargs):
        r"""Multiplies ``R`` by the square root of the diagonal value."""

        return self.factor * R
        
    def eigendecomp(self, x, wn = True, **kwargs):
        r"""Returns the constant diagonal value as every eigenvalue and the identity matrix as the eigenvectors."""

        const_diag = self.diag + wn*self.wn_diag
        
        return const_diag*jnp.ones(x.shape[-1]), jnp.eye(x.shape[-1])

    def inv_sqrt_transform(self, K, x, **K_kwargs):
        r"""Returns ``K`` divided by the diagonal value. Requires :meth:`decompose` to have been called."""

        K_tilde = K.scale(1/self.D)

        return K_tilde

    def matmul(self, x1, x2, R, wn = True, full = True, **kwargs):
        r"""Multiplies ``R`` by the diagonal value, or by the sub-matrix given by ``row_idx`` and ``col_idx`` if ``full = False``."""

        if full:
            const = self.diag + wn*self.wn_diag
            return const * R
        else:
            # Could do something more efficient, return certain elements of R after scaling
            mat = self.evaluate(x1, x2, wn = wn, full = False, **kwargs)
            return mat @ R
        
    def householder_transform(self, x, u):
        r"""Returns itself, as a constant diagonal matrix is unchanged by a Householder reflection."""
        return self

    def scale(self, c):
        r"""Returns a ``ScaledIdentity`` with ``diag`` and ``wn_diag`` multiplied by ``c``."""
        return ScaledIdentity(diag = self.diag * c, wn_diag = self.wn_diag * c)

    def __add__(self, other):
        r"""Adds the constant diagonal to ``other``, keeping its structure where possible: quasiseparable objects
        become ``GeneralQuasisepPlusNoise``, constant diagonals stay ``ScaledIdentity``, ``Diagonal`` stays
        ``Diagonal``, rank-one objects become ``OuterPlusScaledIdentity`` and ``General`` stays ``General``.

        """
        if isinstance(other, (GeneralQuasisep, GeneralQuasisepPlusNoise)):
            K_sum = GeneralQuasisepPlusNoise(other.tinygp_kf, diag = self.diag + other.diag, wn_diag = self.wn_diag + other.wn_diag,
                                             noise_model = other.noise_model, use_block = other.use_block)
            
        elif isinstance(other, (Identity, ScaledIdentity)):
            K_sum = ScaledIdentity(diag = self.diag + other.diag, wn_diag = self.wn_diag + other.wn_diag)

        elif isinstance(other, Diagonal):
            K_sum = Diagonal(diag = self.diag + other.diag, wn_diag = self.wn_diag + other.wn_diag)

        elif isinstance(other, (Outer, OuterPlusScaledIdentity)):
            K_sum = OuterPlusScaledIdentity(other.alpha_init, diag = self.diag + other.diag, wn_diag = self.wn_diag + other.wn_diag)

        elif isinstance(other, General):
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))

        else:
            return NotImplemented
            
        return K_sum
    
    def __mul__(self, other) -> Kernel:
        r"""Scales by a scalar. Returns ``NotImplemented`` for anything else."""
        if is_scalar(other):
            return self.scale(other)
        else:
            return NotImplemented


class Diagonal(CovType):
    r"""A diagonal matrix whose diagonal can vary along the dimension, :math:`\mathrm{diag}(\vec{d} + \vec{d}_{\mathrm{wn}})`.

    Args:
        diag (JAXArray | Scalar, optional): The values :math:`\vec{d}` on the diagonal. Defaults to 0.
        wn_diag (JAXArray | Scalar, optional): The white noise values :math:`\vec{d}_{\mathrm{wn}}` on the
            diagonal, only included when ``wn = True``. Defaults to 0.

    At least one of ``diag`` and ``wn_diag`` must be a 1D array. Use ``ScaledIdentity`` for a constant diagonal.

    """
    def __init__(self, diag = 0., wn_diag = 0.):
        r"""Creates the covariance object, see the class docstring for the arguments."""
        self.N = jnp.size(diag + wn_diag)

        if jnp.ndim(diag + wn_diag) != 1:
            raise Exception("Diagonal CovType requires a 1D vector of diagonal values, not a scalar/array. Use ScaledIdentity for a scalar diagonal value.")
        
        self.diag = diag * jnp.ones(self.N)
        self.wn_diag = wn_diag * jnp.ones(self.N)
        self.opt_name = "WhiteNoiseKernel (1D)"
    
    def evaluate(self, x1, x2, wn = True, row_idx = None, col_idx = None, full = True, **kwargs):
        r"""Returns the diagonal matrix, or the sub-matrix given by ``row_idx`` and ``col_idx`` if ``full = False``."""

        if full:
            diag_mat = jnp.diag(self.diag + wn*self.wn_diag)
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if idx_specified:
                mask = row_idx[:, None] == col_idx[None, :]  # (n1, n2) bool matrix

                # Pick the diagonal values where they match
                diag_vals = self.diag + wn*self.wn_diag
                diag_mat = jnp.where(mask, diag_vals[row_idx[:, None] * mask], 0.0)
                
            else:
                raise Exception("""Cannot evaluate non-stationary diagonal covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                            )
        return diag_mat
    
    def decompose(self, x, wn = True, **kwargs):
        r"""Stores the square root of the diagonal and the log determinant, the sum of the log of the diagonal.

        Returns:
            (Diagonal, dict): Itself and the stored values ``"logdet"`` and ``"hp"`` (``diag`` and ``wn_diag``).

        """

        self.D = self.diag + wn*self.wn_diag
        self.factor = jnp.sqrt(self.D)
        self.logdet = jnp.log(self.D).sum()
        
        return self, {"logdet":self.logdet, "hp":{"diag":self.diag, "wn_diag":self.wn_diag}}
    
    def matrix_sqrt(self, R, **kwargs):
        r"""Multiplies each row of ``R`` by the square root of the corresponding diagonal value."""
        
        return jnp.einsum('i,i...->i...', self.factor, R)
        
    def matrix_inv_sqrt(self, R, transpose=0):
        r"""Divides each row of ``R`` by the square root of the corresponding diagonal value."""
        
        D_inv_sqrt = 1/self.factor

        return jnp.einsum('i,i...->i...', D_inv_sqrt, R)
    
    def eigendecomp(self, x, wn = True, **kwargs):
        r"""Returns the diagonal values as the eigenvalues and the identity matrix as the eigenvectors."""

        D = self.diag + wn*self.wn_diag
        
        # Not very memory efficient building a full identity matrix...
        return D, jnp.eye(x.shape[-1])
    
    def inv_sqrt_transform(self, K, x, **K_kwargs):
        r"""Returns :math:`\mathbf{D}^{-1/2}\mathbf{K}\mathbf{D}^{-1/2}` as a covariance object, keeping the structure
        of ``K`` where possible: diagonal objects stay diagonal, rank-one objects stay rank one, quasiseparable
        objects stay quasiseparable (with rescaled rows and columns) and ``General`` stays ``General``.
        Requires :meth:`decompose` to have been called.

        """

        if isinstance(K, (ScaledIdentity, Diagonal)):
            K_tilde = Diagonal(diag = K.diag/self.D, wn_diag = K.wn_diag/self.D)

        elif isinstance(K, Identity):
            K_tilde = Diagonal(diag = 1/self.D)

        elif isinstance(K, Outer):
            K_tilde = Outer(alpha = K.alpha_init/jnp.sqrt(self.D))
        
        elif isinstance(K, (GeneralQuasisep, GeneralQuasisepPlusNoise)):
            new_kf = ScaledKernel(kernel = K.tinygp_kf, amplitudes = 1/jnp.sqrt(self.D))
            new_diag = K.diag/self.D
            new_wn_diag = K.wn_diag/self.D

            # Yet to implement for the case of a general noise model added
            if K.noise_model is not None:
                raise NotImplementedError("Transforming a GeneralQuasisepPlusNoise CovType with a non-trivial noise model is not yet implemented")

            K_tilde = GeneralQuasisepPlusNoise(new_kf, diag = new_diag, wn_diag = new_wn_diag, use_block = K.use_block)

        elif isinstance(K, General):
            D_inv_sqrt = 1/jnp.sqrt(self.D)
            K_eval = jnp.outer(D_inv_sqrt, D_inv_sqrt) * K.evaluate(x, x, row_idx = K_kwargs.get("idx"), col_idx = K_kwargs.get("idx"), full = K_kwargs.get("full", True))
            # Define a new General covtype for K where it's kernel function transforms it
            def kf_transf(hp, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs):
                if full:
                    return K_eval
                else:
                    return K_eval[jnp.ix_(row_idx, col_idx)]

            K_tilde = General(kf_transf)

        else:
            if hasattr(K, "transform_with_inv_sqrt"):
                K_tilde = K.transform_with_inv_sqrt(self.matrix_inv_sqrt, x, **K_kwargs)
            else:
                raise NotImplementedError(f"Don't know how to whiten this CovType/Kernel of type {type(K).__name__} to K_tilde = L^-1 K L^-T")

        return K_tilde
    
    def inverse(self, R, transpose=0):
        r"""Divides each row of ``R`` by the corresponding diagonal value."""
        
        D_inv = 1/self.D

        return jnp.einsum('i,i...->i...', D_inv, R)

    def matmul(self, x1, x2, R, wn = True, full = True, **kwargs):
        r"""Multiplies each row of ``R`` by the corresponding diagonal value, or by the sub-matrix given by ``row_idx`` and ``col_idx`` if ``full = False``."""

        if full:
            D = self.diag + wn*self.wn_diag
            return jnp.einsum('i,i...->i...', D, R)
        else:
            D_mat = self.evaluate(x1, x2, wn = wn, full = False, **kwargs)
            return D_mat @ R

    def scale(self, c):
        r"""Returns a ``Diagonal`` with ``diag`` and ``wn_diag`` multiplied by ``c``."""
        return Diagonal(diag = self.diag * c, wn_diag = self.wn_diag * c)
    
    def householder_transform(self, x, u):
        r"""Returns :math:`\mathbf{H}\mathbf{D}\mathbf{H}` (see :meth:`General.householder_transform`) as a
        ``GeneralQuasisepPlusNoise`` object: the diagonal plus two rank-one terms, which keeps the matrix
        quasiseparable.

        """
        w_i = self.matmul(x, x, u, full = True)

        # Can this ever be super close to zero? Would be very numerically unstable but seems unlikely?
        u_dot_w = jnp.dot(u, w_i)
        alpha = 2 * u_dot_w * u - w_i

        new_tinygp_kf = luas.kernels.tinygp_ext.Amplitude(alpha = alpha, const = 1/u_dot_w)
        new_tinygp_kf += luas.kernels.tinygp_ext.Amplitude(alpha = w_i, const = -1/u_dot_w)
        
        return GeneralQuasisepPlusNoise(new_tinygp_kf, diag = self.diag, wn_diag = self.wn_diag)

    def __add__(self, other):
        r"""Adds the diagonal to ``other``: quasiseparable objects become ``GeneralQuasisepPlusNoise``, diagonal
        objects stay ``Diagonal``, and rank-one and ``General`` objects become ``General``.

        """
        if isinstance(other, (GeneralQuasisep, GeneralQuasisepPlusNoise)):
            K_sum = GeneralQuasisepPlusNoise(other.tinygp_kf, diag = self.diag + other.diag, wn_diag = self.wn_diag + other.wn_diag,
                                             noise_model = other.noise_model, use_block = other.use_block)

        elif isinstance(other, (Identity, ScaledIdentity, Diagonal)):
            K_sum = Diagonal(diag = self.diag + other.diag, wn_diag = self.wn_diag + other.wn_diag)

        elif isinstance(other, (Outer, OuterPlusScaledIdentity, General)):
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))

        else:
            return NotImplemented
            
        return K_sum

    def __mul__(self, other) -> Kernel:
        r"""Scales by a scalar. Returns ``NotImplemented`` for anything else."""
        if is_scalar(other):
            return self.scale(other)
        else:
            return NotImplemented


class GeneralQuasisepPlusNoise(CovType):
    r"""A quasiseparable matrix plus a diagonal, and optionally a quasiseparable noise model, using ``tinygp``'s
    quasiseparable solvers. Decompositions and matrix products cost :math:`\mathcal{O}(N)`.

    Args:
        tinygp_kf (optional): A ``tinygp`` quasiseparable kernel taking ``(x, index)`` coordinates (see
            ``luas.kernels.tinygp_ext.HandleIdx``), or ``None`` for no kernel term.
        diag (JAXArray | Scalar, optional): Value(s) added to the diagonal. Defaults to 0.
        wn_diag (JAXArray | Scalar, optional): White noise value(s) added to the diagonal, only included when
            ``wn = True``. Defaults to 0.
        noise_model (SymmQSM, optional): An extra symmetric quasiseparable matrix added to the covariance,
            e.g. a banded noise model.
        params (optional): Metadata about the parameters the object depends on.
        use_block (bool, optional): Whether to use ``tinygp``'s block quasiseparable representation when
            kernels are summed. Defaults to ``True``.

    """
    def __init__(self, tinygp_kf, diag = 0., wn_diag = 0., noise_model = None, params = None, use_block = True):
        r"""Creates the covariance object, see the class docstring for the arguments."""
        self.tinygp_kf = tinygp_kf
        self.diag = diag
        self.wn_diag = wn_diag
        self.noise_model = noise_model
        self.use_block = use_block
        self.params = params
        self.opt_name = "Quasiseparable (1D)"


    def _tinygp_coords(self, x1, x2, row_idx = None, col_idx = None, full = True):
        r"""Builds the ``(x, index)`` coordinates passed to the ``tinygp`` kernel, so that kernel terms can depend on
        the position in the matrix (e.g. rank-one ``Amplitude`` terms). Uses all indices if ``full = True``,
        otherwise ``row_idx`` and ``col_idx``.

        """

        if full:
            X1 = (x1, jnp.arange(x1.shape[-1]))
            X2 = (x2, jnp.arange(x2.shape[-1]))
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if idx_specified:
                # Potentially inefficient to build full diag_mat but shouldn't be doing this often
                X1 = (x1, row_idx)
                X2 = (x2, col_idx)
            else:
                raise Exception("""Cannot evaluate non-stationary diagonal covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                            )
        return X1, X2


    def _to_symm_qsm(self, x, wn = True, full = True, idx = None, stored_values = {}):
        r"""Builds the matrix as a ``tinygp`` ``SymmQSM``: the diagonal, plus the noise model, plus the kernel term.

        Args:
            x (JAXArray): Input locations.
            wn (bool, optional): Whether to include white noise. Defaults to ``True``.
            full (bool, optional): Whether to build the full matrix, otherwise the sub-matrix given by ``idx``.
            idx (JAXArray, optional): Indices of the sub-matrix if ``full = False``.
            stored_values (dict, optional): Not used.

        Returns:
            SymmQSM: The symmetric quasiseparable matrix.

        """
        if full:
            idx = jnp.arange(x.shape[-1])
        else:
            if idx is None:
                raise Exception("Must specify the row and col indices to evaluate the covariance matrix when full = False")

        diag = (self.diag + wn*self.wn_diag)
        
        if is_scalar(diag):
            diag = diag*jnp.ones(x.shape[-1])
        else:
            diag = diag[idx]
        
        qsm_matrix = tinygp.noise.Diagonal(diag=diag).to_qsm()

        if self.noise_model is not None:
            qsm_matrix += self.noise_model

        if self.tinygp_kf is not None:
            qsm_matrix += self.tinygp_kf.to_symm_qsm((x, idx))
    
        return qsm_matrix
    
    def decompose(self, x, wn = True, full = True, **kwargs):
        r"""Calculates the quasiseparable Cholesky decomposition and the log determinant.

        Returns:
            (GeneralQuasisepPlusNoise, dict): Itself and the stored value ``"logdet"``.

        """

        matrix = self._to_symm_qsm(x, wn = wn, full = full, **kwargs)
        self.factor = matrix.cholesky()
        self.logdet = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
        return self, {"logdet":self.logdet}
        
    def matrix_inv_sqrt(self, R, transpose=0):
        r"""Solves for :math:`\mathbf{L}^{-1}\mathbf{R}` (or :math:`\mathbf{L}^{-T}\mathbf{R}`) with the quasiseparable Cholesky factor."""

        if transpose:
            R_prime = self.factor.transpose().solve(R)
        else:
            R_prime = self.factor.solve(R)
            
        return R_prime

    def matrix_sqrt(self, R, transpose=0, **kwargs):
        r"""Multiplies by the quasiseparable Cholesky factor, :math:`\mathbf{L}\mathbf{R}` (or :math:`\mathbf{L}^T\mathbf{R}`)."""

        if transpose:
            return self.factor.transpose() @ R
        else:
            return self.factor @ R
    
    def evaluate(self, x1, x2, wn = True, row_idx = None, col_idx = None, full = True, **kwargs):
        r"""Evaluates the dense matrix: the kernel term plus the diagonal and any noise model."""

        X1, X2 = self._tinygp_coords(x1, x2, row_idx = row_idx, col_idx = col_idx, full = full)

        kernel_model = self.tinygp_kf(X1, X2)

        if is_scalar(self.diag + self.wn_diag):
            diag_cov = ScaledIdentity(diag = self.diag, wn_diag = self.wn_diag)
        else:
            diag_cov = Diagonal(diag = self.diag, wn_diag = self.wn_diag)

        noise_model = diag_cov.evaluate(x1, x2, wn = wn, row_idx = row_idx, col_idx = col_idx, full = full)
        
        if self.noise_model is not None:
            if full:
                noise_model += self.noise_model.to_dense()
            else:
                qsm_model = self.noise_model.to_dense()
                noise_model += qsm_model[jnp.ix_(row_idx, col_idx)]
        
        return kernel_model + noise_model
    

    def matmul(self, x1, x2, R, wn = True, full = True, **kwargs):
        r"""Multiplies ``R`` by the matrix in :math:`\mathcal{O}(N)` without building it. Assumes ``x1`` and ``x2`` are
        the locations the diagonal and noise model are defined at.

        """
        # Assumes matrix is evaluated at positions of diagonal, noise_model

        X1 = (x1, jnp.arange(x1.shape[-1]))
        X2 = (x2, jnp.arange(x2.shape[-1]))
        
        diag_eval = (self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1])
        R_prime = jnp.einsum('i,i...->i...', diag_eval, R)

        if self.tinygp_kf is not None:
            if full:
                R_prime += self.tinygp_kf.matmul(X1, R)
            else:
                R_prime += self.tinygp_kf.matmul(X1, X2, R)

        if self.noise_model is not None:
            R_prime += self.noise_model @ R
        
        return R_prime


    def scale(self, c):
        r"""Returns a ``GeneralQuasisepPlusNoise`` with the kernel, diagonal and noise model multiplied by ``c``."""
        if self.tinygp_kf is not None: 
            scaled_kernel = luas.kernels.tinygp_ext.Scale(kernel = self.tinygp_kf, scale = c)
        else:
            scaled_kernel = None

        if self.noise_model is None:
                new_noise_model = None
        else:
            new_noise_model = self.noise_model * c

        return GeneralQuasisepPlusNoise(scaled_kernel, diag = self.diag * c, wn_diag = self.wn_diag * c,
                                        noise_model = new_noise_model, use_block = self.use_block)
    

    def householder_transform(self, x, u):
        r"""Returns :math:`\mathbf{H}\mathbf{K}\mathbf{H}` (see :meth:`General.householder_transform`) by adding two
        rank-one terms to the kernel, which keeps the matrix quasiseparable.

        """
        w_i = self.matmul(x, x, u, full = True)

        # Can this ever be super close to zero? Would be very numerically unstable but seems unlikely?
        u_dot_w = jnp.dot(u, w_i)
        alpha = 2 * u_dot_w * u - w_i

        new_tinygp_kf = self.tinygp_kf + luas.kernels.tinygp_ext.Amplitude(alpha = alpha, const = 1/u_dot_w)
        new_tinygp_kf += luas.kernels.tinygp_ext.Amplitude(alpha = w_i, const = -1/u_dot_w)
        
        return GeneralQuasisepPlusNoise(new_tinygp_kf, diag = self.diag, wn_diag = self.wn_diag,
                                        noise_model = self.noise_model, use_block = self.use_block)
    

    def __add__(self, other):
        r"""Adds ``other`` while keeping the matrix quasiseparable where possible: diagonal objects, other
        quasiseparable objects, ``SymmQSM`` noise models and rank-one objects all give a
        ``GeneralQuasisepPlusNoise``. Adding a ``General`` object gives a ``General`` object.

        """

        if isinstance(other, (Identity, ScaledIdentity, Diagonal)):
            K_sum = GeneralQuasisepPlusNoise(self.tinygp_kf, diag = self.diag + other.diag, wn_diag = self.wn_diag + other.wn_diag,
                                            noise_model = self.noise_model, use_block = self.use_block)

        elif isinstance(other, GeneralQuasisep):
            if self.tinygp_kf is None:
                new_tinygp_kf = other.tinygp_kf
            elif other.tinygp_kf is None:
                new_tinygp_kf = self.tinygp_kf
            else:
                new_tinygp_kf = luas.kernels.tinygp_ext.Sum(self.tinygp_kf, other.tinygp_kf, use_block = self.use_block)
                
            K_sum = GeneralQuasisepPlusNoise(new_tinygp_kf, diag = self.diag, wn_diag = self.wn_diag,
                                             noise_model = self.noise_model, use_block = self.use_block)

        elif isinstance(other, (GeneralQuasisepPlusNoise)):
            if self.tinygp_kf is None:
                new_tinygp_kf = other.tinygp_kf
            elif other.tinygp_kf is None:
                new_tinygp_kf = self.tinygp_kf
            else:
                new_tinygp_kf = luas.kernels.tinygp_ext.Sum(self.tinygp_kf, other.tinygp_kf, use_block = self.use_block)

            if self.noise_model is not None and other.noise_model is not None:
                new_noise_model = self.noise_model + other.noise_model
            elif self.noise_model is not None:
                new_noise_model = self.noise_model
            elif other.noise_model is not None:
                new_noise_model = other.noise_model
            else:
                new_noise_model = None

            K_sum = GeneralQuasisepPlusNoise(new_tinygp_kf, diag = self.diag + other.diag, wn_diag = self.wn_diag + other.wn_diag,
                                             noise_model = new_noise_model, use_block = self.use_block)
            
        elif isinstance(other, SymmQSM):
            if self.noise_model is None:
                new_noise_model = other
            else:
                new_noise_model = other + self.noise_model

            K_sum = GeneralQuasisepPlusNoise(self.tinygp_kf, use_block = self.use_block,
                                            diag = self.diag, wn_diag = self.wn_diag, noise_model = new_noise_model)
            
        elif isinstance(other, (Outer, OuterPlusScaledIdentity)):
            if self.tinygp_kf is None:
                new_tinygp_kf = other.tinygp_kf
            else:
                new_tinygp_kf = luas.kernels.tinygp_ext.Sum(self.tinygp_kf, other.tinygp_kf, use_block = self.use_block)

            K_sum = GeneralQuasisepPlusNoise(new_tinygp_kf, diag = self.diag + other.diag, wn_diag = self.wn_diag + other.wn_diag,
                                             use_block = self.use_block, noise_model = self.noise_model)

        elif isinstance(other, (General)):
            # Should update as don't lose quasiseparability for Outer
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))

        else:
            return NotImplemented
            
        return K_sum

        
    def __mul__(self, other) -> Kernel:
        r"""Multiplies by a scalar, a stationary ``GeneralQuasisep`` (the product of the two kernels, with the
        diagonal scaled by the other kernel's variance), or an ``Outer`` object (scaling the rows and columns).

        Raises:
            Exception: For products which aren't implemented, e.g. when either term has a noise model.

        """
        if is_scalar(other):
            K_mult = self.scale(other)

        elif isinstance(other, GeneralQuasisep):
            if self.noise_model is None:
                if self.tinygp_kf is None or other.tinygp_kf is None:
                    raise Exception("Multiplication of GeneralQuasisepPlusNoise CovType with a GeneralQuasisep CovType without both having kernel functions (somehow?)")
                else:
                    new_tinygp_kf = luas.kernels.tinygp_ext.Product(self.tinygp_kf, other.tinygp_kf)
        
                new_diag = self.diag * other.tinygp_kf.evaluate_diag((jnp.zeros(1), 0))
                new_wn_diag = self.wn_diag * other.tinygp_kf.evaluate_diag((jnp.zeros(1), 0))
                
                K_mult = GeneralQuasisepPlusNoise(new_tinygp_kf, diag = new_diag, wn_diag = new_wn_diag,
                                                use_block = self.use_block, noise_model = None)
            else:
                raise Exception("Multiplication of GeneralQuasisepPlusNoise CovType with a GeneralQuasisep CovType with a non-trivial noise model is not yet implemented")
            
        elif isinstance(other, Outer):
            if is_scalar(other.alpha_init):
                K_mult = self.scale(other.alpha_init**2)
            else:
                new_kf = ScaledKernel(kernel = self.tinygp_kf, amplitudes = other.alpha_init)

                if self.noise_model is not None:
                    diag_term = DiagQSM(d=self.noise_model.diag.d * other.alpha_init**2)

                    p_new = jnp.einsum('i,i...->i...', other.alpha_init, self.noise_model.lower.p)
                    q_new = jnp.einsum('i,i...->i...', other.alpha_init, self.noise_model.lower.q)
                    lower_term = StrictLowerTriQSM(p=p_new, q=q_new, a=self.noise_model.lower.a)

                    new_noise_model = SymmQSM(diag=diag_term, lower=lower_term)
                else:
                    new_noise_model = None

                K_mult = GeneralQuasisepPlusNoise(new_kf, diag = self.diag * other.alpha_init**2, wn_diag = self.wn_diag * other.alpha_init**2,
                                                use_block = self.use_block, noise_model = new_noise_model)

        elif isinstance(other, GeneralQuasisepPlusNoise):
            raise Exception("Multiplication of Quasisep kernels which are non-stationary or include a noise model not yet implemented")
        else:
            return NotImplemented
        
        return K_mult


class GeneralQuasisep(CovType):
    r"""A quasiseparable matrix from a ``tinygp`` quasiseparable kernel, with nothing added to the diagonal.
    Decompositions and matrix products cost :math:`\mathcal{O}(N)`.

    Args:
        tinygp_kf: A ``tinygp`` quasiseparable kernel taking ``(x, index)`` coordinates (see
            ``luas.kernels.tinygp_ext.HandleIdx``).
        params (optional): Metadata about the parameters the object depends on.
        use_block (bool, optional): Whether to use ``tinygp``'s block quasiseparable representation when
            kernels are summed. Defaults to ``True``.

    """
    def __init__(self, tinygp_kf, params = None, use_block = True):
        r"""Creates the covariance object, see the class docstring for the arguments."""
        self.tinygp_kf = tinygp_kf
        self.diag = 0.
        self.wn_diag = 0.
        self.use_block = use_block
        self.noise_model = None
        self.params = params
        self.opt_name = "Quasiseparable (1D)"

    def _to_symm_qsm(self, x, wn = True, idx = None, stored_values = {}):
        r"""Builds the matrix at ``x`` (or the indices ``idx``) as a ``tinygp`` ``SymmQSM``."""

        if idx is None:
            idx = jnp.arange(x.shape[-1])

        matrix = self.tinygp_kf.to_symm_qsm((x, idx))
        return matrix
    
    def _tinygp_coords(self, x1, x2, row_idx = None, col_idx = None, full = True):
        r"""Builds the ``(x, index)`` coordinates passed to the ``tinygp`` kernel, always using all indices."""

        X1 = (x1, jnp.arange(x1.shape[-1]))
        X2 = (x2, jnp.arange(x2.shape[-1]))

        return X1, X2

    def evaluate(self, x1, x2, wn = True, **kwargs):
        r"""Evaluates the dense matrix from the ``tinygp`` kernel."""

        X1, X2 = self._tinygp_coords(x1, x2, **kwargs)
        return self.tinygp_kf(X1, X2)
    
    def decompose(self, x, wn = True, **kwargs):
        r"""Calculates the quasiseparable Cholesky decomposition and the log determinant.

        Returns:
            (GeneralQuasisep, dict): Itself and the stored value ``"logdet"``.

        """

        matrix = self._to_symm_qsm(x, wn = wn, **kwargs)
        self.factor = matrix.cholesky()
        self.logdet = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
        return self, {"logdet":self.logdet}
        
    def matrix_inv_sqrt(self, R, transpose=0):
        r"""Solves for :math:`\mathbf{L}^{-1}\mathbf{R}` (or :math:`\mathbf{L}^{-T}\mathbf{R}`) with the quasiseparable Cholesky factor."""

        if transpose:
            R_prime = self.factor.transpose().solve(R)
        else:
            R_prime = self.factor.solve(R)
            
        return R_prime

    def matrix_sqrt(self, R, transpose=0, **kwargs):
        r"""Multiplies by the quasiseparable Cholesky factor, :math:`\mathbf{L}\mathbf{R}` (or :math:`\mathbf{L}^T\mathbf{R}`)."""

        if transpose:
            return self.factor.transpose() @ R
        else:
            return self.factor @ R

    def scale(self, c):
        r"""Returns a ``GeneralQuasisep`` with the kernel multiplied by ``c``."""
        scaled_kernel = luas.kernels.tinygp_ext.Scale(kernel = self.tinygp_kf, scale = c)
        return GeneralQuasisep(scaled_kernel, use_block = self.use_block)
    
    def matmul(self, x1, x2, R, wn = True, full = True, **kwargs):
        r"""Multiplies ``R`` by the matrix in :math:`\mathcal{O}(N)` without building it."""

        X1, X2 = self._tinygp_coords(x1, x2, full = full, **kwargs)

        if full:
            R_prime = self.tinygp_kf.matmul(X1, R)
        else:
            R_prime = self.tinygp_kf.matmul(X1, X2, R)
        
        return R_prime
    
    def householder_transform(self, x, u):
        r"""Returns :math:`\mathbf{H}\mathbf{K}\mathbf{H}` (see :meth:`General.householder_transform`) as a
        ``GeneralQuasisepPlusNoise`` by adding two rank-one terms to the kernel, which keeps it quasiseparable.

        """
        w_i = self.matmul(x, x, u, full = True)

        # Can this ever be super close to zero? Would be very numerically unstable but seems unlikely?
        u_dot_w = jnp.dot(u, w_i)
        alpha = 2 * u_dot_w * u - w_i

        new_tinygp_kf = self.tinygp_kf + luas.kernels.tinygp_ext.Amplitude(alpha = alpha, const = 1/u_dot_w)
        new_tinygp_kf += luas.kernels.tinygp_ext.Amplitude(alpha = w_i, const = -1/u_dot_w)
        
        return GeneralQuasisepPlusNoise(new_tinygp_kf, use_block = self.use_block)

    def __add__(self, other):
        r"""Adds ``other`` while keeping the matrix quasiseparable where possible: diagonal, quasiseparable,
        ``SymmQSM`` and rank-one terms give a ``GeneralQuasisep`` or ``GeneralQuasisepPlusNoise``. Adding a
        ``General`` object gives a ``General`` object.

        """

        if isinstance(other, Identity) or isinstance(other, ScaledIdentity) or isinstance(other, Diagonal):
            K_sum = GeneralQuasisepPlusNoise(self.tinygp_kf, diag = other.diag, wn_diag = other.wn_diag, use_block = self.use_block)

        elif isinstance(other, GeneralQuasisep):
            sum_kernel = luas.kernels.tinygp_ext.Sum(self.tinygp_kf, other.tinygp_kf, use_block = self.use_block)
            K_sum = GeneralQuasisep(sum_kernel, use_block = self.use_block)

        elif isinstance(other, GeneralQuasisepPlusNoise):
            if other.tinygp_kf is not None:
                new_tinygp_kf = luas.kernels.tinygp_ext.Sum(self.tinygp_kf, other.tinygp_kf, use_block = self.use_block)
            else:
                new_tinygp_kf = self.tinygp_kf
            K_sum = GeneralQuasisepPlusNoise(new_tinygp_kf, diag = other.diag, wn_diag = other.wn_diag, noise_model = other.noise_model, use_block = self.use_block)

        elif isinstance(other, SymmQSM):
            K_sum = GeneralQuasisepPlusNoise(self.tinygp_kf, use_block = self.use_block, noise_model = other)

        elif isinstance(other, (Outer, OuterPlusScaledIdentity)):
            K_sum = GeneralQuasisepPlusNoise(self.tinygp_kf + other.tinygp_kf, diag = other.diag, wn_diag = other.wn_diag, use_block = self.use_block)

        elif isinstance(other, (General)):
            # Should update as don't lose quasiseparability
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))

        else:
            return NotImplemented
            
        return K_sum
    
    def __mul__(self, other) -> Kernel:
        r"""Multiplies by another ``GeneralQuasisep`` (the product of the two kernels), a scalar, a
        ``GeneralQuasisepPlusNoise`` without a noise model, or an ``Outer`` object (scaling the rows and columns).

        Raises:
            Exception: For products which aren't implemented, e.g. with a noise model.

        """
        if isinstance(other, GeneralQuasisep):
            product_kernel = luas.kernels.tinygp_ext.Product(kernel1 = self.tinygp_kf, kernel2 = other.tinygp_kf)
            K_mult = GeneralQuasisep(product_kernel, use_block = self.use_block)
        
        elif is_scalar(other):
            K_mult = self.scale(other)
        
        elif isinstance(other, GeneralQuasisepPlusNoise):
            if other.noise_model is None:
                if self.tinygp_kf is None or other.tinygp_kf is None:
                    raise Exception("Multiplication of GeneralQuasisepPlusNoise CovType with a GeneralQuasisep CovType without both having kernel functions (somehow?)")
                else:
                    K_mult = luas.kernels.tinygp_ext.Product(self.tinygp_kf, other.tinygp_kf)
                    
                new_diag = other.diag * self.tinygp_kf.evaluate_diag((jnp.zeros(1), 0))
                new_wn_diag = other.wn_diag * self.tinygp_kf.evaluate_diag((jnp.zeros(1), 0))
                
                K_mult = GeneralQuasisepPlusNoise(K_mult, diag = new_diag, wn_diag = new_wn_diag,
                                                use_block = self.use_block, noise_model = None)
            else:
                raise Exception("Multiplication of Quasisep kernels which are non-stationary or include a noise model not yet implemented")
            
        elif isinstance(other, Outer):
            if is_scalar(other.alpha_init):
                K_mult = self.scale(other.alpha_init**2)
            else:
                new_kf = ScaledKernel(kernel = self.tinygp_kf, amplitudes = other.alpha_init)
                K_mult = GeneralQuasisepPlusNoise(new_kf, use_block = self.use_block)

        else:
            return NotImplemented
        
        return K_mult
    

class PeriodicQuasisep(GeneralQuasisep):
    r"""A quasiseparable matrix from a periodic ``tinygp`` kernel (e.g. ``Cosine``). Behaves like
    ``GeneralQuasisep``, but is its own type so periodic kernels can be told apart, and scaling keeps that type.

    """
    def scale(self, c):
        r"""Returns a ``PeriodicQuasisep`` with the kernel multiplied by ``c``."""
        scaled_kernel = luas.kernels.tinygp_ext.Scale(kernel = self.tinygp_kf, scale = c)
        return PeriodicQuasisep(scaled_kernel, use_block = self.use_block)


class Exp(GeneralQuasisep):
    r"""The exponential kernel as a quasiseparable matrix, which can also use a fast eigendecomposition.

    The inverse of an exponential kernel matrix is tridiagonal, so its eigendecomposition can be calculated
    quickly with a tridiagonal eigensolver (see :meth:`eigendecomp_fast`). This uses ``scipy`` and is not
    differentiable, so it is off by default.

    Args:
        tinygp_kf: The ``tinygp`` exponential kernel, taking ``(x, index)`` coordinates.
        scale (Scalar): The length scale :math:`l` of the kernel.
        sigma (Scalar, optional): The amplitude of the kernel. Defaults to 1.
        params (optional): Metadata about the parameters the object depends on.
        use_block (bool, optional): Whether to use ``tinygp``'s block quasiseparable representation when
            kernels are summed. Defaults to ``True``.
        fast_eigen (bool, optional): Whether to use :meth:`eigendecomp_fast` for eigendecompositions. Gradients
            are not implemented for it, and a warning is raised. Defaults to ``False``.

    """
    def __init__(self, tinygp_kf, scale, sigma = 1., params = None, use_block = True, fast_eigen = False):
        r"""Creates the covariance object, see the class docstring for the arguments."""
        # Note that this is a special case of the GeneralQuasisep class where the kernel is an exponential kernel
        # It has a special eigendecomposition method which is faster than the general case
        # But since there isn't a fast JAX implementation so far, but scipy version won't be differentiable,
        # so gradients won't work for this method with fast_eigen = True
        self.len_scale = scale
        self.sigma = sigma
        self.tinygp_kf = tinygp_kf
        self.diag = 0.
        self.wn_diag = 0.
        self.noise_model = None
        self.use_block = use_block
        self.params = params
        self.fast_eigen = fast_eigen
        self.opt_name = "Quasiseparable (1D)"

        if self.fast_eigen:
            # Note gradients not implemented yet for this method
            self.eigendecomp = self.eigendecomp_fast
            warnings.warn("Using fast_eigen = True for Exp kernel, but gradients not implemented for this method yet")

    def scale(self, c):
        r"""Returns an ``Exp`` object for :math:`c\mathbf{K}`, with the same length scale and the amplitude multiplied by :math:`\sqrt{c}`."""
        scaled_kernel = luas.kernels.tinygp_ext.Scale(kernel = self.tinygp_kf, scale = c)
        return Exp(scaled_kernel, self.len_scale, sigma = self.sigma * jnp.sqrt(c),
                   use_block = self.use_block, fast_eigen = self.fast_eigen)

    def eigendecomp_fast(self, x, **kwargs):
        r"""Calculates the eigendecomposition using the tridiagonal inverse of the exponential kernel matrix, via
        :meth:`fast_exp_eigh_scipy` called with ``jax.pure_callback``. Not differentiable.

        Returns:
            (JAXArray, JAXArray): The eigenvalues and a matrix whose columns are the eigenvectors.

        """
        N = x.shape[-1]
        result_shape = (jax.ShapeDtypeStruct((N,), x.dtype), jax.ShapeDtypeStruct((N, N), x.dtype))
        lam, Q = jax.pure_callback(self.fast_exp_eigh_scipy, result_shape, x, self.len_scale)
        return self.sigma**2 * lam, Q
        
    def fast_exp_eigh_scipy(self, x1, scale):
        r"""Calculates the eigendecomposition of a unit amplitude exponential kernel matrix with ``scipy``, by applying
        ``scipy.linalg.eigh_tridiagonal`` to its tridiagonal inverse. Uses the first row of ``x1`` if it is 2D.

        Args:
            x1 (numpy.ndarray): Input locations, which must be sorted.
            scale (Scalar): The length scale of the kernel.

        Returns:
            (numpy.ndarray, numpy.ndarray): The eigenvalues of the kernel matrix and its eigenvectors.

        """
    
        if x1.ndim == 2:
            x1 = x1[0, :]
            
        r = np.exp(-np.diff(x1/scale))
        e = 1/(1/r - r)
        
        arr_diag = np.zeros(x1.shape[-1])
        arr_diag[:-1] = 1 + r*e
        arr_diag[1:] += r*e
        arr_diag[-1] += 1
        
        lam, Q = scipy.linalg.eigh_tridiagonal(arr_diag, -e)
        
        return 1/lam, Q
    


class Outer(CovType):
    r"""A rank-one matrix :math:`\vec{\alpha}\vec{\alpha}^T`, or a constant matrix with every entry
    :math:`\alpha^2` if :math:`\alpha` is a scalar.

    The matrix isn't invertible, so it can't be used in the first (noise) term returned by a kernel function.
    Its eigendecomposition is known analytically using a Householder reflection.

    Args:
        alpha (JAXArray | Scalar): The vector :math:`\vec{\alpha}`, either a 1D array or a scalar.
        params (optional): Metadata about the parameters the object depends on.

    """
    def __init__(self, alpha, params = None):
        r"""Creates the covariance object, see the class docstring for the arguments."""
        self.alpha_init = alpha
        self.diag = 0.
        self.wn_diag = 0.
        self.params = params
        self.opt_name = "Not a valid kernel!"

        if is_scalar(alpha):
            self.tinygp_kf = HandleIdx(luas.kernels.tinygp_ext.Constant(const = alpha**2))
        else:
            self.tinygp_kf = luas.kernels.tinygp_ext.Amplitude(alpha = alpha)

        if not is_scalar(self.alpha_init) and self.alpha_init.ndim != 1:
            raise Exception("Outer CovType only supports scalar or 1D array alpha_init")

    def rank(self, x):
        r"""Returns 1."""
        return 1

    def evaluate(self, x1, x2, wn = True, row_idx = None, col_idx = None, full = True, **kwargs):
        r"""Returns :math:`\vec{\alpha}\vec{\alpha}^T`, or the sub-matrix given by ``row_idx`` and ``col_idx`` if ``full = False``."""

        if is_scalar(self.alpha_init):
            mat = self.alpha_init**2 * jnp.ones((x1.shape[-1], x2.shape[-1]))
        elif full:
            mat = jnp.outer(self.alpha_init, self.alpha_init)
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if idx_specified:
                mat = jnp.outer(self.alpha_init[row_idx], self.alpha_init[col_idx])
            else:
                raise Exception("""Cannot evaluate non-stationary outer product covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                                )
        return mat

    def decompose(self, x, **kwargs):
        r"""Nothing needs decomposing. Stores :math:`\vec{\alpha}` (expanded to the number of locations in ``x`` if it
        is a scalar) and returns a log determinant of :math:`-\infty`, as the matrix is singular.

        """
        # Matrix is decomposed by definition
        # Except need to handle ConstantKernel where self.alpha_init is a float

        if is_scalar(self.alpha_init):
            self.alpha = self.alpha_init * jnp.ones(x.shape[-1])
        else:
            self.alpha = self.alpha_init

        return self, {"logdet":-jnp.inf}
    
    def matrix_sqrt(self, R, **kwargs):
        r"""Not implemented for a rank-one matrix.

        Raises:
            Exception: Always.

        """

        raise Exception(f"Matrix square root for type {type(R)} of shape {R.shape} not implemented")

    def matrix_inv_sqrt(self, R, **kwargs):
        r"""Not available, as the matrix is not invertible.

        Raises:
            Exception: Always. This usually means a low-rank kernel was used in the first (noise) term returned
                by the kernel function.

        """

        raise Exception("Outer product matrix is not invertible! You may have specified a low-rank kernel"\
                        " as the first argument returned by kf which is incorrect usage, make sure the first argument returned consists of positive definite terms"\
                        " e.g. by adding Noise or KroneckerDelta terms")
    
    def inv_sqrt_transform(self, K, x, **kwargs):
        r"""Not available, as the matrix is not invertible.

        Raises:
            Exception: Always. This usually means a low-rank kernel was used in the first (noise) term returned
                by the kernel function.

        """

        raise Exception("Outer product matrix is not invertible! You may have specified a low-rank kernel"\
                        " as the first argument returned by kf which is incorrect usage, make sure the first argument returned consists of positive definite terms"\
                        " e.g. by adding Noise or KroneckerDelta terms")

    def scale(self, c):
        r"""Returns an ``Outer`` object for :math:`c\vec{\alpha}\vec{\alpha}^T`, i.e. with :math:`\vec{\alpha}` multiplied by :math:`\sqrt{c}`."""
        return Outer(self.alpha_init * jnp.sqrt(c))

    def eigendecomp(self, x, i = -1, **kwargs):
        r"""Calculates the eigendecomposition with a Householder reflection which maps :math:`\vec{\alpha}` onto the
        ``i``-th basis vector. The only non-zero eigenvalue is :math:`|\vec{\alpha}|^2`.

        Returns:
            (JAXArray, HouseholderTransform): The eigenvalues and the Householder reflection, whose columns are the
            eigenvectors.

        """
        # Calculates the Householder transformation of alpha for the eigenvector matrix
        # And the matrix's sole non-zero eigenvalue which is the squared norm of alpha

        H = HouseholderTransform(self.alpha_init * jnp.ones(x.shape[-1]), i = i)
        
        return H.lam, H
    
    def matmul(self, x1, x2, R, **kwargs):
        r"""Multiplies ``R`` by the matrix as :math:`\vec{\alpha}(\vec{\alpha}^T\mathbf{R})` without building it."""

        if is_scalar(self.alpha_init):
            return jnp.outer(jnp.ones(x1.shape[-1]) * self.alpha_init**2, R.sum(0))
        else:
            self.vec1 = self.alpha_init * jnp.ones(x1.shape[-1])
            self.vec2 = self.alpha_init * jnp.ones(x2.shape[-1])
            
            if R.ndim == 1:
                return jnp.kron(self.vec1, self.vec2 @ R)
            elif R.ndim == 2:
                return jnp.outer(self.vec1, self.vec2 @ R)
            else:
                raise Exception(f"Matrix multiplication between Outer CovType and type {type(R)} of shape {R.shape} not implemented")
        
            

    def __add__(self, other):
        r"""Adds ``other``: constant diagonals give an ``OuterPlusScaledIdentity``, quasiseparable objects give a
        ``GeneralQuasisepPlusNoise``, and anything else gives a ``General`` object.

        """

        if isinstance(other, (Identity, ScaledIdentity)):
            K_sum = OuterPlusScaledIdentity(self.alpha_init, diag = other.diag, wn_diag = other.wn_diag)

        elif isinstance(other, Outer):
            # Could do something better here
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))

        elif isinstance(other, (GeneralQuasisepPlusNoise, GeneralQuasisep, Exp)):
            if other.tinygp_kf is not None:
                new_tinygp_kf = luas.kernels.tinygp_ext.Sum(self.tinygp_kf, other.tinygp_kf, use_block = other.use_block)
            else:
                new_tinygp_kf = self.tinygp_kf

            K_sum = GeneralQuasisepPlusNoise(new_tinygp_kf, diag = other.diag, wn_diag = other.wn_diag,
                                             noise_model = other.noise_model,  use_block = other.use_block)

        elif isinstance(other, (Diagonal, OuterPlusScaledIdentity, General)):
            # Should update as you can always add outer products without losing quasiseparability
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))

        else:
            return NotImplemented
            
        return K_sum

    # Could implement an efficient Quasisep multiplication
    def __mul__(self, other) -> Kernel:
        r"""Multiplies by a scalar or another covariance object. Multiplying by :math:`\vec{\alpha}\vec{\alpha}^T`
        scales row :math:`i` and column :math:`j` by :math:`\alpha_i\alpha_j`, so diagonal objects stay diagonal,
        quasiseparable objects stay quasiseparable and two ``Outer`` objects give an ``Outer`` object.

        """
        if is_scalar(other):
            K_mul = self.scale(other)

        elif is_scalar(self.alpha_init) and isinstance(other, CovType):
            K_mul = other.scale(self.alpha_init**2)
        
        elif isinstance(other, GeneralQuasisep):
            new_kf = ScaledKernel(kernel = other.tinygp_kf, amplitudes = self.alpha_init)
            K_mul = GeneralQuasisepPlusNoise(new_kf, use_block = other.use_block)
        
        elif isinstance(other, GeneralQuasisepPlusNoise):
            new_kf = ScaledKernel(kernel = other.tinygp_kf, amplitudes = self.alpha_init)

            if other.noise_model is not None:
                diag_term = DiagQSM(d=other.noise_model.diag.d * self.alpha_init**2)

                p_new = jnp.einsum('i,i...->i...', self.alpha_init, other.noise_model.lower.p)
                q_new = jnp.einsum('i,i...->i...', self.alpha_init, other.noise_model.lower.q)
                lower_term = StrictLowerTriQSM(p=p_new, q=q_new, a=other.noise_model.lower.a)

                new_noise_model = SymmQSM(diag=diag_term, lower=lower_term)
            else:
                new_noise_model = None

            K_mul = GeneralQuasisepPlusNoise(new_kf, diag = other.diag * self.alpha_init**2, wn_diag = other.wn_diag * self.alpha_init**2,
                                            use_block = other.use_block, noise_model = new_noise_model)
        
        elif isinstance(other, (Identity, ScaledIdentity, Diagonal)):
            K_mul = Diagonal(diag = other.diag * self.alpha_init**2, wn_diag = other.wn_diag * self.alpha_init**2)
        
        elif isinstance(other, Outer):
            K_mul = Outer(alpha = self.alpha_init * other.alpha_init)
        
        elif isinstance(other, General):
            K_mul = General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other.evaluate(x1, x2, **kwargs))
        
        else:
            return NotImplemented
        
        return K_mul


class OuterPlusScaledIdentity(CovType):
    r"""A rank-one matrix plus a constant diagonal, :math:`\vec{\alpha}\vec{\alpha}^T + (d + d_{\mathrm{wn}})\mathbf{I}`.

    Its eigenvectors are given by a Householder reflection which maps :math:`\vec{\alpha}` onto a basis
    vector, so it is decomposed in :math:`\mathcal{O}(N)`.

    Args:
        alpha (JAXArray | Scalar): The vector :math:`\vec{\alpha}`, either a 1D array or a scalar.
        diag (Scalar, optional): The value :math:`d` added to the diagonal. Defaults to 0.
        wn_diag (Scalar, optional): The white noise value :math:`d_{\mathrm{wn}}` added to the diagonal, only
            included when ``wn = True``. Defaults to 0.
        params (optional): Metadata about the parameters the object depends on.

    """
    # TO DO: Surely better to do a Sherman-Morrison CovType instead...
    def __init__(self, alpha, diag = 0., wn_diag = 0., params = None):
        r"""Creates the covariance object, see the class docstring for the arguments."""
        self.alpha_init = alpha
        self.diag = diag
        self.wn_diag = wn_diag
        self.params = params
        self.opt_name = "Rank 1 matrix plus noise"

        if is_scalar(self.alpha_init):
            self.tinygp_kf = HandleIdx(luas.kernels.tinygp_ext.Constant(const = self.alpha_init**2))
        else:
            const_kf = HandleIdx(luas.kernels.tinygp_ext.Constant(1.))
            self.tinygp_kf = ScaledKernel(kernel = const_kf, amplitudes = self.alpha_init)

        if not is_scalar(self.diag) or not is_scalar(self.wn_diag):
            raise Exception("OuterPlusScaledIdentity CovType only supports scalar diag and wn_diag")

        if not is_scalar(self.alpha_init) and self.alpha_init.ndim != 1:
            raise Exception("OuterPlusScaledIdentity CovType only supports scalar or 1D array alpha_init")
    
    def evaluate(self, x1, x2, wn = True, row_idx = None, col_idx = None, full = True, **kwargs):
        r"""Returns the dense matrix, or the sub-matrix given by ``row_idx`` and ``col_idx`` if ``full = False``."""

        if full:
            diag_mat = (self.diag + wn*self.wn_diag) * jnp.eye(x1.shape[0])
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if idx_specified:
                mask = row_idx[:, None] == col_idx[None, :]  # (n1, n2) bool matrix

                # Pick the diagonal values where they match
                diag_val = self.diag + wn*self.wn_diag
                diag_mat = jnp.where(mask, diag_val, 0.0)
                
            else:
                raise Exception("""Cannot evaluate non-stationary diagonal covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                            )
            
        rank1_mat = jnp.outer(self.alpha_init * jnp.ones(x1.shape[-1]), self.alpha_init * jnp.ones(x2.shape[-1]))
        
        return rank1_mat + diag_mat

    def decompose(self, x, wn = True, i = -1, **kwargs):
        r"""Decomposes the matrix with a Householder reflection :math:`\mathbf{H}` which maps :math:`\vec{\alpha}` onto
        the ``i``-th basis vector. The eigenvalues are :math:`|\vec{\alpha}|^2 + d` (once) and :math:`d` (otherwise),
        and the matrix square root is :math:`\mathbf{H}\,\mathrm{diag}(\sqrt{\vec{\lambda}})`.

        Returns:
            (OuterPlusScaledIdentity, dict): Itself and the stored value ``"logdet"``.

        """

        self.alpha = self.alpha_init * jnp.ones(x.shape[-1])
        self.H = HouseholderTransform(self.alpha, i = i)

        # Shift eigenvalues by constant added to diagonal
        self.lam = self.H.lam + (self.diag + wn * self.wn_diag)

        self.factor = HouseholderandDiagonalTransform(self.alpha, jnp.sqrt(self.lam), i = i, transpose = 0)
        
        self.logdet = jnp.log(self.lam).sum()
        return self, {"logdet":self.logdet}

    def matrix_inv_sqrt(self, R, transpose=0, **kwargs):
        r"""Solves for :math:`\mathbf{L}^{-1}\mathbf{R}` (or :math:`\mathbf{L}^{-T}\mathbf{R}`) using the Householder decomposition."""

        if transpose:
            R_prime = self.factor.T.solve(R)
        else:
            R_prime = self.factor.solve(R)
    
        return R_prime

    
    def matrix_sqrt(self, R, transpose=0, **kwargs):
        r"""Multiplies by the matrix square root :math:`\mathbf{L}` (or :math:`\mathbf{L}^T`) from the Householder decomposition."""

        if transpose:
            R_prime = self.factor.T @ R
        else:
            R_prime = self.factor @ R

        return R_prime

    def eigendecomp(self, x, wn = True, i = -1, full = True, **kwargs):
        r"""Calculates the eigendecomposition with a Householder reflection (see :meth:`decompose`).

        Returns:
            (JAXArray, HouseholderTransform): The eigenvalues and the Householder reflection, whose columns are the
            eigenvectors.

        """

        if not full:
            raise NotImplementedError("Eigendecomposition of OuterPlusScaledIdentity CovType only implemented for full matrix")

        self.alpha = self.alpha_init * jnp.ones(x.shape[-1])
        self.H = HouseholderTransform(self.alpha, i = i)
        self.lam = self.H.lam + (self.diag + wn * self.wn_diag)
        self.logdet = jnp.log(self.lam).sum()
            
        return self.lam, self.H
    

    def householder_transform(self, x, u):
        r"""Returns :math:`\mathbf{H}\mathbf{K}\mathbf{H}` as a ``General`` object (see :meth:`General.householder_transform`)."""
        K_general = General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs))
        
        return K_general.householder_transform(x, u)


    def matmul(self, x1, x2, other, wn = True, **kwargs):
        r"""Multiplies ``other`` by the matrix without building it."""
        self.vec1 = self.alpha_init * jnp.ones(x1.shape[-1])
        self.vec2 = self.alpha_init * jnp.ones(x2.shape[-1])

        D = self.diag + wn * self.wn_diag
        
        if other.ndim == 1:
            return D * other + jnp.kron(self.vec1, self.vec2 @ other)
        elif other.ndim == 2:
            return D * other + jnp.outer(self.vec1, self.vec2 @ other)
        else:
            raise NotImplementedError(f"Matrix multiplication between OuterPlusScaledIdentity CovType and type {type(other)} of shape {other.shape} not implemented")
        
    def scale(self, c):
        r"""Returns an ``OuterPlusScaledIdentity`` for :math:`c\mathbf{K}`."""
        return OuterPlusScaledIdentity(alpha = self.alpha_init * jnp.sqrt(c), diag = self.diag * c, wn_diag = self.wn_diag * c)

    def __add__(self, other):
        r"""Adds ``other``: constant diagonals stay ``OuterPlusScaledIdentity``, quasiseparable objects give a
        ``GeneralQuasisepPlusNoise``, and anything else gives a ``General`` object.

        """
        if isinstance(other, (Identity, ScaledIdentity)):
            K_sum = OuterPlusScaledIdentity(self.alpha_init, diag = self.diag + other.diag, wn_diag = self.wn_diag + other.wn_diag)

        elif isinstance(other, (GeneralQuasisepPlusNoise, GeneralQuasisep)):
            if is_scalar(self.alpha_init):
                outer_kf = HandleIdx(luas.kernels.tinygp_ext.Constant(const = self.alpha_init**2))
            else:
                outer_kf = luas.kernels.tinygp_ext.Amplitude(self.alpha_init)

            if other.tinygp_kf is not None:
                new_tinygp_kf = luas.kernels.tinygp_ext.Sum(outer_kf, other.tinygp_kf, use_block = other.use_block)
            else:
                new_tinygp_kf = outer_kf

            K_sum = GeneralQuasisepPlusNoise(new_tinygp_kf, diag = self.diag + other.diag, wn_diag = self.wn_diag + other.wn_diag,
                                             noise_model = other.noise_model,  use_block = other.use_block)
            
        elif isinstance(other, (Diagonal, Outer, OuterPlusScaledIdentity, General)):
            K_sum = General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))

        else:
            return NotImplemented
        
        return K_sum

    def __mul__(self, other) -> Kernel:
        r"""Scales by a scalar. Returns ``NotImplemented`` for anything else."""
        if is_scalar(other):
            return self.scale(other)
        else:
            return NotImplemented

