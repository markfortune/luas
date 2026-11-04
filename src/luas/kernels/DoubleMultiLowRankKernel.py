import numpy as np
import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional
import tinygp

from luas import WhiteNoiseKernel
from luas.kernels.covtype import Outer, Exp, GeneralQuasisep, CovType, Identity, ScaledIdentity, Diagonal
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
from luas.kronecker_fns import kron_prod, tensor_mult
from luas.kernels.tinygp_ext import LowRankProduct
from luas.kernels.BlockKernel import Block2x2Kernel
from luas.kernels.MixingMatQuasisep import MixingMatQuasisep
from luas.kernels.MixingMatGeneral import MixingMatGeneral
from luas.kernels.householder import householder_qr

__all__ = [
    "DoubleMultiLowRankKernel",
]
    
class DoubleMultiLowRankKernel(CovType):
    r"""GP optimisation for a separable noise term plus low-rank kernel terms along
    both dimensions.

    Handles kernel functions of the form

    .. math::

        k(\lambda, \lambda', t, t') = s_\lambda(\lambda, \lambda')\, s_t(t, t')
        + \sum_{i=1}^{N_\alpha} \alpha_i(\lambda)\alpha_i(\lambda')\, k_{t; i}(t, t')
        + \sum_{j=1}^{N_\beta} k_{\lambda; j}(\lambda, \lambda')\, \beta_j(t)\beta_j(t'),

    which on a rectangular grid gives the covariance matrix

    .. math::

        \mathbf{K} = \mathbf{\Sigma}_\lambda \otimes \mathbf{\Sigma}_t
        + \sum_{i=1}^{N_\alpha} \boldsymbol{\alpha}_{\lambda; i}
        \boldsymbol{\alpha}_{\lambda; i}^T \otimes \mathbf{K}_{t; i}
        + \sum_{j=1}^{N_\beta} \mathbf{K}_{\lambda; j} \otimes
        \boldsymbol{\beta}_{t; j} \boldsymbol{\beta}_{t; j}^T.

    For example, common-mode systematics (low-rank in wavelength) combined with a
    detector settling ramp of fixed shape in time but with a wavelength-correlated
    amplitude (low-rank in time). There is no full-rank 2D kernel term; see
    :class:`luas.ExtendedLuasKernel` for that.

    **Method.** Extends :class:`luas.MultiLowRankKernel` by whitening and performing
    a Householder QR decomposition along both dimensions, then decomposing the
    transformed covariance matrix as a 2x2 block matrix: the
    :math:`(N_\alpha N_t, N_\alpha N_t)` A block, the correction
    :math:`\mathbf{C}\mathbf{A}^{-1}\mathbf{B}`, then :math:`\mathbf{D} -
    \mathbf{C}\mathbf{A}^{-1}\mathbf{B}`. Each Householder transformation
    increases the quasiseparable rank of the transformed kernels by 2, but keeps them
    quasiseparable, so the runtime can be linear in both dimensions.

    **Runtime scaling.** With diagonal :math:`\mathbf{\Sigma}_\lambda`,
    :math:`\mathbf{\Sigma}_t` and quasiseparable :math:`k_{t; i}`,
    :math:`k_{\lambda; j}` (mean ranks :math:`\bar{J}_{K_t}`,
    :math:`\bar{J}_{K_\lambda}`):

    .. math::

        \mathcal{O}\left(N_\alpha^3 N_t (\bar{J}_{K_t} + 3N_\beta)^2
        + N_\beta^3 N_\lambda (\bar{J}_{K_\lambda} + 3N_\alpha)^2
        + N_\lambda N_t (N_\alpha + N_\beta)\right).

    The runtime grows quickly with the product :math:`N_\alpha N_\beta`.

    **Selection.** :class:`luas.GP` selects this optimisation automatically when the
    kernel function returns the noise term followed by low-rank terms along both
    dimensions and no full-rank 2D term. ``use_quasi = True`` is only selected if
    :math:`\mathbf{\Sigma}_\lambda` and :math:`\mathbf{\Sigma}_t` are both
    diagonal and all other kernels are quasiseparable or diagonal; otherwise general
    Cholesky decompositions are used (still exact, but no longer linear).

    Warning:
        Needs at least one low-rank term along each dimension, with
        :math:`N_\alpha < N_\lambda` and :math:`N_\beta < N_t`. Every term other than
        the first must be low-rank (an outer product) in exactly one dimension. The
        noise term must be positive definite and be the **first** tuple returned by
        the kernel function. Measurement covariance (non-diagonal
        :math:`\mathbf{\Sigma}_\lambda` or :math:`\mathbf{\Sigma}_t`) prevents
        quasiseparable kernels from being exploited.

        The regression variables in ``X`` should currently be JAX arrays (e.g.
        ``jnp.linspace``); NumPy arrays raise a ``TracerArrayConversionError`` when
        the log-likelihood is JIT compiled.

    Example:
        Common-mode systematics, extra systematics in a single light curve, and a
        wavelength-correlated settling ramp (from the ``luas`` tutorial):

        .. code-block:: python

            def kf(p, X):
                # Diagonal noise only, so quasiseparability is exploited in both dimensions
                s_l = luas.kernels.quasisep.Noise(jnp.exp(p["log_sigma"]))
                s_t = luas.kernels.quasisep.KroneckerDelta()

                # alpha terms: fixed vectors along wavelength, kernels in time
                alpha_l1 = luas.kernels.lowrank.Constant(jnp.exp(p["log_h_const"][0]))
                k_t1 = luas.kernels.quasisep.Matern32(jnp.exp(p["log_l_t1"][0]))

                e_10 = jnp.zeros(X[0].shape[-1]).at[9].set(jnp.exp(p["log_h10"][0]))
                alpha_l2 = luas.kernels.lowrank.Amplitude(e_10)
                k_t2 = luas.kernels.quasisep.SquaredExpApprox(jnp.exp(p["log_l_t2"][0]))

                # beta term: fixed vector along time, kernel in wavelength
                k_l1 = jnp.exp(2*p["log_h_ramp"][0]) * luas.kernels.quasisep.Matern32(jnp.exp(p["log_l_l_ramp"][0]))
                settle_vec = jnp.exp(-(X[1] - X[1][0])/jnp.exp(p["log_l_t_ramp"][0]))
                beta_t1 = luas.kernels.lowrank.Amplitude(settle_vec)

                return (s_l, s_t), (alpha_l1, k_t1), (alpha_l2, k_t2), (k_l1, beta_t1)

            gp = luas.GP(p, (x_l, x_t), kf = kf)  # Uses DoubleMultiLowRankKernel

    Args:
        Sigma (tuple[CovType, CovType]): The covariance objects
            :math:`(\mathbf{\Sigma}_\lambda, \mathbf{\Sigma}_t)` for the noise term.
        *K_list (tuple[CovType, CovType]): Each low-rank term, as a tuple with one
            covariance object per dimension, exactly one of which must be a rank-1
            outer product (e.g. ``luas.kernels.lowrank.Constant`` or
            ``luas.kernels.lowrank.Amplitude``).
        fast_dim (int, optional): Currently unused; both dimensions are always
            transformed.
        never_reduce_dim (bool, optional): Currently unused.
        use_stored_values (bool, optional): Currently unused.
        use_quasi (bool, optional): If ``True``, decompose the dense blocks using
            quasiseparable methods, which requires diagonal
            :math:`\mathbf{\Sigma}_\lambda`, :math:`\mathbf{\Sigma}_t` and
            quasiseparable or diagonal kernels for every term. If ``False``, use
            general Cholesky decompositions. Defaults to ``True``.
        **kwargs: Ignored, accepted so that all Kronecker kernels share a common
            constructor interface.

    See Also:
        :class:`luas.MultiLowRankKernel` for low-rank terms along one dimension only.
        :class:`luas.ExtendedLuasKernel` for also including a full-rank 2D kernel
        term.
    """

    def __init__(
        self,
        Sigma,
        *K_list,
        use_stored_values: Optional[bool] = False,
        use_quasi = True,
        **kwargs,
    ):
        r"""Creates the kernel object, see the class docstring for the arguments."""
        
        self.Sigma = Sigma[0], Sigma[1]
        self.K_list = K_list
        self.N_alpha = len(K_list)
        self.use_quasi = use_quasi
        self.opt_name = "DoubleMultiLowRankKernel"

        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            raise NotImplementedError("Use of pre-calculated stored values not yet implemented.")
        else:
            self.decompose = self.decompose_no_stored_values

    
    def _truncated_basis_outer_tensor(self, N_alpha, N_beta, N_t):
        r"""Builds the basis of :math:`(N_\alpha, N_t)` arrays which are zero except for a single one in one of the first
        :math:`N_\beta` columns.

        Args:
            N_alpha (int): The number of low-rank terms along dimension 0.
            N_beta (int): The number of low-rank terms along dimension 1.
            N_t (int): The length of dimension 1.

        Returns:
            JAXArray: The basis, of shape ``(N_alpha * N_beta, N_alpha, N_t)``.

        """

        e_alpha = jnp.eye(N_alpha)              # (N_alpha, N_alpha)
        e_t = jax.nn.one_hot(jnp.arange(N_beta), N_t)       # (N_beta, N_t)
    
        T = jnp.einsum("ia,jt->ijat", e_alpha, e_t)
        return T.reshape(N_alpha * N_beta, N_alpha, N_t)
        

    def decompose_no_stored_values(
        self,
        X: Tuple[JAXArray],
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:
        r"""Decomposes the covariance matrix.

        Both dimensions are whitened by the noise term, and the low-rank vectors along each dimension are QR
        decomposed with Householder transformations. The transformed matrix is decomposed as a 2x2 block matrix:
        the :math:`(N_\alpha N_t, N_\alpha N_t)` A block mixing the kernels of the low-rank terms along dimension 0,
        the off-diagonal B block, which is only non-zero in the first :math:`N_\beta` columns, and the Schur
        complement :math:`\mathbf{D} - \mathbf{C}\mathbf{A}^{-1}\mathbf{B}`. The quasiseparable structure is kept if
        ``use_quasi = True``.

        Args:
            X (tuple): Input locations along each dimension.
            stored_values (PyTree, optional): Dictionary the stored values are added to.

        Returns:
            (DoubleMultiLowRankKernel, PyTree): Itself and the stored values, including ``"logdet"``.

        Raises:
            Exception: If any term isn't low rank in exactly one dimension, or if there are at least as many
                low-rank terms as the length of their dimension.

        """

        if self.use_quasi:
            MixingMatTerm = MixingMatQuasisep
        else:
            MixingMatTerm = MixingMatGeneral

        self.N_l = X[0].shape[-1]
        self.N_t = X[1].shape[-1]

        # Generate transformations
        Sigma_transf0, stored_values_Sigma0 = self.Sigma[0].decompose(X[0])
        Sigma_transf1, stored_values_Sigma1 = self.Sigma[1].decompose(X[1])

        self.Sigma_transf = (Sigma_transf0, Sigma_transf1)

        # Computes the log determinant of Sigma_l \otimes Sigma_t
        stored_values["logdet"] = X[0].shape[-1]*stored_values_Sigma1["logdet"]
        stored_values["logdet"] += X[1].shape[-1]*stored_values_Sigma0["logdet"]

        self.N_alpha = 0
        self.N_beta = 0

        self.K_transf_list = ()
        for K_i in self.K_list:
            K0 = self.Sigma_transf[0].inv_sqrt_transform(K_i[0], X[0])
            K1 = self.Sigma_transf[1].inv_sqrt_transform(K_i[1], X[1])
            self.K_transf_list += ((K0, K1),)

            if isinstance(K0, Outer) and not isinstance(K1, Outer):
                self.N_alpha += 1
            elif not isinstance(K0, Outer) and isinstance(K1, Outer):
                self.N_beta += 1
            else:
                raise Exception("Require only terms that are low-rank in exactly one of the two dimensions, \
                                in addition to diagonal noise for the first Kronecker term.")
                
        stored_values["A"] = jnp.zeros((X[0].shape[-1], self.N_alpha))
        stored_values["B"] = jnp.zeros((X[1].shape[-1], self.N_beta))
        A_kernel_order = []
        B_kernel_order = []
        
        col_i = 0
        col_j = 0
        for (i, K_i) in enumerate(self.K_transf_list):
            if isinstance(K_i[0], Outer):
                K0_outer, _ = K_i[0].decompose(X[0])
                stored_values["A"] = stored_values["A"].at[:, col_i].set(K0_outer.alpha)
                A_kernel_order.append(K_i[1])
                col_i += 1
            elif isinstance(K_i[1], Outer):
                K1_outer, _ = K_i[1].decompose(X[1])
                stored_values["B"] = stored_values["B"].at[:, col_j].set(K1_outer.alpha)
                B_kernel_order.append(K_i[0])
                col_j += 1

        # stored_values["A"] = self.Sigma_transf[0].matrix_inv_sqrt(stored_values["A"])
        # stored_values["B"] = self.Sigma_transf[1].matrix_inv_sqrt(stored_values["B"])
        

        reduce_dim = self.N_alpha < X[0].shape[-1] and self.N_beta < X[1].shape[-1]
        if not reduce_dim:
            raise Exception(r"Requires fewer low-rank terms than the dimension size, i.e. N_\alpha < N_\lambda and N_\beta < N_t")

        stored_values["J_A"], stored_values["U_A"], self.householder_transform_A = householder_qr(stored_values["A"])
        stored_values["J_B"], stored_values["U_B"], self.householder_transform_B = householder_qr(stored_values["B"])

        kf_A = MixingMatTerm(stored_values["J_A"], A_kernel_order, diag = 1., fast_dim = 1)
        kf_B = MixingMatTerm(stored_values["J_B"], B_kernel_order, diag = 1., fast_dim = 0)

        for i in range(self.N_alpha):
            kf_B = kf_B.householder_transform(X, stored_values["U_A"][:, i])
            
        for j in range(self.N_beta):
            kf_A = kf_A.householder_transform(X, stored_values["U_B"][:, j])

        # Calculate component of K_l \otimes \beta \beta^T which falls within block A (concentrated in top left corner of matrix)
        top_corner_ind = (jnp.arange(self.N_alpha), jnp.arange(self.N_beta))
        X_top_corner = (X[0][:self.N_alpha], X[1][:self.N_beta])

        top_corner_eval = kf_B.evaluate(X_top_corner, X_top_corner,
                                        row_idx = top_corner_ind, col_idx = top_corner_ind,
                                        flip_kron = True, calc_diag = False)

        # if quasisep
        banded_term = as_banded(top_corner_eval, zero_pad_to_len = self.N_alpha * X[1].shape[-1])
        kf_A = MixingMatTerm(kf_A.mixing_mat, kf_A.kernel_list,
                                 fast_dim = 1, noise_model = banded_term, diag = 1.)

        non_zero_B_ind = (jnp.arange(self.N_alpha, X[0].shape[-1]), jnp.arange(self.N_beta))
        X_at_non_zero_B = (jnp.asarray(X[0])[non_zero_B_ind[0]], jnp.asarray(X[1])[non_zero_B_ind[1]])
        self.B_rows = kf_B.evaluate(X_top_corner, X_at_non_zero_B, full = False,
                                    row_idx = top_corner_ind, col_idx = non_zero_B_ind,
                                    flip_kron = False, calc_diag = False)

        self.kf_B = kf_B

        x_A = X[0][..., :self.N_alpha]
        X_A = (x_A, X[1])

        # Calc C A^-1 B for D block
        kf_A, stored_vals_A = kf_A.decompose(X_A, full = True)

        A_inv_loc = self._truncated_basis_outer_tensor(self.N_alpha, self.N_beta, self.N_t)

        L_A_inv_loc = jax.vmap(kf_A.matrix_inv_sqrt)(A_inv_loc)
        K_A_inv_alphabeta = jnp.einsum('iat,jat->ij', L_A_inv_loc, L_A_inv_loc)

        K_A_inv_B = K_A_inv_alphabeta @ self.B_rows

        C_A_inv_B = LowRankProduct(self.B_rows.T, K_A_inv_B.T).to_qsm()

        if not self.use_quasi:
            shape_F_style = (self.N_beta, self.N_l - self.N_alpha, self.N_beta, self.N_l - self.N_alpha)
            shape_C_style = (self.N_beta * (self.N_l - self.N_alpha), self.N_beta * (self.N_l - self.N_alpha))
            C_A_inv_B = C_A_inv_B.to_dense().reshape(shape_F_style, order = "F").reshape(shape_C_style, order = "C")

        
        kf_D_corr = MixingMatTerm(kf_B.mixing_mat, kf_B.kernel_list, 
                                  fast_dim = 0, diag = 1., noise_model = -C_A_inv_B)

        # self.kf_D = MixingMatTerm(kf_B.mixing_mat, kf_B.kernel_list,
        #                               fast_dim = 0, diag = 1.)

        kf_D_wn = WhiteNoiseKernel(diag = 1.)
        K_D_CAB = Block2x2Kernel(kf_A = kf_D_corr, kf_D_CAB = kf_D_wn, A_full = False,
                                 dim_split = 1, split_loc = self.N_beta, split_idx = True)

        self.kf_tilde = Block2x2Kernel(kf_A = kf_A, kf_B = self.K_B_matmul, kf_D_CAB = K_D_CAB,
                                       dim_split = 0, split_loc = self.N_alpha, split_idx = True)
        
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose(X)

        stored_values["logdet"] += stored_values["kf_tilde_stored"]["logdet"]
        self.logdet = stored_values["logdet"]

        return self, stored_values

    
    def transform_fn(self, R, transpose = 0):
        r"""Applies the transformation :math:`\mathbf{T}` (or :math:`\mathbf{T}^T` if ``transpose = 1``): along each
        dimension, the matrix square root of the noise covariance combined with the Householder transformation of
        that dimension's low-rank vectors. Requires :meth:`decompose` to have been called.

        """
        
        if transpose:
            R_prime = self.Sigma_transf[0].matrix_sqrt(R, transpose = 1)
            R_prime = self.householder_transform_A(R_prime, transpose = 0)
        else:
            R_prime = self.householder_transform_A(R, transpose = 1)
            R_prime = self.Sigma_transf[0].matrix_sqrt(R_prime, transpose = 0)

        R_prime = R_prime.T

        if transpose:
            R_prime = self.Sigma_transf[1].matrix_sqrt(R_prime, transpose = 1)
            R_prime = self.householder_transform_B(R_prime, transpose = 0)
        else:
            R_prime = self.householder_transform_B(R_prime, transpose = 1)
            R_prime = self.Sigma_transf[1].matrix_sqrt(R_prime, transpose = 0)
            
        return R_prime.T


    def inv_transform_fn(self, R, transpose = 0):
        r"""Applies the inverse transformation :math:`\mathbf{T}^{-1}` (or :math:`\mathbf{T}^{-T}` if ``transpose = 1``),
        see :meth:`transform_fn`. Requires :meth:`decompose` to have been called.

        """

        if transpose:
            R_prime = self.householder_transform_A(R, transpose = 1)
            R_prime = self.Sigma_transf[0].matrix_inv_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.Sigma_transf[0].matrix_inv_sqrt(R, transpose = 0)
            R_prime = self.householder_transform_A(R_prime, transpose = 0)

        R_prime = R_prime.T

        if transpose:
            R_prime = self.householder_transform_B(R_prime, transpose = 1)
            R_prime = self.Sigma_transf[1].matrix_inv_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.Sigma_transf[1].matrix_inv_sqrt(R_prime, transpose = 0)
            R_prime = self.householder_transform_B(R_prime, transpose = 0)

        return R_prime.T

    
    def K_B_matmul(self, X1, X2, R_B, transpose = 0, **kwargs):
        r"""Multiplies by the off-diagonal B block of the transformed covariance matrix (or its transpose if
        ``transpose = 1``), using that it is only non-zero in the first :math:`N_\beta` columns. Requires
        :meth:`decompose` to have been called.

        Args:
            X1, X2: Not used.
            R_B (JAXArray): The array to multiply.
            transpose (int, optional): Whether to multiply by the transpose. Defaults to 0.

        Returns:
            JAXArray: The product.

        """

        if transpose:
            B_mat = self.B_rows.T
            ravel_style = "C"
        else:
            B_mat = self.B_rows.copy()
            ravel_style = "C"
            
        R_sparse = R_B[:, :self.N_beta]

        r_sparse = R_sparse.ravel(ravel_style)
        sparse_B_r = B_mat @ r_sparse

        if transpose:
            sparse_B_R = sparse_B_r.reshape((-1, self.N_beta), order = ravel_style)
            B_R = jnp.concatenate([sparse_B_R,
                                   jnp.zeros((sparse_B_R.shape[0], self.N_t - sparse_B_R.shape[1])),
                                  ], axis = 1)
        else:
            sparse_B_R = sparse_B_r.reshape((self.N_alpha, self.N_beta), order = ravel_style)
            
            B_R = jnp.concatenate([sparse_B_R,
                                   jnp.zeros((sparse_B_R.shape[0], self.N_t - sparse_B_R.shape[1])),
                                  ], axis = 1)
            
        return B_R

        
    def matrix_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
    ) -> JAXArray:
        r"""Multiplies ``R`` by the matrix square root :math:`\mathbf{L} = \mathbf{T}\tilde{\mathbf{L}}` (or :math:`\mathbf{L}^T` if
        ``transpose = 1``), where :math:`\mathbf{T}` is applied by :meth:`transform_fn` and :math:`\tilde{\mathbf{L}}` is the matrix
        square root of the transformed covariance matrix. Requires :meth:`decompose` to have been called.

        """

        if transpose:
            R_prime = self.transform_fn(R, transpose = 1)
            R_prime = self.kf_tilde.matrix_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.kf_tilde.matrix_sqrt(R, transpose = 0)
            R_prime = self.transform_fn(R_prime, transpose = 0)
        
        return R_prime

    def matrix_inv_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
    ) -> JAXArray:
        r"""Multiplies ``R`` by :math:`\mathbf{L}^{-1} = \tilde{\mathbf{L}}^{-1}\mathbf{T}^{-1}` (or :math:`\mathbf{L}^{-T}` if
        ``transpose = 1``), see :meth:`matrix_sqrt`. Requires :meth:`decompose` to have been called.

        """

        if transpose:
            R_prime = self.kf_tilde.matrix_inv_sqrt(R, transpose = 1)
            R_prime = self.inv_transform_fn(R_prime, transpose = 1)
        else:
            R_prime = self.inv_transform_fn(R, transpose = 0)
            R_prime = self.kf_tilde.matrix_inv_sqrt(R_prime, transpose = 0)
        
        return R_prime
       
    def matmul(self, X1, X2, R, **kwargs):
        r"""Multiplies the data-shaped array ``R`` by the covariance matrix without building it, by applying the noise
        term and every other Kronecker term along each dimension (see :func:`luas.kronecker_fns.tensor_mult`).

        Args:
            X1 (tuple): Input locations along each dimension for the rows.
            X2 (tuple): Input locations along each dimension for the columns, matching the shape of ``R``.
            R (JAXArray): The data-shaped array to multiply.
            **kwargs: Passed to each covariance object's ``matmul`` method, e.g. ``wn = False``.

        Returns:
            JAXArray: The product.

        """

        K_R = tensor_mult(self.Sigma, X1, X2, R, **kwargs)

        for K in self.K_list:
            K_R += tensor_mult(K, X1, X2, R, **kwargs)
        
        return K_R


def as_banded(A, zero_pad_to_len = None):
    r"""Converts a small dense symmetric matrix into a ``tinygp`` banded quasiseparable matrix, zero-padded to a
    given length.

    Args:
        A (JAXArray): The symmetric matrix, of shape ``(n, n)``.
        zero_pad_to_len (int, optional): The size of the output matrix. Defaults to ``n``.

    Returns:
        SymmQSM: The banded matrix, with ``A`` in its top-left corner and zeros elsewhere.

    """
    # Assumes matrix is symmetric

    N = A.shape[1]

    if zero_pad_to_len is not None:
        zero_padding = zero_pad_to_len - N
    else:
        zero_padding = 0
    
    diag = jnp.concatenate([jnp.diag(A), jnp.zeros(zero_padding)])
    off_diags = jnp.zeros((N + zero_padding, N-1))
    
    for i in range(1, A.shape[1]):
        off_diags = off_diags.at[:N-i,i-1].set(jnp.diag(A, k = i))

    return tinygp.noise.Banded(diag, off_diags).to_qsm()

