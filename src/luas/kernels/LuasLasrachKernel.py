import numpy as np
import jax.numpy as jnp
from typing import Callable, Tuple, Union, Any, Optional

from luas.kernels.covtype import CovType, Outer, Identity, Diagonal
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import calc_total_size, cyclic_transpose, vmap_for_tensors, tensor_mult
from luas.kernels.SeparableKernel import SeparableKernel
from luas.kernels.BlockKernel import Block2x2Kernel, BlockKernel

__all__ = [
    "LuasLasrachKernel",
]

class LuasLasrachKernel(CovType):
    r"""GP optimisation for a covariance matrix that is the sum of two Kronecker
    products, which decouples a 2D GP into independent 1D GPs along one "fast"
    dimension.

    Handles the same kernel structure as :class:`luas.LuasKernel`,

    .. math::

        k(\lambda, \lambda', t, t') = s_\lambda(\lambda, \lambda')\, s_t(t, t')
        + k_\lambda(\lambda, \lambda')\, k_t(t, t'),
        \qquad
        \mathbf{K} = \mathbf{\Sigma}_\lambda \otimes \mathbf{\Sigma}_t
        + \mathbf{K}_\lambda \otimes \mathbf{K}_t,

    but only performs simultaneous diagonalisation in the dimension(s) other than
    ``fast_dim``. Writing the fast dimension as time :math:`t`,

    .. math::

        \mathbf{K} = [\mathbf{\Sigma}_\lambda^{\frac{1}{2}} \mathbf{Q}_{K'_\lambda}
        \otimes \mathbf{I}_{N_t}]\,
        [\mathbf{I}_{N_\lambda} \otimes \mathbf{\Sigma}_t
        + \mathbf{\Lambda}_{K'_\lambda} \otimes \mathbf{K}_t]\,
        [\mathbf{\Sigma}_\lambda^{\frac{1}{2}} \mathbf{Q}_{K'_\lambda}
        \otimes \mathbf{I}_{N_t}]^T,

    where :math:`\mathbf{K}'_\lambda = \mathbf{\Sigma}_\lambda^{-\frac{1}{2}}
    \mathbf{K}_\lambda \mathbf{\Sigma}_\lambda^{-\frac{T}{2}}`. The middle matrix is
    block-diagonal, so the log-likelihood becomes a sum of independent 1D GP
    log-likelihoods, one per eigenvalue :math:`\lambda_{K'_\lambda; i}`, each with
    covariance :math:`\mathbf{\Sigma}_t + \lambda_{K'_\lambda; i} \mathbf{K}_t`. These
    blocks are never eigendecomposed, so any fast 1D method valid for that sum can
    be used, e.g. quasiseparable (``tinygp``/celerite) Cholesky decompositions in
    linear time. The blocks are decomposed in parallel with ``jax.vmap`` (or
    ``jax.pmap``).

    If :math:`\mathbf{K}_\lambda` is low-rank with rank :math:`R_\lambda` (e.g. a
    ``luas.kernels.lowrank`` kernel), only :math:`R_\lambda` blocks contain
    :math:`\mathbf{K}_t` and the remaining blocks reduce to :math:`\mathbf{\Sigma}_t`,
    which is exploited automatically.

    This optimisation also works for more than two dimensions: every dimension
    other than ``fast_dim`` is simultaneously diagonalised.

    **Runtime scaling (2D, fast dimension** :math:`t` **).** With quasiseparable
    :math:`\mathbf{\Sigma}_t` and :math:`\mathbf{K}_t` of rank :math:`J_{K_t}`:

    - Arbitrary :math:`\mathbf{\Sigma}_\lambda`, :math:`\mathbf{K}_\lambda`:
      :math:`\mathcal{O}(N_\lambda^3 + N_\lambda^2 N_t + R_\lambda N_t J_{K_t}^2)`.
    - Quasiseparable :math:`\mathbf{\Sigma}_\lambda` (rank :math:`J_{\Sigma_\lambda}`):
      the decomposition and transformation by :math:`\mathbf{\Sigma}_\lambda`
      drop to :math:`\mathcal{O}(N_\lambda J_{\Sigma_\lambda}^2 + N_\lambda N_t
      J_{\Sigma_\lambda})`, but the eigendecomposition of
      :math:`\mathbf{K}'_\lambda` (:math:`\mathcal{O}(N_\lambda^3)`) and applying
      its dense eigenvector matrix to the residuals
      (:math:`\mathcal{O}(N_\lambda^2 N_t)`) remain. Both are avoided only for a
      rank-1 ``lowrank.Constant``/``lowrank.Amplitude`` term, which uses a
      Householder transformation instead, giving
      :math:`\mathcal{O}(N_\lambda J_{\Sigma_\lambda}^2 + N_\lambda N_t
      J_{\Sigma_\lambda} + N_t J_{K_t}^2)`.

    If the fast-dimension kernels are not quasiseparable, each block costs a general
    :math:`\mathcal{O}(N_t^3)` Cholesky decomposition, which can still beat the
    :math:`\mathcal{O}(N_t^3)` eigendecomposition in :class:`luas.LuasKernel` when
    :math:`R_\lambda` is small or many parallel workers (e.g. a GPU) are available.

    **Selection.** :class:`luas.GP` selects this optimisation automatically for two
    Kronecker terms with no low-rank terms when all kernels along the longer
    dimension are quasiseparable or diagonal (setting ``fast_dim`` to that
    dimension). It is also selected when *neither* dimension is
    quasiseparable/diagonal but the shorter dimension has fewer than
    ``max_gen_cholesky_blocks`` points (default 10). A single low-rank term
    (e.g. ``luas.kernels.lowrank.Constant``) is instead routed to
    :class:`luas.MultiLowRankKernel`, which reduces to the same optimisation.

    Warning:
        :math:`\mathbf{\Sigma}_\lambda` and :math:`\mathbf{\Sigma}_t` must be positive
        definite and must be the **first** tuple returned by the kernel function.
        Along ``fast_dim``, :math:`\mathbf{\Sigma}_t` and :math:`\mathbf{K}_t` are
        scaled and added together, so they must be compatible covariance types
        (e.g. both from ``luas.kernels.quasisep``/``luas.kernels.diagonal`` to get a
        linear-time decomposition, or both general).

    Example:
        Measurement covariance plus a 2D kernel term, quasiseparable in time (from
        the ``luas`` tutorial):

        .. code-block:: python

            def kf(p, X):
                s_l = luas.kernels.quasisep.Noise(jnp.exp(p["log_sigma"]))
                s_l += jnp.exp(2*p["log_h_readout"][0]) * luas.kernels.quasisep.Exp(jnp.exp(p["log_l_readout"][0]))
                s_t = luas.kernels.quasisep.KroneckerDelta()

                # Eigendecomposed, so quasisep gives no speed-up here
                k_l = jnp.exp(2*p["log_h"][0]) * luas.kernels.Matern52(jnp.exp(p["log_l_l"][0]))

                # Keep quasiseparable for linear scaling in time
                k_t = luas.kernels.quasisep.Matern32(jnp.exp(p["log_l_t"][0]))

                return (s_l, s_t), (k_l, k_t)

            # Uses LuasLasrachKernel with fast_dim = 1 (assuming N_t >= N_l)
            gp = luas.GP(p, (x_l, x_t), kf = kf)

            # Or explicitly:
            gp = luas.GP(p, (x_l, x_t), kf = kf, use_kernel = luas.LuasLasrachKernel, fast_dim = 1)

    Args:
        Sigma (tuple[CovType, ...]): One positive definite covariance object per
            dimension for the first Kronecker product term.
        K (tuple[CovType, ...]): One covariance object per dimension for the second
            Kronecker product term. May be positive semi-definite or low-rank.
        fast_dim (int): **Required.** The dimension which is decomposed as
            independent 1D GPs and is never eigendecomposed. Choose the dimension
            where :math:`\mathbf{\Sigma}` and :math:`\mathbf{K}` have fast
            decompositions (e.g. quasiseparable kernels), typically the longer one.
            Must be passed as a keyword argument to :class:`luas.GP` when using
            ``use_kernel = luas.LuasLasrachKernel``.
        use_stored_values (bool, optional): Reusing stored decompositions is not yet
            implemented; must be ``False`` (the default) or an exception is raised.
        use_pmap (bool, optional): Experimental, and not currently working. Intended
            to decompose the independent blocks across multiple devices with
            ``jax.pmap`` rather than ``jax.vmap`` (requiring at least as many devices
            as blocks). Defaults to ``False``.
        **kwargs: Ignored, accepted so that all Kronecker kernels share a common
            constructor interface.

    See Also:
        :class:`luas.LuasKernel` for the same kernel structure using
        eigendecomposition in both dimensions.
        :class:`luas.MultiLowRankKernel`, which reduces to this optimisation for a
        single low-rank term and generalises it to several.
        :class:`luas.LuasPlusMultiLowRankKernel` for adding low-rank terms.
    """

    def __init__(
        self,
        Sigma,
        K,
        *, # force keyword only after this
        fast_dim: int | None = None,
        use_stored_values: Optional[bool] = False,
        use_pmap = False,
        **kwargs,
    ):
        r"""Creates the kernel object, see the class docstring for the arguments."""
        if fast_dim is None:
            # Must specify the fast_dim
            raise Exception("Must specify fast_dim, which is the dimension which will not be eigendecomposed/simultaneously diagonlised.\
                            Typically this dimension should be fast to decompose e.g. using luas.kernels.quasisep, or else the other dimension should be small.")

        self.Sigma = Sigma
        self.K = K
        self.fast_dim = fast_dim
        self.K_list = (K,)
        self.use_pmap = use_pmap
        self.opt_name = "LuasLasrachKernel"
           
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            raise NotImplementedError("Use of pre-calculated stored values not yet implemented.")
        else:
            self.decompose = self.decompose_no_stored_values
    
    def evaluate(self, X1, X2, full = True, row_idx = None, col_idx = None, **kwargs):
        r"""Builds the full dense covariance matrix, :math:`\bigotimes_d \mathbf{\Sigma}_d + \bigotimes_d \mathbf{K}_d`.
        Only practical for small data sets.

        Args:
            X1 (tuple): Input locations along each dimension for the rows.
            X2 (tuple): Input locations along each dimension for the columns.
            full (bool, optional): Whether to evaluate the full matrices along each dimension, otherwise the
                sub-matrices given by ``row_idx`` and ``col_idx``. Defaults to ``True``.
            row_idx (tuple, optional): Row indices along each dimension if ``full = False``.
            col_idx (tuple, optional): Column indices along each dimension if ``full = False``.
            **kwargs: Passed to each covariance object's ``evaluate`` method, e.g. ``wn``.

        Returns:
            JAXArray: The covariance matrix, ordered as the row-major (``"C"``) flattening of the data array.

        """

        dim = len(X1)

        if row_idx is None and col_idx is None:
            row_idx = (None,) * dim
            col_idx = (None,) * dim
        else:
            assert full or (row_idx is not None and col_idx is not None)

        Sigma = self.Sigma[0].evaluate(X1[0], X2[0], row_idx = row_idx[0], col_idx = col_idx[0], full = full, **kwargs)
        
        for d in range(1, dim):
            Sigma = jnp.kron(Sigma, self.Sigma[d].evaluate(X1[d], X2[d], row_idx = row_idx[d], col_idx = col_idx[d], full = full, **kwargs))
        
        K = self.K[0].evaluate(X1[0], X2[0], row_idx = row_idx[0], col_idx = col_idx[0], full = full, **kwargs)
        for d in range(1, dim):
            K = jnp.kron(K, self.K[d].evaluate(X1[d], X2[d], row_idx = row_idx[d], col_idx = col_idx[d], full = full, **kwargs))
        
        Sigma += K    
        return Sigma
    

    def decompose_no_stored_values(
        self,
        X: Tuple,
        stored_values: Optional[PyTree] = None,
    ) -> PyTree:
        r"""Decomposes the covariance matrix.

        Along every dimension except ``fast_dim``, :math:`\mathbf{\Sigma}_d` is decomposed, :math:`\mathbf{K}_d` is
        whitened by it and the result eigendecomposed, as for :class:`luas.LuasKernel`. In this transformed basis
        the covariance matrix becomes block diagonal, with one block per combination of eigenvalues along the
        other dimensions: :math:`\mathbf{\Sigma}_{\mathrm{fast}} + \lambda\mathbf{K}_{\mathrm{fast}}`. These
        independent blocks along ``fast_dim`` are decomposed with ``BlockKernel`` using their own (e.g.
        quasiseparable) decompositions. If :math:`\tilde{\mathbf{K}}_d` is low rank, only the blocks with non-zero
        eigenvalues need this, and the rest only need :math:`\mathbf{\Sigma}_{\mathrm{fast}}` (see
        ``Block2x2Kernel``).

        Args:
            X (tuple): Input locations along each dimension.
            stored_values (PyTree, optional): Dictionary the stored values are added to.

        Returns:
            (LuasLasrachKernel, PyTree): Itself and the stored values, including ``"logdet"`` and the eigenvalues
            ``"lam_d"``, eigenvectors ``"Q_d"`` and rank ``"rank_d"`` along each dimension except ``fast_dim``.

        """

        total_size = calc_total_size(X)
        self.dim = len(X)
        stored_values = {} if stored_values is None else stored_values

        # Initialise before loop
        self.sigma_decomp_mats = ()
        self.eigen_decomp_mats = ()
        non_fast_dim_size = 1
        all_lam = jnp.ones(1)
        all_lam_shape = ()
        stored_values["logdet"] = 0.

        for d in range(self.dim):
            if d != self.fast_dim:
                Sigma_d_new, stored_values_d = self.Sigma[d].decompose(X[d])
                stored_values["logdet"] += (total_size/X[d].shape[-1])*stored_values_d["logdet"]

                if isinstance(self.K[d], Outer):
                    K_d, _ = self.K[d].decompose(X[d])
                else:
                    K_d = self.K[d]
                
                K_d_new = Sigma_d_new.inv_sqrt_transform(K_d, X[d])
                
                if self.dim > 2:
                    Sigma_d_new.matrix_sqrt = vmap_for_tensors(Sigma_d_new.matrix_sqrt)
                    Sigma_d_new.matrix_inv_sqrt = vmap_for_tensors(Sigma_d_new.matrix_inv_sqrt)
                    K_d_new.matrix_sqrt = vmap_for_tensors(K_d_new.matrix_sqrt)
                    K_d_new.matrix_inv_sqrt = vmap_for_tensors(K_d_new.matrix_inv_sqrt)
                
                stored_values[f"lam_{d}"], stored_values[f"Q_{d}"] = K_d_new.eigendecomp(X[d])

                # Reduce to non-zero eigenvalues for low rank kernels
                # Assumes eigenvalues are sorted in order from smallest to largest, default for jax.linalg.eigh
                stored_values[f"rank_{d}"] = K_d_new.rank(X[d])
                stored_values[f"lam_{d}"] = stored_values[f"lam_{d}"][-stored_values[f"rank_{d}"]:]
                
                all_lam = jnp.kron(all_lam.reshape(all_lam_shape + (1,)), stored_values[f"lam_{d}"])
                all_lam_shape = all_lam.shape
                
                self.sigma_decomp_mats += (Sigma_d_new,)
                self.eigen_decomp_mats += (stored_values[f"Q_{d}"],)

                # Keep track of size of combined non-blackbox length to check if low-rank kernels can be exploited
                non_fast_dim_size *= X[d].shape[-1]
            else:
                self.sigma_decomp_mats += (Identity(),)
                self.eigen_decomp_mats += (None,)

        all_lam = cyclic_transpose(all_lam, -(self.dim-1-self.fast_dim)).ravel()

        kf_dense_blocks = BlockKernel((Identity(), self.Sigma[self.fast_dim]),
                                      (Diagonal(diag = all_lam), self.K[self.fast_dim]),
                                      non_block_dim_size = all_lam.size, block_dim=1, use_pmap = self.use_pmap)
        X_block = (jnp.zeros(non_fast_dim_size), X[self.fast_dim]) # all_lam is not used but is an array of the expected length

        # Optimised for low-rank kernels where a 2x2 block matrix is formed
        # A block is dense covariance matrix blocks
        # D block is I \otimes Sigma_bb
        # B and C blocks are zero and ignored
        if all_lam.size < non_fast_dim_size:
            kf_A = SeparableKernel(Identity(), self.Sigma[self.fast_dim],
                                        inv_dims=self.Sigma[self.fast_dim].use_inv)
            self.kf_tilde = Block2x2Kernel(kf_A = kf_A, kf_D_CAB = kf_dense_blocks, D_full = False,
                                           dim_split = 0, split_loc = -all_lam.size)
        else:
            self.kf_tilde = kf_dense_blocks

        self.kf_tilde, stored_values_kf_tilde = self.kf_tilde.decompose(X_block)
        stored_values["logdet"] += stored_values_kf_tilde["logdet"]

        self.logdet = stored_values["logdet"]

        return self, stored_values


    def transform_fn(self, R, transpose = 0):
        r"""Applies the transformation :math:`\mathbf{T}` along every dimension except ``fast_dim`` (or
        :math:`\mathbf{T}^T` if ``transpose = 1``), where along each such dimension it is :math:`\mathbf{L}_d\mathbf{Q}_d`:
        the matrix square root of :math:`\mathbf{\Sigma}_d` times the eigenvectors of the whitened
        :math:`\tilde{\mathbf{K}}_d`. The matrix square root of the covariance matrix is
        :math:`\mathbf{T}\tilde{\mathbf{L}}`, where :math:`\tilde{\mathbf{L}}` is the matrix square root of the
        independent blocks along ``fast_dim``. Requires :meth:`decompose` to have been called.

        """
        
        R_prime = cyclic_transpose(R, 2)

        if transpose:
            for d in range(self.dim):
                if d != self.fast_dim:
                    R_prime = self.sigma_decomp_mats[d].matrix_sqrt(R_prime, transpose = 1)
                    R_prime = self.eigen_decomp_mats[d].T @ R_prime
                R_prime = cyclic_transpose(R_prime, 1)
        else:
            for d in range(self.dim):
                if d != self.fast_dim:
                    R_prime = self.eigen_decomp_mats[d] @ R_prime
                    R_prime = self.sigma_decomp_mats[d].matrix_sqrt(R_prime, transpose = 0)
                R_prime = cyclic_transpose(R_prime, 1)

        R_prime = cyclic_transpose(R_prime, -2)
            
        return R_prime


    def inv_transform_fn(self, R, transpose = 0):
        r"""Applies the inverse transformation :math:`\mathbf{T}^{-1}` (or :math:`\mathbf{T}^{-T}` if ``transpose = 1``),
        see :meth:`transform_fn`. Requires :meth:`decompose` to have been called.

        """
        R_prime = cyclic_transpose(R, 2)

        if transpose:
            for d in range(self.dim):
                if d != self.fast_dim:
                    R_prime = self.eigen_decomp_mats[d] @ R_prime
                    R_prime = self.sigma_decomp_mats[d].matrix_inv_sqrt(R_prime, transpose = 1)
                R_prime = cyclic_transpose(R_prime, 1)
        else:
            for d in range(self.dim):
                if d != self.fast_dim:
                    R_prime = self.sigma_decomp_mats[d].matrix_inv_sqrt(R_prime, transpose = 0)
                    R_prime = self.eigen_decomp_mats[d].T @ R_prime
                R_prime = cyclic_transpose(R_prime, 1)

        R_prime = cyclic_transpose(R_prime, -2)
        return R_prime
    

    def _calc_in_flat_fast_dim(self, fn, R, **kwargs):
        r"""Applies ``fn`` to ``R`` after rotating ``fast_dim`` to the last axis and flattening every other axis into
        one, so ``fn`` acts on a 2D array of shape ``(N_other, N_fast)``. The result is reshaped and rotated back.

        Args:
            fn (Callable): The function to apply, e.g. a matrix square root of the blocks along ``fast_dim``.
            R (JAXArray): The data-shaped array.
            **kwargs: Passed to ``fn``.

        Returns:
            JAXArray: The result, with the same shape as ``R``.

        """
        
        R_rot = cyclic_transpose(R, -(self.dim-self.fast_dim-1))
        R_rot_shape = R_rot.shape
        R_flat = R_rot.reshape(-1, R_rot_shape[-1])

        R_prime_flat = fn(R_flat, **kwargs)
        R_rot_prime = R_prime_flat.reshape(R_rot_shape)
        R_prime = cyclic_transpose(R_rot_prime, (self.dim-self.fast_dim-1))

        return R_prime


    def matrix_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
    ) -> JAXArray:
        r"""Multiplies ``R`` by the matrix square root :math:`\mathbf{L} = \mathbf{T}\tilde{\mathbf{L}}` (or
        :math:`\mathbf{L}^T` if ``transpose = 1``), see :meth:`transform_fn`. Requires :meth:`decompose` to have
        been called.

        """

        if transpose:
            R_prime = self.transform_fn(R, transpose = 1)
            R_prime = self._calc_in_flat_fast_dim(self.kf_tilde.matrix_sqrt, R_prime, transpose = 1)
        else:
            R_prime = self._calc_in_flat_fast_dim(self.kf_tilde.matrix_sqrt, R, transpose = 0)
            R_prime = self.transform_fn(R_prime, transpose = 0)
        
        return R_prime


    def matrix_inv_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
    ) -> JAXArray:
        r"""Multiplies ``R`` by :math:`\mathbf{L}^{-1} = \tilde{\mathbf{L}}^{-1}\mathbf{T}^{-1}` (or :math:`\mathbf{L}^{-T}`
        if ``transpose = 1``), see :meth:`transform_fn`. Requires :meth:`decompose` to have been called.

        """

        if transpose:
            R_prime = self._calc_in_flat_fast_dim(self.kf_tilde.matrix_inv_sqrt, R, transpose = 1)
            R_prime = self.inv_transform_fn(R_prime, transpose = 1)
        else:
            R_prime = self.inv_transform_fn(R, transpose = 0)
            R_prime = self._calc_in_flat_fast_dim(self.kf_tilde.matrix_inv_sqrt, R_prime, transpose = 0)
        
        return R_prime

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

        K_R = tensor_mult(self.Sigma, X1, X2, R, **kwargs)
        K_R += tensor_mult(self.K, X1, X2, R, **kwargs)
        
        return K_R


    # def fast_LA(self, D_tensor, b_mat, stored_values, vec_dim = 1):
        
    #     assert self.dim == 2
        
    #     N_b, N_l, N_t = D_tensor.shape

    #     assert b_mat.shape == (N_b, D_tensor.shape[vec_dim+1])

    #     b_vecs_transf = self.sigma_decomp_mats[vec_dim].matrix_inv_sqrt(b_mat.T, transpose = 0)
    #     b_vecs_transf = (self.eigen_decomp_mats[vec_dim].T @ b_vecs_transf).T

    #     block_dot_solve_vmap = jax.vmap(jax.vmap(self.kf_tilde.block_dot_solve, in_axes = (0, None, None)), in_axes = (None, 0, None))
    #     diag_tilde_tensor = self.kf_tilde.block_dot_solve(b_vecs_transf, b_vecs_transf, vec_dim)

    #     def D_transform_fn(D_tensor):
    #         D_tensor_tilde = cyclic_transpose(D_tensor, 2)
    #         for d in range(self.dim):
    #             if d != vec_sim:
    #                 D_tensor_tilde = self.sigma_decomp_mats[vec_dim].matrix_inv_sqrt(D_tensor_tilde)
    #                 D_tensor_tilde = self.eigen_decomp_mats[vec_dim].T @ D_tensor_tilde
    #                 D_tensor_tilde = cyclic_transpose(D_tensor_tilde, 1)
    #         D_tensor_tilde = cyclic_transpose(D_tensor_tilde, -2)

    #         return D_tensor_tilde

    #     D_tensor_tilde = jax.vmap(D_transform_fn, in_axes = 0)(D_tensor)

    #     T_tilde = jnp.tensordot(diag_tilde_tensor, D_tensor_tilde, axes = ([0], [0]))
    #     TKT = jnp.tensordot(D_tensor_tilde.T, T_tilde, axes = ([0], [0]))
        

    #     # Calc celerite block inverse times each time vector
    #     b_vecs = jnp.kron(b_mat, jnp.ones((N_l, 1, 1)))
    #     L_inv_b = calc_Linv_mat_vmap(*stored_values["cel_decomp"], b_vecs)

    #     # Multiply each pair of time vectors and sum over time dimension
    #     # Computes diagonal entries of diagonal matrices D_ij
    #     D_ij = L_inv_b[None, :, :, :] * L_inv_b[:, None, :, :]
    #     D_ij = D_ij.sum(3)
        
    #     # JAX/NumPy treats tensor matrix multiplication as if it is a stack of matrices stored in last two dimensions
    #     # Which works here as D_tensor is of shape (N_b, N_l, N_l)
    #     W_l_D = stored_values["W_l"].T @ D_tensor
        
    #     # Dj_W_l_D shape will initially be (N_b, N_b, N_l, N_l)
    #     Dj_W_l_D = D_ij[:, :, :, jnp.newaxis] * W_l_D[:, :, :]

    #     # We then sum over axis 1, summing over j
    #     Dj_W_l_D = Dj_W_l_D.sum(1)

    #     # Must be careful to transpose W_l_D just along the two wavelength axes
    #     # Broadcasting will result in N_b stacks of matrices multiplying together
    #     TKT = jnp.transpose(W_l_D, (0, 2, 1)) @ Dj_W_l_D

    #     # We then do our second sum over axis 0, summing over i
    #     TKT = TKT.sum(0)

    #     # Finally we invert T.T K^-1 T to get our covariance matrix
    #     Sigma_d = jnp.linalg.inv(TKT)
    #     logdetSigma_d = -jnp.linalg.slogdet(TKT)[1]
    
    #     return Sigma_d, logdetSigma_d
    