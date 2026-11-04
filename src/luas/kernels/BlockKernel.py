import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional

from luas.kernels.covtype import CovType, Diagonal

class BlockKernel(CovType):
    r"""Block diagonal covariance matrix with independent blocks along ``block_dim``, one for each index along the
    other dimension, all decomposed in parallel.

    Block :math:`n` is :math:`\sum_i \lambda_{n,i}\mathbf{K}_i`, where :math:`\mathbf{K}_i` are the covariance
    objects along ``block_dim`` (from ``Sigma`` and each term of ``K_list``) and :math:`\lambda_{n,i}` are the
    diagonal values of the matching covariance objects along the other dimension. This is the structure
    :class:`luas.LuasLasrachKernel` reduces the covariance matrix to, giving independent 1D GPs along its fast
    dimension.

    Warning:
        Internal to ``luas`` and not intended to be used directly. The interface is not stable and may change
        without notice.

    Args:
        Sigma (tuple[CovType, CovType]): The first Kronecker term. The covariance object along the dimension
            other than ``block_dim`` must be diagonal (with ``diag`` and ``wn_diag`` attributes).
        *K_list (tuple[CovType, CovType]): Further Kronecker terms, with the same requirement.
        block_dim (int, optional): The dimension each block runs along. Defaults to 1.
        non_block_dim_size (int, optional): The number of blocks, i.e. the size of the other dimension.
        use_stored_values (bool, optional): Not used.
        use_pmap (bool, optional): Experimental. Whether to decompose the blocks across devices with
            ``jax.pmap`` rather than ``jax.vmap``. Defaults to ``False``.

    """
    def __init__(
        self,
        Sigma,
        *K_list,
        block_dim = 1,
        non_block_dim_size = None,
        use_stored_values: Optional[bool] = True,
        use_pmap = False,
    ):
        r"""Creates the covariance object, see the class docstring for the arguments."""
        
        self.Sigma = Sigma
        self.K_list = K_list
        self.block_dim = block_dim
        self.non_block_dim_size = non_block_dim_size
        self.opt_name = "Block diagonal covariance matrix"
        
        if use_pmap:
            self.parallel_kwargs = {"devices":jax.devices()[:self.non_block_dim_size], "static_broadcasted_argnums":(3,)}
            self.parallel_decompose_kwargs = {"devices":jax.devices()[:self.non_block_dim_size]}
            self.parallel_method = jax.pmap
        else:
            self.parallel_kwargs = {}
            self.parallel_decompose_kwargs = {}
            self.parallel_method = jax.vmap



    def evaluate(self, X1, X2, full = True, row_idx = (None, None), col_idx = (None, None), **kwargs):
        r"""Builds the full dense block diagonal matrix. Requires :meth:`decompose` to have been called, as that sets
        up the blocks.

        Args:
            X1 (tuple): Input locations along each dimension for the rows.
            X2 (tuple): Input locations along each dimension for the columns.
            full (bool, optional): Must be ``True``.
            row_idx, col_idx (tuple, optional): Indices along each dimension passed to the block kernels.

        Returns:
            JAXArray: The dense covariance matrix.

        """
        assert full == True

        for i in range(self.non_block_dim_size):
            Sigma_eval = self.block_kernels[0].evaluate(X1[self.block_dim], X2[self.block_dim],
                                                        full = full,
                                                        row_idx = row_idx[self.block_dim],
                                                        col_idx = col_idx[self.block_dim],
                                                        **kwargs)

            if self.block_dim == 1:
                K_eval = jnp.kron(jnp.diag(self.lam_arr[:, 0]), Sigma_eval)
            else:
                K_eval = jnp.kron(Sigma_eval, jnp.diag(self.lam_arr[:, 0]))
                                
            for i in range(1, len(self.block_kernels)):
                K_i_eval = self.block_kernels[i].evaluate(X1[self.block_dim], X2[self.block_dim],
                                                          row_idx = row_idx[self.block_dim],
                                                          col_idx = col_idx[self.block_dim],
                                                          full = full, **kwargs)

                if self.block_dim == 1:
                    K_eval += jnp.kron(jnp.diag(self.lam_arr[:, i]), K_i_eval)
                else:
                    K_eval += jnp.kron(K_i_eval, jnp.diag(self.lam_arr[:, i]))

        return K_eval
            
    
    def decompose(self, X, full = True, idx = (None, None), **kwargs):
        r"""Builds every block and decomposes them in parallel with ``jax.vmap`` (or ``jax.pmap``), storing their
        factors. The log determinant is the sum of the log determinants of the blocks.

        Args:
            X (tuple): Input locations along each dimension.
            full (bool, optional): Not used.
            idx (tuple, optional): Indices along ``block_dim`` passed to each block's ``decompose``.

        Returns:
            (BlockKernel, dict): Itself and the stored value ``"logdet"``.

        """

        self.block_kernels = (self.Sigma[self.block_dim],)

        self.lam_arr = jnp.zeros((self.non_block_dim_size, 1 + len(self.K_list)))
        diag_term_0 = self.Sigma[1-self.block_dim].diag + self.Sigma[1-self.block_dim].wn_diag
        self.lam_arr = self.lam_arr.at[:, 0].set(diag_term_0)

        for i in range(len(self.K_list)):
            self.block_kernels += (self.K_list[i][self.block_dim],)

            diag_term_i = self.K_list[i][1-self.block_dim].diag + self.K_list[i][1-self.block_dim].wn_diag
            self.lam_arr = self.lam_arr.at[:, i+1].set(diag_term_i)

        def build_block(lam_arr):
            kf_block = lam_arr[0] * self.block_kernels[0]
            for i in range(1, len(self.block_kernels)):
                kf_block += lam_arr[i] * self.block_kernels[i]
            return kf_block

        def block_decompose(lam_arr):
            kf_block = build_block(lam_arr)
            kf_block, stored_values = kf_block.decompose(X[self.block_dim], idx = idx[self.block_dim])
            return kf_block.factor, stored_values["logdet"]

        block_decompose_vmap = self.parallel_method(block_decompose, in_axes = 0, out_axes = (0, 0), **self.parallel_decompose_kwargs)
        self.factor, logdet_arr = block_decompose_vmap(self.lam_arr)
        self.logdet = logdet_arr.sum()

        def matrix_inv_sqrt_calc(r, lam_arr, factor, transpose):
            kf_block = build_block(lam_arr)
            kf_block.factor = factor
            return kf_block.matrix_inv_sqrt(r, transpose = transpose)

        def matrix_sqrt_calc(r, lam_arr, factor, transpose):
            kf_block = build_block(lam_arr)
            kf_block.factor = factor
            return kf_block.matrix_sqrt(r, transpose = transpose)

        self.matrix_inv_sqrt_calc_vmap = self.parallel_method(matrix_inv_sqrt_calc, in_axes = (1-self.block_dim, 0, 0, None),
                                                          out_axes = 1-self.block_dim, **self.parallel_kwargs)
        self.matrix_sqrt_calc_vmap = self.parallel_method(matrix_sqrt_calc, in_axes = (1-self.block_dim, 0, 0, None),
                                                out_axes = 1-self.block_dim, **self.parallel_kwargs)
        

        return self, {"logdet":self.logdet}

    def matrix_sqrt(self, R, transpose = 0):
        r"""Multiplies ``R`` by the matrix square root of each block in parallel (or its transpose if ``transpose = 1``)."""
        R_prime = self.matrix_sqrt_calc_vmap(R, self.lam_arr, self.factor, transpose)

        return R_prime
    
    def matrix_inv_sqrt(self, R, transpose = 0):
        r"""Multiplies ``R`` by the inverse matrix square root of each block in parallel (or its transpose if ``transpose = 1``)."""
        R_prime = self.matrix_inv_sqrt_calc_vmap(R, self.lam_arr, self.factor, transpose)

        return R_prime
    
    def scale(self, c):
        r"""Scaling by a scalar, which is not implemented for this covariance object.

        Raises:
            Exception: Always.

        """
        raise Exception("Not implemented")

    # def matmul(self, x1, x2, R, wn = True, **kwargs):

    #     R_prime = tensor_mult(self.Sigma, X1, X2, R, **kwargs)

    #     for i in range(len(self.K_list)):
    #         R_prime += tensor_mult(self.K_list[i], X1, X2, R, **kwargs)
        
    #     return R_prime



class DifferentBlocksKernel(CovType):
    r"""Block diagonal covariance matrix where each block has its own kernel: block :math:`i` is
    :math:`a_i\mathbf{K}_i + \mathrm{diag}(\vec{d}_i)`. Used by :class:`luas.MultiLowRankKernel` with
    ``orthogonalise = True``, where the low-rank terms become independent.

    Warning:
        Internal to ``luas`` and not intended to be used directly. The interface is not stable and may change
        without notice.

    Args:
        D (JAXArray): The diagonal values :math:`\vec{d}_i` of each block, with one row per block if
            ``block_dim = 1`` (one column per block if ``block_dim = 0``).
        amplitudes (JAXArray): The amplitude :math:`a_i` of each block's kernel.
        kernel_list (list[CovType]): The kernel :math:`\mathbf{K}_i` of each block.
        block_dim (int, optional): The dimension each block runs along. Defaults to 1.
        use_stored_values (bool, optional): Not used.

    """
    def __init__(
        self,
        D,
        amplitudes,
        kernel_list,
        block_dim = 1,
        use_stored_values: Optional[bool] = True,
    ):
        r"""Creates the covariance object, see the class docstring for the arguments."""
        
        self.D = D
        self.amplitudes = amplitudes
        self.block_kernels = kernel_list
        self.block_dim = block_dim
        self.non_block_dim_size = D.shape[1-self.block_dim]
        self.opt_name = "Block diagonal covariance matrix"
        


    def evaluate(self, X1, X2, full = True, row_idx = (None, None), col_idx = (None, None), **kwargs):
        r"""Builds the dense block diagonal matrix from each block's kernel and diagonal.

        Args:
            X1 (tuple): Input locations along each dimension for the rows.
            X2 (tuple): Input locations along each dimension for the columns.
            full (bool, optional): Must be ``True``.
            row_idx, col_idx (tuple, optional): Indices along each dimension passed to the block kernels.

        Returns:
            JAXArray: The dense covariance matrix.

        """
        assert full == True

        if self.block_dim == 1:
            K_eval = jnp.diag(self.D.ravel(order = "C"))
        elif self.block_dim == 0:
            K_eval = jnp.diag(self.D.ravel(order = "F"))

        for i in range(self.non_block_dim_size):
            e_i = jnp.zeros(self.non_block_dim_size)
            e_i = e_i.at[i].set(1.)
                                
            K_i_eval = self.amplitudes[i] * self.block_kernels[i].evaluate(X1[self.block_dim], X2[self.block_dim],
                                                        row_idx = row_idx[self.block_dim],
                                                        col_idx = col_idx[self.block_dim],
                                                        full = full, **kwargs)
            if self.block_dim == 1:
                K_eval += jnp.kron(jnp.outer(e_i, e_i), K_i_eval)
            else:
                K_eval += jnp.kron(K_i_eval, jnp.outer(e_i, e_i))

        return K_eval


    def decompose(self, X, full = True, idx = (None, None), **kwargs):
        r"""Builds and decomposes each block separately. The log determinant is the sum of the log determinants of the
        blocks.

        Returns:
            (DifferentBlocksKernel, dict): Itself and the stored value ``"logdet"``.

        """

        self.K_decomp_list = []
        self.logdet = 0.
        for i in range(len(self.block_kernels)):
            if self.block_dim == 1:
                self.K_decomp_list.append(self.amplitudes[i] * self.block_kernels[i] + Diagonal(diag = self.D[i, :]))
                self.K_decomp_list[-1], stored_values_i = self.K_decomp_list[-1].decompose(X[self.block_dim], idx = idx[self.block_dim])
            else:
                self.K_decomp_list.append(self.amplitudes[i] * self.block_kernels[i] + Diagonal(diag = self.D[:, i]))
                self.K_decomp_list[-1], stored_values_i = self.K_decomp_list[-1].decompose(X[self.block_dim], idx = idx[self.block_dim])

            self.logdet += stored_values_i["logdet"]

        return self, {"logdet":self.logdet}


    def matrix_inv_sqrt(self, R, transpose = 0):
        r"""Multiplies each block of ``R`` by the inverse matrix square root of that block (or its transpose if ``transpose = 1``)."""

        R_prime = jnp.zeros_like(R)
        for i in range(len(self.block_kernels)):
            if self.block_dim == 1:
                R_prime = R_prime.at[i, :].set(self.K_decomp_list[i].matrix_inv_sqrt(R[i, :], transpose = transpose))
            else:
                R_prime = R_prime.at[:, i].set(self.K_decomp_list[i].matrix_inv_sqrt(R[:, i], transpose = transpose))
        
        return R_prime
    

    def matrix_sqrt(self, R, transpose = 0):
        r"""Multiplies each block of ``R`` by the matrix square root of that block (or its transpose if ``transpose = 1``)."""

        R_prime = jnp.zeros_like(R)
        for i in range(len(self.block_kernels)):
            if self.block_dim == 1:
                R_prime = R_prime.at[i, :].set(self.K_decomp_list[i].matrix_sqrt(R[i, :], transpose = transpose))
            else:
                R_prime = R_prime.at[:, i].set(self.K_decomp_list[i].matrix_sqrt(R[:, i], transpose = transpose))
        
        return R_prime

    
    def scale(self, c):
        r"""Scaling by a scalar, which is not implemented for this covariance object.

        Raises:
            Exception: Always.

        """
        raise Exception("Not implemented")

    # def matmul(self, x1, x2, R, wn = True, **kwargs):

    #     R_prime = tensor_mult(self.Sigma, X1, X2, R, **kwargs)

    #     for i in range(len(self.K_list)):
    #         R_prime += tensor_mult(self.K_list[i], X1, X2, R, **kwargs)
        
    #     return R_prime


class Block2x2Kernel(CovType):
    r"""Covariance matrix split into 2x2 blocks along one dimension,
    :math:`\begin{pmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{pmatrix}` with
    :math:`\mathbf{C} = \mathbf{B}^T`, decomposed with a block Cholesky decomposition:

    .. math::

        \mathbf{L} = \begin{pmatrix} \mathbf{L}_A & \mathbf{0} \\ \mathbf{C}\mathbf{L}_A^{-T} & \mathbf{L}_S \end{pmatrix},

    where :math:`\mathbf{L}_S` is the matrix square root of the Schur complement
    :math:`\mathbf{S} = \mathbf{D} - \mathbf{C}\mathbf{A}^{-1}\mathbf{B}`. The Schur complement is given directly
    as a covariance object, so it can keep its own structure. Without ``kf_B`` the matrix is block diagonal.

    Warning:
        Internal to ``luas`` and not intended to be used directly. The interface is not stable and may change
        without notice.

    Args:
        kf_A (CovType): The A block.
        kf_B (Callable, optional): Function ``kf_B(X_A, X_D, R, transpose = 0, **kwargs)`` which multiplies by
            :math:`\mathbf{B}` (or :math:`\mathbf{C}` if ``transpose = 1``). ``None`` for a block diagonal matrix.
        kf_D_CAB (CovType, optional): The Schur complement :math:`\mathbf{D} - \mathbf{C}\mathbf{A}^{-1}\mathbf{B}`,
            which is just :math:`\mathbf{D}` when ``kf_B`` is ``None``.
        kf_D (CovType, optional): The D block, only needed by :meth:`evaluate` and :meth:`matmul`. Defaults to
            ``kf_D_CAB`` when ``kf_B`` is ``None``.
        A_full (bool, optional): Passed as ``full`` when decomposing the A block. Defaults to ``True``.
        D_full (bool, optional): Passed as ``full`` when decomposing and evaluating the D block. Defaults to ``True``.
        dim_split (int, optional): The dimension the matrix is split along. Defaults to 0.
        split_loc (int, optional): The index along ``dim_split`` where the D block starts.
        split_idx (bool, optional): Whether to pass the indices of each block on to their ``decompose``
            methods. Defaults to ``False``.
        kf_B_eval (Callable, optional): Function ``kf_B_eval(X_A, X_D)`` building :math:`\mathbf{B}` densely,
            needed by :meth:`evaluate` when ``kf_B`` is given.

    """
    def __init__(self, kf_A, kf_B = None, kf_D_CAB = None, kf_D = None, A_full = True, D_full = True,
                 dim_split = 0, split_loc = None, split_idx = False, kf_B_eval = None):
        r"""Creates the covariance object, see the class docstring for the arguments."""
        self.kf_A = kf_A
        self.kf_B_matmul = kf_B
        self.kf_D_CAB = kf_D_CAB
        self.kf_D = kf_D
        self.kf_B_eval = kf_B_eval

        self.dim_split = dim_split
        self.split_loc = split_loc
        self.split_idx = split_idx
        self.A_full = A_full
        self.D_full = D_full

        if self.kf_D is None and self.kf_B_matmul is None:
            self.kf_D = self.kf_D_CAB


    def evaluate(self, x1, x2, wn = True, full = True, row_idx = None, col_idx = None, **kwargs):
        r"""Builds the full dense matrix from its blocks, mainly for testing. Needs ``kf_D`` (and ``kf_B_eval`` if
        ``kf_B`` is given).

        Returns:
            JAXArray: The dense covariance matrix.

        """

        x1_A, x1_D, row_idx_A, row_idx_D = self.x_split(x1, idx = row_idx)
        x2_A, x2_D, col_idx_A, col_idx_D = self.x_split(x2, idx = col_idx)
        
        if self.kf_B_matmul is None:
            zeros_fn = lambda x1, x2, R, **kwargs: jnp.zeros(x1_A.shape[-1], x2_D.shape[-1])
            self.kf_B_matmul = zeros_fn
        else:
            assert self.kf_B_eval is not None

        A_mat = self.kf_A.evaluate(x1_A, x2_A)

        if self.kf_B_eval is not None:
            B_mat = self.kf_B_eval(x1_A, x2_D)

        C_A_inv_B = B_mat.T @ jnp.linalg.inv(A_mat) @ B_mat
        if self.kf_D is not None:
            D_mat = self.kf_D.evaluate(x1_D, x2_D, full = self.D_full)
        else:
            D_mat = self.kf_D_CAB.evaluate(x1_D, x2_D, full = self.D_full) + C_A_inv_B

        calc_C_A_inv_B = self.kf_D.evaluate(x1_D, x2_D, full = self.D_full)
        calc_C_A_inv_B -= self.kf_D_CAB.evaluate(x1_D, x2_D, full = self.D_full)
        
        self.dense_C_A_inv_B = C_A_inv_B

        if self.dim_split == 0 or self.dim_split == 1:
            top_rows = jnp.concatenate([A_mat, B_mat], axis = 1)
            bottom_rows = jnp.concatenate([B_mat.T, D_mat], axis = 1)
        # else:

        #     raise Exception("Not implemented")
        
        return jnp.concatenate([top_rows, bottom_rows], axis = 0)
    

    def x_split(self, X, idx = None):
        r"""Splits the input locations along ``dim_split`` at ``split_loc`` into those of the A and D blocks.

        Args:
            X (tuple): Input locations along each dimension.
            idx (tuple, optional): Indices along each dimension to split in the same way.

        Returns:
            (tuple, tuple, tuple, tuple): The input locations of the A block and the D block, and their indices
            (``(None, None)`` unless ``split_idx = True``).

        """
        x_A = X[self.dim_split][..., :self.split_loc]
        x_D = X[self.dim_split][..., self.split_loc:]

        if self.dim_split == 0:
            X_A = (x_A, X[1])
            X_D = (x_D, X[1])

            if idx is None:
                idx_A = (jnp.arange(X[0].shape[-1])[:self.split_loc], jnp.arange(X[1].shape[-1]))
                idx_D = (jnp.arange(X[0].shape[-1])[self.split_loc:], jnp.arange(X[1].shape[-1]))
            else:
                idx_A = (idx[0][:self.split_loc], idx[1])
                idx_D = (idx[0][self.split_loc:], idx[1])
        else:
            X_A = (X[0], x_A)
            X_D = (X[0], x_D)

            if idx is None:
                idx_A = (jnp.arange(X[0].shape[-1]), jnp.arange(X[1].shape[-1])[:self.split_loc])
                idx_D = (jnp.arange(X[0].shape[-1]), jnp.arange(X[1].shape[-1])[self.split_loc:])
            else:
                idx_A = (idx[0], idx[1][:self.split_loc])
                idx_D = (idx[0], idx[1][self.split_loc:])

        if not self.split_idx:
            idx_A = (None, None)
            idx_D = (None, None)
        
        return X_A, X_D, idx_A, idx_D


    def decompose(self, X, idx = None, full = True, **kwargs):
        r"""Decomposes the A block and the Schur complement, and sets up the multiplications by :math:`\mathbf{B}` and
        :math:`\mathbf{C}`. The log determinant is the sum of the log determinants of the A block and the Schur
        complement.

        Args:
            X (tuple): Input locations along each dimension.
            idx (tuple, optional): Indices along each dimension.
            full (bool, optional): Not used, see ``A_full`` and ``D_full``.
            **kwargs: Passed to the A block's ``decompose``.

        Returns:
            (Block2x2Kernel, dict): Itself and the stored value ``"logdet"``.

        Raises:
            Exception: If ``kf_D_CAB`` was not given.

        """

        X_A, X_D, idx_A, idx_D = self.x_split(X, idx = idx)

        self.kf_A, stored_values_A = self.kf_A.decompose(X_A, idx = idx_A, full = self.A_full, **kwargs)

        if self.kf_D_CAB is not None:
            self.kf_D_CAB, stored_values_D = self.kf_D_CAB.decompose(X_D, idx = idx_D, full = self.D_full)
        else:
            raise Exception("Not Implemented")

        if self.kf_B_matmul is not None:
            self.B_mult = lambda R, **kwargs: self.kf_B_matmul(X_A, X_D, R, **kwargs)
            self.C_mult = lambda R, **kwargs: self.kf_B_matmul(X_A, X_D, R, transpose = 1, **kwargs)

        self.logdet = stored_values_A["logdet"] + stored_values_D["logdet"]
        
        return self, {"logdet":self.logdet}
        
    
    def matrix_inv_sqrt(self, R, transpose=0):
        r"""Multiplies ``R`` by :math:`\mathbf{L}^{-1}` (or :math:`\mathbf{L}^{-T}` if ``transpose = 1``) using block
        forward (or back) substitution. Requires :meth:`decompose` to have been called.

        """

        if self.dim_split == 0:
            R_A = R[:self.split_loc, :]
            R_D = R[self.split_loc:, :]
        elif self.dim_split == 1:
            R_A = R[:, :self.split_loc]
            R_D = R[:, self.split_loc:]

        if self.kf_B_matmul is not None:
            if transpose == 0:
                R_prime_A = self.kf_A.matrix_inv_sqrt(R_A, transpose = 0)
                R_C = self.kf_A.matrix_inv_sqrt(R_prime_A, transpose = 1)
                R_C = self.C_mult(R_C, wn = False)
                R_prime_D = self.kf_D_CAB.matrix_inv_sqrt(R_D - R_C, transpose = 0)
            else:
                R_prime_D = self.kf_D_CAB.matrix_inv_sqrt(R_D, transpose = 1)
                R_B = self.B_mult(R_prime_D, wn = False)
                R_B = self.kf_A.matrix_inv_sqrt(R_B, transpose = 0)
                R_prime_A = self.kf_A.matrix_inv_sqrt(R_A - R_B, transpose = 1)
        else:
            R_prime_A = self.kf_A.matrix_inv_sqrt(R_A, transpose = transpose)
            R_prime_D = self.kf_D_CAB.matrix_inv_sqrt(R_D, transpose = transpose)

        R_prime = jnp.concatenate([R_prime_A, R_prime_D], axis = self.dim_split)
            
        return R_prime
        
    def matrix_sqrt(self, R, transpose=0, **kwargs):
        r"""Multiplies ``R`` by the block Cholesky factor :math:`\mathbf{L}` (or :math:`\mathbf{L}^T` if
        ``transpose = 1``). Requires :meth:`decompose` to have been called.

        """

        if self.dim_split == 0:
            R_A = R[:self.split_loc, :]
            R_D = R[self.split_loc:, :]
        elif self.dim_split == 1:
            R_A = R[:, :self.split_loc]
            R_D = R[:, self.split_loc:]
        
        R_prime_A = self.kf_A.matrix_sqrt(R_A, transpose = transpose)
        R_prime_D = self.kf_D_CAB.matrix_sqrt(R_D, transpose = transpose)

        if self.kf_B_matmul is not None:
            if transpose == 0:
                R_C = self.kf_A.matrix_inv_sqrt(R_A, transpose = 1)
                R_prime_D += self.C_mult(R_C, wn = False)
            else:
                R_B = self.B_mult(R_D, wn = False)
                R_prime_A += self.kf_A.matrix_inv_sqrt(R_B, transpose = 0)
        
        R_prime = jnp.concatenate([R_prime_A, R_prime_D], axis = self.dim_split)
        
        return R_prime

    def scale(self, c):
        r"""Scaling by a scalar, which is not implemented for this covariance object.

        Raises:
            Exception: Always.

        """
        raise Exception("Not implemented")

    def matmul(self, x1, x2, R, wn = True, **kwargs):
        r"""Multiplies ``R`` by the full block matrix, multiplying each block separately. Needs ``kf_D`` if ``kf_B`` is
        given.

        Args:
            x1 (tuple): Input locations along each dimension for the rows.
            x2 (tuple): Input locations along each dimension for the columns.
            R (JAXArray): The array to multiply.
            wn (bool, optional): Whether to include white noise. Defaults to ``True``.

        Returns:
            JAXArray: The product.

        """

        if self.kf_D is None and self.kf_B_matmul is not None:
            raise Exception("Need to specify kernel function in block D for this method")

        if self.dim_split == 0:
            R_A = R[:self.split_loc, :]
            R_D = R[self.split_loc:, :]
        elif self.dim_split == 1:
            R_A = R[:, :self.split_loc]
            R_D = R[:, self.split_loc:]

        X1_A, X1_D, idx_A1, idx_D1 = self.x_split(x1)
        X2_A, X2_D, idx_A2, idx_D2 = self.x_split(x2)

        if self.kf_B_matmul is None:
            K_R_A = self.kf_A.matmul(X1_A, X2_A, R_A)
            K_R_D = self.kf_D.matmul(X1_D, X2_D, R_D)
        else:
            K_R_A = self.kf_A.matmul(X1_A, X2_A, R_A) + self.kf_B_matmul(X1_A, X2_D, R_D)
            K_R_D = self.kf_B_matmul(X1_D, X2_A, R_A) + self.kf_D.matmul(X1_D, X2_D, R_D, row_idx = idx_D1, col_idx = idx_D2)

        K_R = jnp.concatenate([K_R_A, K_R_D], axis = self.dim_split)
        
        return K_R
    

