from luas.kernels.tinygp_ext import Multiband
from luas.kernels.covtype import CovType
import luas.kernels.quasisep
from luas.luas_types import is_scalar
import jax.numpy as jnp
import tinygp

class MixingMatQuasisep(CovType):
    r"""Covariance matrix of a linear model of coregionalisation: a sum of Kronecker products of a rank-one
    mixing term along one dimension and a kernel along ``fast_dim``, plus a diagonal and an optional noise model,

    .. math::

        \mathbf{K} = \sum_i \vec{m}_i\vec{m}_i^T \otimes \mathbf{K}_i + \mathrm{diag}(\vec{d} + \vec{d}_{\mathrm{wn}}) [+ \mathbf{N}],

    where :math:`\vec{m}_i` is column :math:`i` of the mixing matrix. Kept quasiseparable with the multiband method of Gordon et al. (2020), so it is decomposed in linear time. Used by the low-rank Kronecker
    kernels for the dense part of their decomposition.

    Warning:
        Internal to ``luas`` and not intended to be used directly. The interface is not stable and may change
        without notice.

    Args:
        mixing_mat (JAXArray): The mixing matrix, of shape ``(N_mix, rank)``.
        kernel_list (list[CovType]): The covariance object :math:`\mathbf{K}_i` along ``fast_dim`` for each column.
        noise_model (optional): An extra matrix :math:`\mathbf{N}` added to the covariance (a ``tinygp`` ``SymmQSM``).
        diag (JAXArray | Scalar, optional): Value(s) added to the diagonal. Defaults to 0.
        wn_diag (JAXArray | Scalar, optional): White noise value(s) added to the diagonal, only included when
            ``wn = True``. Defaults to 0.
        params (optional): Metadata about the parameters the object depends on.
        fast_dim (int, optional): The dimension the kernels run along. Defaults to 1.
        use_block (bool, optional): Whether to use ``tinygp``'s block representation when kernels are summed.
            Defaults to ``True``.

    """
    def __init__(self, mixing_mat, kernel_list, noise_model = None,
                 diag = 0., wn_diag = 0., params = None, fast_dim = 1, use_block = True, **kwargs):
        r"""Creates the object, see the class docstring for the arguments."""
        self.mixing_mat = mixing_mat # shape [N_l, N_alpha]
        self.kernel_list = kernel_list # N_alpha list of tinygp kernel functions
        self.diag = diag
        self.wn_diag = wn_diag
        self.rank = mixing_mat.shape[1]
        self.noise_model = noise_model
        self.params = params
        self.fast_dim = fast_dim
        self.use_block = use_block
        self.opt_name = "Gordon et al. (2020)/tinygp method"
    
    def _flatten_R(self, R):
        r"""Flattens a 2D array into a vector in the order the rows and columns of the matrix are stored in."""
        if self.fast_dim == 0:
            return R.ravel("C")
        else:
            return R.ravel("F")

    def _reshape_R(self, r, R_shape):
        r"""Reshapes a vector back into a 2D array of shape ``R_shape``, inverting :meth:`_flatten_R`."""
        if self.fast_dim == 0:
            return r.reshape(R_shape, order='C')
        else:
            return r.reshape(R_shape, order='F')
        

    def _tinygp_coords(self, X, full = False, idx = (None, None)):
        r"""Builds the ``(x, index)`` coordinates for the multiband ``tinygp`` kernel: each input location along
        ``fast_dim`` repeated for every row of the mixing matrix, with interleaved indices.

        """
        # assert full or idx is not None

        cel_vec = X[self.fast_dim]
        non_cel_vec = X[1-self.fast_dim]
        
        x_t_long = jnp.kron(cel_vec, jnp.ones(non_cel_vec.shape[-1]))
        multiband_idx = jnp.arange(x_t_long.shape[-1])

        if idx[self.fast_dim] is not None:
            multiband_idx_2D = jnp.add.outer(idx[1-self.fast_dim], self.mixing_mat.shape[0] * idx[self.fast_dim])
            multiband_idx = multiband_idx_2D.ravel("F")
        
        return (x_t_long, multiband_idx)


    def _to_symm_qsm(self, X, wn = True, full = False, idx = (None, None)):
        r"""Builds the matrix as a ``tinygp`` ``SymmQSM``: a ``Multiband`` kernel for each column of the mixing matrix,
        summed, plus the diagonal and noise model.

        Returns:
            SymmQSM: The symmetric quasiseparable matrix.

        """
        
        for i in range(self.rank):
            if i == 0:
                self.kf_2D = Multiband(
                    kernel=self.kernel_list[0].tinygp_kf,
                    band_amplitudes=self.mixing_mat[:, 0],
                )
            else:
                new_term = Multiband(
                    kernel=self.kernel_list[i].tinygp_kf,
                    band_amplitudes=self.mixing_mat[:, i],
                )
                self.kf_2D = tinygp.kernels.quasisep.Sum(self.kf_2D, new_term, use_block = self.use_block)

        tiny_coords = self._tinygp_coords(X, full = full, idx = idx)
        quasisep_cov = self.kf_2D.to_symm_qsm(tiny_coords)

        diag = (self.diag + wn*self.wn_diag)*jnp.ones(tiny_coords[0].shape[-1])
        quasisep_cov += tinygp.noise.Diagonal(diag).to_qsm()
        
        if self.noise_model is not None:
            quasisep_cov += self.noise_model
    
        return quasisep_cov
        

    def decompose(self, X, wn = True, **kwargs):
        r"""Calculates the quasiseparable Cholesky decomposition and the log determinant.

        Returns:
            (MixingMatQuasisep, dict): Itself and the stored value ``"logdet"``.

        """

        self.quasi_mat = self._to_symm_qsm(X, wn = wn, **kwargs)
        self.factor = self.quasi_mat.cholesky()
        self.logdet = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
        return self, {"logdet":self.logdet}
        

    def evaluate(self, X1, X2, row_idx = None, col_idx = None, full = False, wn = True, flip_kron = False, calc_diag = True):
        r"""Evaluates the dense covariance matrix (or the sub-matrix given by ``row_idx`` and ``col_idx``).

        Args:
            X1 (tuple): Input locations along each dimension for the rows.
            X2 (tuple): Input locations along each dimension for the columns.
            row_idx, col_idx (tuple, optional): Indices along each dimension if ``full = False``.
            full (bool, optional): Whether to evaluate the full matrix. Defaults to ``False``.
            wn (bool, optional): Whether to include ``wn_diag``. Defaults to ``True``.
            flip_kron (bool, optional): Which dimension is the outer index of the Kronecker products for sub-matrices.
                Defaults to ``False``.
            calc_diag (bool, optional): Whether to include the diagonal for sub-matrices. Defaults to ``True``.

        Returns:
            JAXArray: The covariance matrix.

        Raises:
            Exception: If a sub-matrix is requested and there is a noise model.

        """

        if full:
            # Really not optimised, nor should it be I suppose
            return self._to_symm_qsm(X1, wn = wn, full = True).to_dense()
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if not idx_specified:
                raise Exception("""Cannot evaluate non-stationary diagonal covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                            )

            # Evaluate diagonal values if included
            if calc_diag:
                N_non_cel = self.mixing_mat.shape[0]
                ind1 = jnp.add.outer(N_non_cel * row_idx[0], row_idx[1]).ravel("C")
                ind2 = jnp.add.outer(N_non_cel * col_idx[0], col_idx[1]).ravel("C")

                mask = ind1[:, None] == ind2[None, :]

                diag_val = self.diag + wn * self.wn_diag
                if is_scalar(diag_val):
                    K_eval = jnp.where(mask, diag_val, 0.0)
                else:
                    K_eval = jnp.where(mask, diag_val[ind1[:, None] * mask], 0.0)

            if self.noise_model is not None:
                raise Exception("MixingMatQuasisep CovType with noise model cannot be evaluated for non-full covariance matrices")

            for i in range(self.rank):
                K_i_eval = self.kernel_list[i](X1[self.fast_dim], X2[self.fast_dim], full = False,
                                               row_idx = row_idx[self.fast_dim], col_idx = col_idx[self.fast_dim])

                mixing_eval = self.mixing_mat[row_idx[1-self.fast_dim], i:i+1] @ self.mixing_mat[col_idx[1-self.fast_dim], i:i+1].T

                if flip_kron:
                    if self.fast_dim == 0:
                        K_kron = jnp.kron(mixing_eval, K_i_eval)
                    else:
                        K_kron = jnp.kron(K_i_eval, mixing_eval)
                else:
                    if self.fast_dim == 0:
                        K_kron = jnp.kron(K_i_eval, mixing_eval)
                    else:
                        K_kron = jnp.kron(mixing_eval, K_i_eval)

                if i == 0 and not calc_diag:
                    K_eval = K_kron
                else:
                    K_eval += K_kron

            return K_eval

        
    def matrix_inv_sqrt(self, R, transpose=0):
        r"""Multiplies the 2D array ``R`` by :math:`\mathbf{L}^{-1}` (or :math:`\mathbf{L}^{-T}`), flattening and reshaping it. Requires :meth:`decompose` to have been called."""
        
        R_shape = R.shape
        r = self._flatten_R(R)

        if transpose:
            r_prime = self.factor.transpose().solve(r)
        else:
            r_prime = self.factor.solve(r)

        R_prime = self._reshape_R(r_prime, R_shape)
            
        return R_prime

    def matrix_sqrt(self, R, transpose=0, **kwargs):
        r"""Multiplies the 2D array ``R`` by :math:`\mathbf{L}` (or :math:`\mathbf{L}^T`), flattening and reshaping it. Requires :meth:`decompose` to have been called."""
        
        R_shape = R.shape
        r = self._flatten_R(R)

        if transpose:
            r_prime = self.factor.transpose() @ r
        else:
            r_prime = self.factor @ r

        R_prime = self._reshape_R(r_prime, R_shape)
            
        return R_prime

    def inv_sqrt_transform(self, K):
        r"""Whitening another covariance object, which is not implemented for this covariance object.

        Raises:
            Exception: Always.

        """
        raise Exception("Inv_sqrt_transform not implemented for MixingMatQuasisep CovType")

    def scale(self, c):
        r"""Returns a ``MixingMatQuasisep`` for :math:`c\mathbf{K}`, with the mixing matrix multiplied by :math:`\sqrt{c}`."""
        
        if self.noise_model is not None:
            new_noise_model = self.noise_model * c
        else:
            new_noise_model = None

        # c should be positive!
        
        return MixingMatQuasisep(self.mixing_mat * jnp.sqrt(c), self.kernel_list, diag = self.diag * c, wn_diag = self.wn_diag * c,
                                 fast_dim = self.fast_dim, noise_model = new_noise_model)

    def matmul(self, x1, x2, R, wn = True, **kwargs):
        r"""Multiplies the 2D array ``R`` by the matrix in linear time, flattening and reshaping it."""

        R_shape = R.shape
        r = self._flatten_R(R)

        self.quasi_mat = self._to_symm_qsm(x1, wn = wn)
        r_prime = self.quasi_mat @ r
        
        R_prime = self._reshape_R(r_prime, R_shape)
            
        return R_prime
    

    def householder_transform(self, X, u):
        r"""Returns the matrix transformed by a Householder reflection with vector ``u`` along ``fast_dim``
        (see ``covtype.General.householder_transform``), by adding two rank-one terms to each kernel. Note that this
        modifies ``kernel_list`` in place.

        Args:
            X (tuple): Input locations along each dimension.
            u (JAXArray): The unit Householder vector along ``fast_dim``.

        Returns:
            MixingMatQuasisep: The transformed covariance object.

        """
        
        for i in range(self.rank):
            w_i = self.kernel_list[i].matmul(X[self.fast_dim], X[self.fast_dim], u, full = True)

            # Can this ever be super close to zero? Seems unlikely?
            u_dot_w = jnp.dot(u, w_i)
            alpha = 2 * u_dot_w * u - w_i
            
            self.kernel_list[i] += luas.kernels.quasisep.Amplitude(alpha, const = 1/u_dot_w)
            self.kernel_list[i] += luas.kernels.quasisep.Amplitude(w_i, const = -1/u_dot_w)
        
        return MixingMatQuasisep(self.mixing_mat, self.kernel_list, diag = self.diag, wn_diag = self.wn_diag,
                                 fast_dim = self.fast_dim, noise_model = self.noise_model)
    

    def __add__(self, other):
        r"""Adds another ``MixingMatQuasisep`` with the same ``fast_dim`` and mixing dimension, by joining their mixing matrices and
        kernel lists.

        Raises:
            Exception: If ``fast_dim`` or the length of the mixing dimension differ.

        """

        if isinstance(other, MixingMatQuasisep):
            if self.fast_dim != other.fast_dim or self.mixing_mat.shape[0] != other.mixing_mat.shape[0]:
                raise Exception("Cannot add two MixingMatQuasisep CovTypes with different fast_dim or different lengths")

            new_mixing_mat = jnp.concatenate([self.mixing_mat, other.mixing_mat], axis = 1)
            new_kernel_list = self.kernel_list + other.kernel_list

            if self.noise_model is not None and other.noise_model is not None:
                new_noise_model = self.noise_model + other.noise_model
            elif self.noise_model is not None:
                new_noise_model = self.noise_model
            elif other.noise_model is not None:
                new_noise_model = other.noise_model
            else:
                new_noise_model = None
                
            K_sum = MixingMatQuasisep(new_mixing_mat, new_kernel_list, diag = self.diag + other.diag, wn_diag = self.wn_diag + other.wn_diag,
                                      fast_dim = self.fast_dim, noise_model = new_noise_model)
            
        else:
            return NotImplemented
            
        return K_sum
    

