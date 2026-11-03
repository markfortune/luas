import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional

import luas.kernels
from luas.kernels.covtype import CovType, Identity, Diagonal, Outer
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
from luas.kronecker_fns import tensor_mult, vmap_for_tensors, cyclic_transpose
from luas.kernel_interface import read_K_list_2D
from luas.kernels.householder import householder_qr
from luas.kernels.BlockKernel import Block2x2Kernel
from luas.kernels.GeneralKernel import GeneralKernel
from luas.kernels.LuasPlusMultiLowRankKernel import LuasPlusMultiLowRankKernel
from luas.stable_gradients import stable_eigh

__all__ = [
    "ExtendedLuasKernel",
]

class ExtendedLuasKernel(CovType):
    r"""The most general Kronecker kernel optimisation: measurement covariance, a
    full-rank 2D kernel term, and low-rank kernel terms along both dimensions.

    Handles kernel functions of the form

    .. math::

        k(\lambda, \lambda', t, t') &= s_\lambda(\lambda, \lambda')\, s_t(t, t')
        + k_{\lambda; \mathrm{2D}}(\lambda, \lambda')\, k_{t; \mathrm{2D}}(t, t') \\
        &+ \sum_{i=1}^{N_\alpha} \alpha_i(\lambda)\alpha_i(\lambda')\, k_{t; i}(t, t')
        + \sum_{j=1}^{N_\beta} k_{\lambda; j}(\lambda, \lambda')\, \beta_j(t)\beta_j(t'),

    which on a rectangular grid gives the covariance matrix

    .. math::

        \mathbf{K} = \mathbf{\Sigma}_\lambda \otimes \mathbf{\Sigma}_t
        + \mathbf{K}_{\lambda; \mathrm{2D}} \otimes \mathbf{K}_{t; \mathrm{2D}}
        + \sum_{i=1}^{N_\alpha} \boldsymbol{\alpha}_{\lambda; i}
        \boldsymbol{\alpha}_{\lambda; i}^T \otimes \mathbf{K}_{t; i}
        + \sum_{j=1}^{N_\beta} \mathbf{K}_{\lambda; j} \otimes
        \boldsymbol{\beta}_{t; j} \boldsymbol{\beta}_{t; j}^T.

    All the other 2D Kronecker kernel optimisations in ``luas`` are special cases of
    this kernel structure. Any valid kernels can be used, but no quasiseparable
    structure is exploited, so it is best suited to modest dataset sizes or GPUs.

    **Method.** Whitens with :math:`\mathbf{\Sigma}_\lambda` and
    :math:`\mathbf{\Sigma}_t`, then applies a Householder QL decomposition to the
    low-rank vectors along both dimensions. The transformed matrix is decomposed as
    a 2x2 block matrix. The A block is handled by
    :class:`luas.LuasPlusMultiLowRankKernel`, and the
    :math:`\mathbf{C}\mathbf{A}^{-1}\mathbf{B}` term exploits sparsity in the
    blocks. :math:`\mathbf{D} - \mathbf{C}\mathbf{A}^{-1}\mathbf{B}` is decomposed as
    a general :math:`(N_\alpha N_t, N_\alpha N_t)` matrix.

    **Runtime scaling.**

    .. math::

        \mathcal{O}\left((1 + N_\alpha^3) N_t^3 + (1 + N_\beta^3) N_\lambda^3
        + N_\lambda N_t (N_\lambda + N_t + N_\alpha^2 + N_\beta^2)\right)

    runtime and :math:`\mathcal{O}(N_\alpha^2 N_t^2 + N_\beta^2 N_\lambda^2 +
    N_\alpha N_\beta N_\lambda N_t)` memory. For small :math:`N_\alpha`,
    :math:`N_\beta` this is similar to :class:`luas.LuasKernel`. It can be
    substantially faster on a GPU.

    **Selection.** :class:`luas.GP` selects this optimisation automatically when the
    kernel function returns the noise term, one full-rank 2D term, and low-rank
    terms along both dimensions.

    Warning:
        Requires exactly one full-rank 2D term, at least one low-rank term along
        each dimension, :math:`0 < N_\alpha < N_\lambda` and
        :math:`0 < N_\beta < N_t`. The noise term must be positive definite and be
        the **first** tuple returned by the kernel function. Low-rank terms are only
        recognised as rank-1 outer products (e.g. ``luas.kernels.lowrank.Constant``
        or ``luas.kernels.lowrank.Amplitude``).

    Example:
        Combining every type of term from the ``luas`` tutorial:

        .. code-block:: python

            def kf(p, X):
                s_l = luas.kernels.Noise(jnp.exp(p["log_sigma"]))
                s_l += jnp.exp(2*p["log_h_readout"][0]) * luas.kernels.Exp(jnp.exp(p["log_l_readout"][0]))
                s_t = luas.kernels.KroneckerDelta()

                # Full-rank 2D term with a wavelength-dependent amplitude
                k_l = luas.kernels.lowrank.Amplitude(jnp.exp(p["log_h_l_vec"])) * luas.kernels.Matern52(jnp.exp(p["log_l_l"][0]))
                k_t = luas.kernels.Matern32(jnp.exp(p["log_l_t"][0]))

                # alpha term: fixed vector along wavelength, kernel in time (common-mode systematics)
                alpha_l1 = luas.kernels.lowrank.Constant(jnp.exp(p["log_h_const"][0]))
                k_t1 = luas.kernels.Matern32(jnp.exp(p["log_l_t1"][0]))

                # beta term: fixed vector along time, kernel in wavelength (settling ramp)
                k_l1 = jnp.exp(2*p["log_h_ramp"][0]) * luas.kernels.Matern32(jnp.exp(p["log_l_l_ramp"][0]))
                settle_vec = jnp.exp(-(X[1] - X[1][0])/jnp.exp(p["log_l_t_ramp"][0]))
                beta_t1 = luas.kernels.lowrank.Amplitude(settle_vec)

                return (s_l, s_t), (k_l, k_t), (alpha_l1, k_t1), (k_l1, beta_t1)

            gp = luas.GP(p, (x_l, x_t), kf = kf)  # Uses ExtendedLuasKernel

    Args:
        Sigma (tuple[CovType, CovType]): The covariance objects
            :math:`(\mathbf{\Sigma}_\lambda, \mathbf{\Sigma}_t)` for the noise term.
            Both must be positive definite.
        *K_list (tuple[CovType, CovType]): Exactly one full-rank 2D term plus the
            low-rank terms, each as a tuple with one covariance object per dimension.
        use_stored_values (bool, optional): Currently unused.
        **kwargs: Ignored, accepted so that all Kronecker kernels share a common
            constructor interface.

    See Also:
        :class:`luas.LuasPlusMultiLowRankKernel` for low-rank terms along only one
        dimension, which can exploit quasiseparable kernels.
        :class:`luas.DoubleMultiLowRankKernel` for low-rank terms along both
        dimensions without a full-rank 2D term.
    """

    def __init__(
        self,
        Sigma,
        *K_list,
        use_stored_values: Optional[bool] = False,
        **kwargs,
    ):
        
        self.Sigma = Sigma[0], Sigma[1]
        self.K_list = K_list
        self.opt_name = "ExtendedLuasKernel"
        
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            raise NotImplementedError("Use of pre-calculated stored values not yet implemented.")
        else:
            self.decompose = self.decompose_no_stored_values
    
    

    def decompose_no_stored_values(
        self,
        X: Tuple[JAXArray],
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:

        self.N_l = X[0].shape[-1]
        self.N_t = X[1].shape[-1]

        dense_kron, (alpha_mat, alpha_kernel_list), (beta_mat, beta_kernel_list) = read_K_list_2D(self.K_list, X)

        self.N_alpha = alpha_mat.shape[1]
        self.N_beta = beta_mat.shape[1]

        # Required for this optimisation to work
        kernel_valid = 0 < self.N_alpha < self.N_l and 0 < self.N_beta < self.N_t
        if not kernel_valid:
            raise Exception("ExtendedLuasKernel requires 0 < N_alpha < N_l and 0 < N_beta < N_t for a dataset of shape (N_l, N_t).")

        ######### Do transforms in both dims #########
        # Decompose slow Sigma mat and add its contribution to the log determinant
        self.Sigma_transf = [None, None]
        self.Sigma_transf[0], stored_values_Sigma_l = self.Sigma[0].decompose(X[0])
        stored_values["logdet"] = self.N_t*stored_values_Sigma_l["logdet"]

        self.Sigma_transf[1], stored_values_Sigma_t = self.Sigma[1].decompose(X[1])
        stored_values["logdet"] += self.N_l*stored_values_Sigma_t["logdet"]

        alpha_mat = self.Sigma_transf[0].matrix_inv_sqrt(alpha_mat)
        beta_mat = self.Sigma_transf[1].matrix_inv_sqrt(beta_mat)
        
        # Get Householder transform of all Outer covariances
        self.J_A, stored_values["U_A"], self.householder_transform_A = householder_qr(alpha_mat, reverse = True)
        self.J_B, stored_values["U_B"], self.householder_transform_B = householder_qr(beta_mat, reverse = True)
        
        ####### Transform kernels #######

        self.K_transf = [None, None]
        self.K_transf[0] = self.Sigma_transf[0].inv_sqrt_transform(dense_kron[0], X[0])
        for i in range(self.N_alpha):
            self.K_transf[0] = self.K_transf[0].householder_transform(X[0], stored_values["U_A"][:, i])

            # Transform alpha terms
            alpha_kernel_list[i] = self.Sigma_transf[1].inv_sqrt_transform(alpha_kernel_list[i], X[1])
            for j in range(self.N_beta):
                alpha_kernel_list[i] = alpha_kernel_list[i].householder_transform(X[1], stored_values["U_B"][:, j])


        K_transf0_eval = self.K_transf[0].evaluate(X[0], X[0], full = True)
        self.K_transf0_A = K_transf0_eval[:-self.N_alpha, :-self.N_alpha]
        self.K_transf0_B = K_transf0_eval[:-self.N_alpha, -self.N_alpha:]
        self.K_transf0_D = K_transf0_eval[-self.N_alpha:, -self.N_alpha:]
        
        self.lam_K_transf0_A, self.Q_K_transf0_A = stable_eigh(self.K_transf0_A)

        self.K_beta_A = jnp.zeros((self.N_beta, self.N_l - self.N_alpha, self.N_l - self.N_alpha))
        self.K_beta_B = jnp.zeros((self.N_beta, self.N_l - self.N_alpha, self.N_alpha))
        self.K_beta_D = jnp.zeros((self.N_beta, self.N_alpha, self.N_alpha))

        self.K_transf[1] = self.Sigma_transf[1].inv_sqrt_transform(dense_kron[1], X[1])
        for j in range(self.N_beta):
            self.K_transf[1] = self.K_transf[1].householder_transform(X[1], stored_values["U_B"][:, j])

            # Transform beta kernel terms
            beta_kernel_list[j] = self.Sigma_transf[0].inv_sqrt_transform(beta_kernel_list[j], X[0])
            for i in range(self.N_alpha):
                beta_kernel_list[j] = beta_kernel_list[j].householder_transform(X[0], stored_values["U_A"][:, i])

            K_beta_j_eval = beta_kernel_list[j].evaluate(X[0], X[0], full = True)
            
            self.K_beta_A = self.K_beta_A.at[j, :, :].set(K_beta_j_eval[:-self.N_alpha, :-self.N_alpha])
            self.K_beta_B = self.K_beta_B.at[j, :, :].set(K_beta_j_eval[:-self.N_alpha, -self.N_alpha:])
            self.K_beta_D = self.K_beta_D.at[j, :, :].set(K_beta_j_eval[-self.N_alpha:, -self.N_alpha:])


        ######### Build A Block #########


        def K_A_transform_fn(R, transpose = 0):
            if transpose:
                R_prime = self.Q_K_transf0_A.T @ R
            else:
                R_prime = self.Q_K_transf0_A @ R
            return R_prime
        
        def K_A_inv_transform_fn(R, transpose = 0):
            if transpose:
                R_prime = self.Q_K_transf0_A @ R
            else:
                R_prime = self.Q_K_transf0_A.T @ R
            return R_prime

        K_A_kernel = ((Identity(), Identity()),
                      (Diagonal(diag=self.lam_K_transf0_A), self.K_transf[1]),
                     )
        
        for j in range(self.N_beta):
            K_A_kernel += ((luas.kernels.Fixed(self.Q_K_transf0_A.T @ self.K_beta_A[j] @ self.Q_K_transf0_A, ignore_idx = True), Outer(self.J_B[:, j])),)

        kf_A = LuasPlusMultiLowRankKernel(*K_A_kernel, eigen_both = True, fast_dim=1,
                                        transform=False, transform_fn = K_A_transform_fn, inv_transform_fn = K_A_inv_transform_fn)
        kf_A, stored_values_A = kf_A.decompose((X[0][:-self.N_alpha], X[1]))

        # C_A_inv_B calc

        C_A_inv_B = self._calc_C_A_inv_B(kf_A, X)

        
        ######### Build D block #########

        kernel_D = ((Identity(), Identity()),)

        for i in range(self.N_alpha):
            kernel_D += ((Outer(self.J_A[:, i]), alpha_kernel_list[i]),)

        K_t_eval = self.K_transf[1].evaluate(X[1], X[1], full = True)
        K_D_dense = jnp.kron(self.K_transf0_D, K_t_eval)
        for j in range(self.N_beta):
            beta_vec = jnp.zeros(self.N_t)
            beta_vec = beta_vec.at[-self.N_beta:].set(self.J_B[:, j])
            K_D_dense += jnp.kron(self.K_beta_D[j], jnp.outer(beta_vec, beta_vec))
        

        kf_D_CAB = GeneralKernel(*kernel_D, dense_mat = K_D_dense - C_A_inv_B)
        kf_D = GeneralKernel(*kernel_D, dense_mat = K_D_dense)


        ######### Define full 2x2 Block kernel and decompose #########
        self.kf_tilde = Block2x2Kernel(kf_A = kf_A, kf_B = self.K_B_matmul, kf_D_CAB = kf_D_CAB, kf_D = kf_D,
                                       dim_split = 0, split_loc = self.N_l-self.N_alpha,
                                       split_idx = True, kf_B_eval = self.kf_B_eval)
        
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose(X, full = True)

        stored_values["logdet"] += stored_values["kf_tilde_stored"]["logdet"]
        self.logdet = stored_values["logdet"]

        return self, stored_values


    def K_B_matmul(
        self,
        X1,
        X2,
        R: JAXArray,
        transpose = 0,
        **kwargs,
    ) -> JAXArray:

        if transpose:
            R_prime = self.K_transf0_B.T @ R
            R_prime = self.K_transf[1].matmul(X1[1], X2[1], R_prime.T, full = True).T

            R_C_corr = jnp.zeros((R_prime.shape[0], self.N_beta))
            for j in range(self.N_beta):
                R_prime_j = self.K_beta_B[j].T @ R
                R_prime_j = R_prime_j[:, -self.N_beta:] @ self.J_B[:, j]
                R_C_corr += jnp.outer(R_prime_j, self.J_B[:, j])

            R_prime = R_prime.at[:, -self.N_beta:].set(R_prime[:, -self.N_beta:] + R_C_corr)
        else:
            R_prime = self.K_transf0_B @ R
            R_prime = self.K_transf[1].matmul(X1[1], X2[1], R_prime.T, full = True).T

            R_C = R[:, -self.N_beta:]
            R_C_corr = jnp.zeros((R_prime.shape[0], self.N_beta))
            for j in range(self.N_beta):
                R_prime_j = R_C @ self.J_B[:, j]
                R_prime_j = jnp.outer(R_prime_j, self.J_B[:, j])
                R_C_corr += self.K_beta_B[j] @ R_prime_j
            R_prime = R_prime.at[:, -self.N_beta:].set(R_prime[:, -self.N_beta:] + R_C_corr)
        
        return R_prime
    

    def _calc_C_A_inv_B(self, kf_A, X):
        """
        This calculation is pretty cursed and could probably be optimised or cleaned up quite a lot 
        if this optimisation actually gets used for something
        """

        K_tilde_B = self.K_transf[1].evaluate(X[1], X[1], full = True)

        self.lam_K_transf1_A, self.Q_K_transf1_A = kf_A.lam_fast_A, kf_A.Q_fast_A
        self.K_transf1_B, self.K_transf1_D = kf_A.V_K_B, kf_A.V_K_D
        block_AD_matrix_inv_sqrt = kf_A.kf_tilde.kf_D_CAB.matrix_inv_sqrt

        # Calc R_A_K_inv_R_A

        Q_K_transf0_B = self.Q_K_transf0_A.T @ self.K_transf0_B

        R_A = jnp.kron(jnp.ones((self.N_t - self.N_beta, 1, 1)), Q_K_transf0_B) # shape (N_t - N_beta, N_l - N_alpha, N_alpha)
        lam_A = 1 + jnp.outer(self.lam_K_transf1_A, self.lam_K_transf0_A) # shape (N_t - N_beta, N_l - N_alpha)
        inv_lam_A = 1/lam_A

        Lam_Q_R_A = jnp.einsum("ij,ijm->ijm", inv_lam_A, R_A)
        R_A_K_A_inv_R_A = jnp.einsum("ijm,ijn->imn", R_A, Lam_Q_R_A)

        n = jnp.arange(self.N_t - self.N_beta)
        R_A_K_A_inv_R_A_dense = jnp.zeros((self.N_alpha, self.N_t - self.N_beta, self.N_alpha, self.N_t - self.N_beta))
        R_A_K_A_inv_R_A_dense = R_A_K_A_inv_R_A_dense.at[:, n, :, n].add(R_A_K_A_inv_R_A)

        # vmapped function to calculate fast D_inv_sqrt for tensors of shape (N_beta, N_beta_vecs, N_l - N_alpha, N_alpha_vecs)
        # returns matrix of shape (N_beta * N_l - N_alpha, N_alpha_vecs, N_beta_vecs)
        def D_inv_sqrt_R(D_inv_sqrt, R):
            return D_inv_sqrt(R).ravel()
        D_inv_sqrt_K = jax.vmap(jax.vmap(D_inv_sqrt_R, in_axes = (None, 1), out_axes = 1), in_axes = (None, 3), out_axes = 1)

        # Calc L_D_inv_C_A_inv_R_A
        C_Q_R_A = jnp.einsum("ij,jkm->ijkm", self.K_transf1_B.T @ self.Q_K_transf1_A, Lam_Q_R_A)
        C_A_inv_R_A = jnp.einsum("k,ijkl->ijkl", self.lam_K_transf0_A, C_Q_R_A)
        L_D_inv_C_A_inv_R_A = D_inv_sqrt_K(block_AD_matrix_inv_sqrt, C_A_inv_R_A)
        
        # Calc L_D_inv_R_D
        K_1_D = jnp.eye(self.N_beta)
        K_0 = self.Q_K_transf0_A.T @ self.K_transf0_B
        total_K_cov = K_1_D[:, :, None, None] * K_0[None, None, :, :]
        L_D_inv_R_D = D_inv_sqrt_K(block_AD_matrix_inv_sqrt, total_K_cov)

         # Calc L_D_inv_R_D2
        total_beta_cov = jnp.zeros((self.N_beta, self.N_beta, self.N_l - self.N_alpha, self.N_alpha))
        for k in range(self.N_beta):
            K_1_k = jnp.outer(self.J_B[:, k], self.J_B[:, k])
            K_0_k = self.Q_K_transf0_A.T @ self.K_beta_B[k]
            total_beta_cov += K_1_k[:, :, None, None] * K_0_k[None, None, :, :]

        L_D_inv_R_D2 = D_inv_sqrt_K(block_AD_matrix_inv_sqrt, total_beta_cov)


        # Calculate dot products of matrix inv sqrts for R_A, R_D and R_D2
        R_A_B_D_inv_C_R_A = jnp.einsum("ijk,imn->jkmn", L_D_inv_C_A_inv_R_A, L_D_inv_C_A_inv_R_A)
        R_D_K_D_inv_R_D = jnp.einsum("ijk,imn->jkmn", L_D_inv_R_D, L_D_inv_R_D)
        R_D_K_D_inv_R_A = jnp.einsum("ijk,imn->jkmn", L_D_inv_R_D, L_D_inv_C_A_inv_R_A)

        R_D2_K_D_inv_R_D = jnp.einsum("ijk,imn->jkmn", L_D_inv_R_D2, L_D_inv_R_D)
        R_D2_K_D_inv_R_A = jnp.einsum("ijk,imn->jkmn", L_D_inv_R_D2, L_D_inv_C_A_inv_R_A)
        R_D2_K_D_inv_R_D2 = jnp.einsum("ijk,imn->jkmn", L_D_inv_R_D2, L_D_inv_R_D2)

        # Combine into full matrix (N_alpha * N_t, N_alpha * N_t) block by block
        full_A = jnp.zeros((self.N_alpha, self.N_t, self.N_alpha, self.N_t))
        full_A = full_A.at[:, -self.N_beta:, :, :-self.N_beta].set(-R_D_K_D_inv_R_A)
        full_A = full_A.at[:, :-self.N_beta, :, :-self.N_beta].set(R_A_K_A_inv_R_A_dense + R_A_B_D_inv_C_R_A)
        full_A = full_A.at[:, :-self.N_beta, :, -self.N_beta:].set(jnp.transpose(-R_D_K_D_inv_R_A, [2, 3, 0, 1]))
        full_A = full_A.at[:, -self.N_beta:, :, -self.N_beta:].set(R_D_K_D_inv_R_D)

        # Do transformations for I \otimes K_tilde1 factored out of B term so far
        N_total = self.N_alpha * self.N_t
        full_A_dense = full_A.reshape(N_total, N_total)

        B_Q = jax.scipy.linalg.block_diag(self.Q_K_transf1_A, jnp.eye(self.N_beta))
        Lam_QKT = jnp.kron(jnp.eye(self.N_alpha), K_tilde_B @ B_Q)
        C_A_inv_B = Lam_QKT @ full_A_dense @ Lam_QKT.T

        # Add additional terms corresponding to beta_kernels in inner D block of outer B block
        full_A2 = jnp.zeros((self.N_alpha, self.N_t, self.N_alpha, self.N_t))
        full_A2 = full_A2.at[:, -self.N_beta:, :, :-self.N_beta].set(-R_D2_K_D_inv_R_A)
        full_A2 = full_A2.at[:, -self.N_beta:, :, -self.N_beta:].set(R_D2_K_D_inv_R_D)

        full_A2_dense = full_A2.reshape(N_total, N_total)
        A2_term = full_A2_dense @ Lam_QKT.T
        C_A_inv_B2 = A2_term + A2_term.T

        C_A_inv_B2_reshape = C_A_inv_B2.reshape(self.N_alpha, self.N_t, self.N_alpha, self.N_t)
        current_D_block = C_A_inv_B2_reshape[:, -self.N_beta:, :, -self.N_beta:]
        C_A_inv_B2_reshape = C_A_inv_B2_reshape.at[:, -self.N_beta:, :, -self.N_beta:].set(current_D_block + R_D2_K_D_inv_R_D2)
        C_A_inv_B2 = C_A_inv_B2_reshape.reshape(N_total, N_total)

        return C_A_inv_B + C_A_inv_B2
    
    
    def kf_B_eval(self, X1, X2):
        K_B = jnp.kron(self.K_transf0_B, self.K_transf[1].evaluate(X1[1], X2[1])) 
        
        for j in range(self.N_beta):
            beta_vec = jnp.zeros(self.N_t)
            beta_vec = beta_vec.at[-self.N_beta:].set(self.J_B[:, j])
            K_B += jnp.kron(self.K_beta_B[j], jnp.outer(beta_vec, beta_vec))

        return K_B
    

    def transform_fn(self, R, transpose = 0):

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


    def matrix_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
    ) -> JAXArray:

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

        if transpose:
            R_prime = self.kf_tilde.matrix_inv_sqrt(R, transpose = 1)
            R_prime = self.inv_transform_fn(R_prime, transpose = 1)
        else:
            R_prime = self.inv_transform_fn(R, transpose = 0)
            R_prime = self.kf_tilde.matrix_inv_sqrt(R_prime, transpose = 0)
        
        return R_prime
       

    def matmul(self, X1, X2, R, **kwargs):

        K_R = tensor_mult(self.Sigma, X1, X2, R, **kwargs)

        for K in self.K_list:
            K_R += tensor_mult(K, X1, X2, R, **kwargs)
        
        return K_R

