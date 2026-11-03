import jax.numpy as jnp
from typing import Callable, Tuple, Union, Any, Optional

from luas.kernels.covtype import Outer, Exp, GeneralQuasisep, CovType, Identity, ScaledIdentity, Diagonal, General
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
from luas.kronecker_fns import tensor_mult, vmap_for_tensors, cyclic_transpose
from luas.kernel_interface import read_K_list_2D
from luas.kernels.householder import householder_qr
from luas.kernels.BlockKernel import Block2x2Kernel, BlockKernel
from luas.kernels.GeneralKernel import GeneralKernel
from luas.kernels.LuasKernelND import LuasKernelND

__all__ = [
    "LuasPlusMultiLowRankKernel",
]

class LuasPlusMultiLowRankKernel(CovType):
    r"""GP optimisation for measurement covariance plus a full-rank 2D kernel term
    plus low-rank kernel terms along one dimension.

    With the low-rank vectors along time (``fast_dim = 1``), this handles kernel
    functions of the form

    .. math::

        k(\lambda, \lambda', t, t') = s_\lambda(\lambda, \lambda')\, s_t(t, t')
        + k_{\lambda; \mathrm{2D}}(\lambda, \lambda')\, k_{t; \mathrm{2D}}(t, t')
        + \sum_{j=1}^{N_\beta} k_{\lambda; j}(\lambda, \lambda')\, \beta_j(t)\beta_j(t'),

    which on a rectangular grid gives the covariance matrix

    .. math::

        \mathbf{K} = \mathbf{\Sigma}_\lambda \otimes \mathbf{\Sigma}_t
        + \mathbf{K}_{\lambda; \mathrm{2D}} \otimes \mathbf{K}_{t; \mathrm{2D}}
        + \sum_{j=1}^{N_\beta} \mathbf{K}_{\lambda; j} \otimes
        \boldsymbol{\beta}_{t; j} \boldsymbol{\beta}_{t; j}^T.

    With ``fast_dim = 0`` the low-rank vectors are instead along wavelength, i.e.
    terms :math:`\boldsymbol{\alpha}_{\lambda; i}\boldsymbol{\alpha}_{\lambda; i}^T
    \otimes \mathbf{K}_{t; i}`. Unlike :class:`luas.MultiLowRankKernel`, here
    ``fast_dim`` is the dimension **holding the low-rank vectors**. This is also
    the dimension where quasiseparable kernels can be exploited, giving linear
    scaling along it if :math:`\mathbf{\Sigma}` and :math:`\mathbf{K}_\mathrm{2D}` are
    quasiseparable or diagonal in that dimension.

    For example, a 2D correlated noise term plus a detector settling ramp with a
    fixed shape in time and a wavelength-correlated amplitude.

    **Method.** In the other (slow) dimension, :math:`\mathbf{\Sigma}` and
    :math:`\mathbf{K}_\mathrm{2D}` are simultaneously diagonalised as in
    :class:`luas.LuasKernel`. In ``fast_dim``, the low-rank vectors are
    transformed with a Householder QL decomposition, which keeps quasiseparable
    kernels quasiseparable (each transformation raises the rank by 2). The
    transformed matrix is decomposed as a 2x2 block matrix. The large A block is
    handled either as independent 1D GPs like :class:`luas.LuasLasrachKernel`
    (``eigen_both = False``) or with eigendecomposition in both dimensions like
    :class:`luas.LuasKernel` (``eigen_both = True``). The
    :math:`(N_\beta N_\lambda, N_\beta N_\lambda)` block
    :math:`\mathbf{D} - \mathbf{C}\mathbf{A}^{-1}\mathbf{B}` is decomposed with a
    general Cholesky decomposition, which is usually the bottleneck.

    **Runtime scaling** (low-rank vectors along :math:`t`, quasiseparable
    :math:`k_{t; \mathrm{2D}}` of rank :math:`J_{K_t}`, diagonal :math:`s_t`,
    ``eigen_both = False``):

    .. math::

        \mathcal{O}\left((1 + N_\beta^3) N_\lambda^3 + N_\lambda^2 N_t
        + N_\lambda N_t (J_{K_t} + 2N_\beta)^2\right),

    i.e. linear in :math:`N_t`. With ``eigen_both = True`` no quasiseparable kernels
    are needed but the scaling is cubic in both dimensions,
    :math:`\mathcal{O}((1 + N_\beta^3) N_\lambda^3 + N_t^3 + N_\lambda N_t
    (N_\lambda + N_t + N_\beta^2))`.

    **Selection.** :class:`luas.GP` selects this optimisation automatically when the
    kernel function returns the noise term, exactly one full-rank 2D term, and
    low-rank terms along only one dimension. ``fast_dim`` is set to the dimension
    of the low-rank vectors, and ``eigen_both = True`` is used unless all kernels
    along the longer dimension are quasiseparable or diagonal.

    Warning:
        The noise term must be positive definite and be the **first** tuple returned
        by the kernel function. Exactly one full-rank 2D term is required, all
        low-rank terms must have their vectors along ``fast_dim``, and the number of
        low-rank terms must be less than the length of ``fast_dim``.

    Example:
        Measurement covariance, a 2D kernel term and a wavelength-correlated
        settling ramp (from the ``luas`` tutorial):

        .. code-block:: python

            def kf(p, X):
                s_l = luas.kernels.quasisep.Noise(jnp.exp(p["log_sigma"]))
                s_l += jnp.exp(2*p["log_h_readout"][0]) * luas.kernels.quasisep.Exp(jnp.exp(p["log_l_readout"][0]))
                s_t = luas.kernels.quasisep.KroneckerDelta()

                # Full-rank 2D term, keep the time kernel quasiseparable for linear scaling in time
                k_l = jnp.exp(2*p["log_h"][0]) * luas.kernels.Matern52(jnp.exp(p["log_l_l"][0]))
                k_t = luas.kernels.quasisep.Matern32(jnp.exp(p["log_l_t"][0]))

                # beta term: fixed ramp shape along time, wavelength-correlated amplitude
                k_l1 = jnp.exp(2*p["log_h_ramp"][0]) * luas.kernels.quasisep.Matern32(jnp.exp(p["log_l_l_ramp"][0]))
                settle_vec = jnp.exp(-(X[1] - X[1][0])/jnp.exp(p["log_l_t_ramp"][0]))
                beta_t1 = luas.kernels.lowrank.Amplitude(settle_vec)

                return (s_l, s_t), (k_l, k_t), (k_l1, beta_t1)

            gp = luas.GP(p, (x_l, x_t), kf = kf)  # Uses LuasPlusMultiLowRankKernel with fast_dim = 1

    Args:
        Sigma (tuple[CovType, CovType]): The covariance objects
            :math:`(\mathbf{\Sigma}_\lambda, \mathbf{\Sigma}_t)` for the noise term.
        *K_list (tuple[CovType, CovType]): Exactly one full-rank 2D term plus one or
            more low-rank terms, each as a tuple with one covariance object per
            dimension. Low-rank terms must have a rank-1 outer product (e.g.
            ``luas.kernels.lowrank.Amplitude``) along ``fast_dim``.
        fast_dim (int, optional): The dimension holding the low-rank vectors, which
            is also the dimension that can exploit quasiseparable kernels. Defaults
            to ``1``.
        never_reduce_dim (bool, optional): Currently unused.
        use_stored_values (bool, optional): Currently unused.
        transform (bool, optional): Internal. If ``False``, the inputs are assumed
            to be already transformed (as done when this class is used inside
            :class:`luas.ExtendedLuasKernel`), and ``transform_fn`` and
            ``inv_transform_fn`` must be given. Defaults to ``True``.
        transform_fn (Callable, optional): Internal. Replaces the default transform
            applied to the residuals by ``matrix_sqrt``.
        inv_transform_fn (Callable, optional): Internal. Replaces the default
            inverse transform applied to the residuals by ``matrix_inv_sqrt``.
        eigen_both (bool, optional): If ``True``, decompose the A block using
            eigendecomposition in both dimensions (no restrictions on kernels, cubic
            scaling in both dimensions). If ``False``, decompose it as independent 1D
            GPs along ``fast_dim`` (linear in that dimension when its kernels are
            quasiseparable). Defaults to ``False``.
        use_pmap (bool, optional): Experimental, and not currently working. Intended
            (when ``eigen_both = False``) to decompose the independent blocks across
            devices with ``jax.pmap`` instead of ``jax.vmap``. Defaults to ``False``.
        **kwargs: Ignored, accepted so that all Kronecker kernels share a common
            constructor interface.

    See Also:
        :class:`luas.LuasKernel`, :class:`luas.LuasLasrachKernel` for the same model
        without low-rank terms.
        :class:`luas.ExtendedLuasKernel` for low-rank terms along both dimensions.
    """

    def __init__(
        self,
        Sigma,
        *K_list,
        fast_dim = 1,
        use_stored_values: Optional[bool] = False,
        transform = True,
        transform_fn = None,
        inv_transform_fn = None,
        eigen_both = False,
        use_pmap = False,
        **kwargs,
    ):
        
        self.Sigma = Sigma[0], Sigma[1]
        self.K_list = K_list
        self.fast_dim = fast_dim
        self.eigen_both = eigen_both
        self.transform = transform
        self.use_pmap = use_pmap
        self.opt_name = "LuasPlusMultiLowRankKernel"

        if transform_fn is not None:
            self.transform_fn = transform_fn
        else:
            self.transform_fn = self.default_transform_fn

        if inv_transform_fn is not None:
            self.inv_transform_fn = inv_transform_fn
        else:
            self.inv_transform_fn = self.default_inv_transform_fn
        
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            raise NotImplementedError("Use of pre-calculated stored values not yet implemented.")
        else:
            self.decompose = self.decompose_no_stored_values


    def _rotate_to_fast_dim_wrapper(self, fn, fast_dim):

        def wrapped_fn(R, **kwargs):
            if fast_dim == 1:
                R_prime = R.T
            else:
                R_prime = R.copy()

            R_prime = fn(R_prime, **kwargs)
            if fast_dim == 1:
                R_prime = R_prime.T
            return R_prime

        return wrapped_fn
    

    def _calc_C_A_inv_B(self, kf_A_matrix_inv_sqrt, lam_K_tilde, V_K_B, V_Sigma_B):

        # self.V_K_B of shape (N_fast - N_alpha, N_alpha)
        # K_B_nonzero of shape (N_slow, N_alpha, N_fast - N_alpha)
        K_B_nonzero = jnp.kron(lam_K_tilde.reshape((lam_K_tilde.size, 1, 1)), V_K_B.T)
        K_B_nonzero += jnp.kron(jnp.ones((lam_K_tilde.size, 1, 1)), V_Sigma_B.T)

        # Reshape to (N_alpha, N_fast - N_alpha, N_slow) for easy vmap
        K_B_nonzero = cyclic_transpose(K_B_nonzero, 1)

        # L_A_inv_K_B should still be (N_alpha, N_fast - N_alpha, N_slow)
        A_inv_sqrt_fn = vmap_for_tensors(kf_A_matrix_inv_sqrt)
        L_A_inv_K_B = A_inv_sqrt_fn(K_B_nonzero, transpose = 0)

        # C_A_inv_B will be of shape (N_alpha, N_alpha, N_slow)
        C_A_inv_B = jnp.einsum('mij,nij->mnj', L_A_inv_K_B, L_A_inv_K_B)
        C_A_inv_B = cyclic_transpose(C_A_inv_B, -1)

        n = jnp.arange(self.N_slow)
        N_total = self.N_slow * self.N_alpha

        C_A_inv_B_dense = jnp.zeros((N_total, N_total))
        C_A_inv_B_dense = C_A_inv_B_dense.reshape(self.N_alpha, self.N_slow, self.N_alpha, self.N_slow).at[:, n, :, n].add(C_A_inv_B).reshape(N_total, N_total)

        return C_A_inv_B_dense
    

    def decompose_no_stored_values(
        self,
        X: Tuple[JAXArray],
        stored_values: Optional[PyTree] = {},
        full = True,
        idx = (None, None),
    ) -> PyTree:

        self.N_slow = X[1-self.fast_dim].shape[-1]
        self.N_fast = X[self.fast_dim].shape[-1]

        dense_kron, *low_rank_terms = read_K_list_2D(self.K_list, X)
        alpha_mat, kernel_list = low_rank_terms[self.fast_dim]
        self.N_alpha = len(self.K_list) - 1

        # Required for this optimisation to work
        if self.N_alpha >= X[self.fast_dim].shape[-1]:
            raise Exception(r"Must have fewer low-rank terms than size of dimension these terms are fixed vectors in.")

        # Check low rank terms have correct number of terms and dimensions
        assert alpha_mat.shape[-1] == len(kernel_list) == self.N_alpha

        if self.transform:
            assert alpha_mat.shape[0] == self.N_fast
        else:
            assert alpha_mat.shape[0] == self.N_alpha

        # Check only have low rank terms in one dimension
        assert low_rank_terms[1-self.fast_dim] is None


        ######### Do transforms in slow_dim #########

        if self.transform:
            # Decompose slow Sigma mat and add its contribution to the log determinant
            self.Sigma_slow, stored_values_Sigma_slow = self.Sigma[1-self.fast_dim].decompose(X[1-self.fast_dim])
            stored_values["logdet"] = self.N_fast*stored_values_Sigma_slow["logdet"]
            
            dense_kron[1-self.fast_dim] = self.Sigma_slow.inv_sqrt_transform(dense_kron[1-self.fast_dim],
                                                                            X[1-self.fast_dim])

            # Take eigendecomp of slow_dim K_tilde
            self.lam_K_tilde, self.Q_K_slow_tilde = dense_kron[1-self.fast_dim].eigendecomp(X[1-self.fast_dim],
                                                                                            full = full,
                                                                                            idx = idx[1-self.fast_dim])

            # Transform all kernels in kernel_list with simultaneous diagonalisation transforms in slow_dim
            for i in range(self.N_alpha):
                kernel_list[i] = self.Sigma_slow.inv_sqrt_transform(kernel_list[i], X[1-self.fast_dim])
                kernel_list[i] = kernel_list[i].general_transf(self.Q_K_slow_tilde.T, X[1-self.fast_dim])


            ######### Do transforms in fast_dim #########
            # Get Householder transform of all Outer covariances
            stored_values["J_A"], stored_values["U_A"], self.householder_transform = householder_qr(alpha_mat,
                                                                                                            reverse = True)
            
            # Now must transform fast covariance matrices with Householder transform i.e. H K H.T, H Sigma H.T
            K_fast_tilde = dense_kron[self.fast_dim]
            Sigma_fast_tilde = self.Sigma[self.fast_dim]
            for i in range(self.N_alpha):
                K_fast_tilde = K_fast_tilde.householder_transform(X[self.fast_dim], stored_values["U_A"][:, i])
                Sigma_fast_tilde = Sigma_fast_tilde.householder_transform(X[self.fast_dim], stored_values["U_A"][:, i])
        
        else:
            # Pre-transformed, e.g. called from LuasPlusMultiTermBothDimKernel
            # Should have inputs (Identity, Sigma_fast_tilde), (lam_K_tilde, K_fast_tilde), *(J_alpha_i, kernel_list[i])
            stored_values["logdet"] = 0.
            stored_values["J_A"] = alpha_mat
            self.lam_K_tilde = dense_kron[1-self.fast_dim].diag
            Sigma_fast_tilde = self.Sigma[self.fast_dim] # Should just be the identity
            K_fast_tilde = dense_kron[self.fast_dim]
            

        ######### Build A Block #########
        # Assumes data residuals pre-rotated to ensure fast dim is second dim
        K_A_kernel = ((Sigma_fast_tilde, Identity()),
                    (K_fast_tilde, Diagonal(diag = self.lam_K_tilde))
                    )

        # A block consists of a Block Kernel which could be handled through LuasLasrachKernel or LuasKernelND
        if self.eigen_both:
            kf_A = LuasKernelND(*K_A_kernel)
                
            kf_A, kf_A_stored_values = kf_A.decompose((X[self.fast_dim][:-self.N_alpha], X[1-self.fast_dim]),
                                                            full = False, idx = (jnp.arange(self.N_fast - self.N_alpha), jnp.arange(self.N_slow))
            )
            # Use for ExtendedLuasKernel
            self.lam_fast_A, self.Q_fast_A = kf_A_stored_values["lam_0"], kf_A_stored_values["Q_0"]
        else:
            kf_A = BlockKernel(*K_A_kernel, non_block_dim_size = self.N_slow, block_dim=0, use_pmap = self.use_pmap)
            
            kf_A, kf_A_stored_values = kf_A.decompose((X[self.fast_dim][:-self.N_alpha], X[1-self.fast_dim]),
                                                            full = False, 
                                                            idx = (jnp.arange(self.N_fast - self.N_alpha), jnp.arange(self.N_slow)))



        ######### Build B and B.T block #########
        # Evaluate these fast covariance matrices within the B, B.T and D blocks
        V_K = K_fast_tilde.evaluate(X[self.fast_dim], X[self.fast_dim][-self.N_alpha:], full = False,
                                    row_idx = jnp.arange(self.N_fast),
                                    col_idx = jnp.arange(self.N_fast-self.N_alpha, self.N_fast))
        V_Sigma = Sigma_fast_tilde.evaluate(X[self.fast_dim], X[self.fast_dim][-self.N_alpha:], full = False,
                                    row_idx = jnp.arange(self.N_fast),
                                    col_idx = jnp.arange(self.N_fast-self.N_alpha, self.N_fast))
        
        # Split these matrices between the B and D blocks (D block is only the last N_alpha rows and columns)
        self.V_K_B, self.V_K_D = V_K[:-self.N_alpha, :], V_K[-self.N_alpha:, :]
        self.V_Sigma_B, self.V_Sigma_D = V_Sigma[:-self.N_alpha, :], V_Sigma[-self.N_alpha:, :]
        

        ######### Build D block #########
        K_alpha = ()
        for i in range(self.N_alpha):
            K_alpha += ((Outer(stored_values["J_A"][:, i]), kernel_list[i]),)
        
        K_D_dense = jnp.kron(self.V_Sigma_D, jnp.eye(self.N_slow))
        K_D_dense += jnp.kron(self.V_K_D, jnp.diag(self.lam_K_tilde))
        
        # Calculates B.T @ A_inv @ B for inverse of D block
        self.C_A_inv_B = self._calc_C_A_inv_B(kf_A.matrix_inv_sqrt, self.lam_K_tilde, self.V_K_B, self.V_Sigma_B)

        kf_D_CAB = GeneralKernel(*K_alpha, dense_mat = K_D_dense - self.C_A_inv_B)
        kf_D = GeneralKernel(*K_alpha, dense_mat = K_D_dense)


        ######### Define full 2x2 Block kernel and decompose #########
        self.kf_tilde = Block2x2Kernel(kf_A = kf_A, kf_B = self.K_B_matmul, kf_D_CAB = kf_D_CAB, kf_D = kf_D,
                                       dim_split = 0, split_loc = self.N_fast-self.N_alpha,
                                       split_idx = True, kf_B_eval = self.kf_B_eval, A_full = False, D_full = True)
        
        self.kf_tilde.matrix_sqrt = self._rotate_to_fast_dim_wrapper(self.kf_tilde.matrix_sqrt, self.fast_dim)
        self.kf_tilde.matrix_inv_sqrt = self._rotate_to_fast_dim_wrapper(self.kf_tilde.matrix_inv_sqrt, self.fast_dim)
        
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose((X[self.fast_dim], X[1-self.fast_dim]),
                                                                                  full = False)

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
            R_prime = self.V_K_B.T @ R 
            R_prime = R_prime * self.lam_K_tilde
            R_prime += self.V_Sigma_B.T @ R
        else:
            R_prime = self.V_K_B @ R 
            R_prime = R_prime * self.lam_K_tilde
            R_prime += self.V_Sigma_B @ R
        
        return R_prime
    
    def kf_B_eval(self, X1, X2):
        return jnp.kron(self.V_K_B, jnp.diag(self.lam_K_tilde)) + jnp.kron(self.V_Sigma_B, jnp.eye(self.N_slow))
    

    def default_transform_fn(self, R, transpose = 0):

        if self.fast_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()
        
        if transpose:
            R_prime = self.Sigma_slow.matrix_sqrt(R_prime, transpose = 1)
            R_prime = self.Q_K_slow_tilde.T @ R_prime
        else:
            R_prime = self.Q_K_slow_tilde @ R_prime
            R_prime = self.Sigma_slow.matrix_sqrt(R_prime, transpose = 0)

        R_prime = R_prime.T
        R_prime = self.householder_transform(R_prime, transpose = 1-transpose)

        if self.fast_dim == 1:
            R_prime = R_prime.T
        
        return R_prime


    def default_inv_transform_fn(self, R, transpose = 0):

        if self.fast_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()
        
        if transpose:
            R_prime = self.Q_K_slow_tilde @ R_prime
            R_prime = self.Sigma_slow.matrix_inv_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.Sigma_slow.matrix_inv_sqrt(R_prime, transpose = 0)
            R_prime = self.Q_K_slow_tilde.T @ R_prime

        R_prime = R_prime.T
        R_prime = self.householder_transform(R_prime, transpose = transpose)

        if self.fast_dim == 1:
            R_prime = R_prime.T
        
        return R_prime


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
