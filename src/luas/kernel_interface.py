import jax.numpy as jnp
from luas.kernels.covtype import Outer, CovType

__all__ = [
    "read_K_list_2D",
]

def read_K_list_2D(K_list, X):

    # Initialise for loop reading K_list
    dense_kron = None
    alpha_list = [] # np.zeros((X[self.fast_dim].shape[-1], self.N_alpha))
    beta_list = []

    low_rank_kernels_dim0 = []
    low_rank_kernels_dim1 = []
    for K_i in K_list:
        K_0 = K_i[0]
        K_1 = K_i[1]

        # rank_0 = K_0.rank(X[0])
        # rank_1 = K_1.rank(X[1])

        if isinstance(K_0, Outer) and isinstance(K_1, CovType):
            K_0, _ = K_0.decompose(X[0])
            alpha_list.append(K_0.alpha)
            low_rank_kernels_dim1.append(K_1)

        elif isinstance(K_0, CovType) and isinstance(K_1, Outer):
            K_1, _ = K_1.decompose(X[1])
            beta_list.append(K_1.alpha)
            low_rank_kernels_dim0.append(K_0)

        elif isinstance(K_0, CovType) and isinstance(K_1, CovType):
            if dense_kron is None:
                dense_kron = [K_0, K_1]
            else:
                raise Exception("luas only implements an optimisation for a maximum of two dense kernels in the Kronecker product."\
                                " Options are to switch to low-rank kernel terms in one of the dimensions beyond two dense terms"\
                                " or force use_kernel = luas.GeneralKernel which will build the full covariance matrix and decompose it"\
                                " (can be very slow and memory intensive for large datasets).")

        else:
            raise Exception("All kernel terms must be valid CovType objects")
        
    if alpha_list:
        alpha_mat = jnp.stack(alpha_list, axis = 1)
        alpha_terms = (alpha_mat, low_rank_kernels_dim1)
    else:
        alpha_terms = None

    if beta_list:
        beta_mat = jnp.stack(beta_list, axis = 1)
        beta_terms = (beta_mat, low_rank_kernels_dim0)
    else:
        beta_terms = None

    return dense_kron, alpha_terms, beta_terms

