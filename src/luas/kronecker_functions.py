import jax.numpy as jnp

def make_vec(R):
    return R.ravel("C")

def make_mat(r, N_l, N_t):
    return r.reshape((N_l, N_t))

def kron_prod(A, B, R):
    return A @ R @ B.T


def kronecker_inv_vec(R, eigen_dict):
    
    b = kron_prod(eigen_dict["W_l"].T, eigen_dict["W_t"].T, R)
    b = jnp.multiply(eigen_dict["D_inv"], b)
    b = kron_prod(eigen_dict["W_l"], eigen_dict["W_t"], b)

    return b
