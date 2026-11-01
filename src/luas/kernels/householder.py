from typing import Optional, Callable, Tuple, Any, Union
from jax import tree_util
import jax.numpy as jnp

from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
from luas.kronecker_fns import vmap_for_tensors


@tree_util.register_pytree_node_class
class HouseholderProduct:
    def __init__(self, V, transpose = 0):

        self.N = V.shape[0]
        self.N_alpha = V.shape[1]

        self.V = V
        self.transpose = transpose
        self.U = jnp.zeros_like(V)
        
        # Householder vector
        for i in range(self.N_alpha):
            w_i = V[:, i]
            for j in range(i):
                w_i -= 2 * jnp.dot(self.U[:, j], w_i) * self.U[:, j]

            e_i = jnp.zeros(self.N)
            e_i = e_i.at[-self.N_alpha + i].set(1)
            u_i = w_i - e_i
            u_i /= jnp.linalg.norm(u_i)

            self.U = self.U.at[:, i].set(u_i)

    def householder_transform(self, R, transpose = 0):
        R_prime = R.copy()

        if not transpose:
            for i in range(self.N_alpha):
                u_R_prime = self.U[:, -i-1].T @ R_prime
                R_prime -= jnp.outer(2*self.U[:, -i-1], u_R_prime)
        else:
            for i in range(self.N_alpha):
                u_R_prime = self.U[:, i].T @ R_prime
                R_prime -= jnp.outer(2*self.U[:, i], u_R_prime)

        return R_prime
    
    @property
    def T(self):
        return HouseholderProduct(V = self.V, transpose = 1-self.transpose)

    def __matmul__(self, other):
        return self.householder_transform(other, transpose = self.transpose)
    
    def __rmatmul__(self, other):
        house_transf = self.T
        R_prime_transpose = house_transf.__matmul__(other.T)
        return R_prime_transpose.T

    def __repr__(self):
        return f"HouseholderProduct({self.U})"
    
    def to_dense(self):
        return self.__matmul__(jnp.eye(self.N), transpose = self.transpose)
    
    # Required for JAX
    def tree_flatten(self):
        return (self.__repr__(),), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@tree_util.register_pytree_node_class
class HouseholderandDiagonalTransform:
    """Represents H @ D where H = I - 2 * vec @ vec^T is a Householder matrix     
    """
    def __init__(self, vec, diag, i = -1, transpose = 0):
        self.vec = vec
        self.diag = diag
        self.transpose = transpose
        basis_vec = jnp.zeros_like(vec)
        basis_vec = basis_vec.at[i].set(1)

        norm_alpha = jnp.linalg.norm(self.vec)

        alpha_normalised = self.vec/norm_alpha
        u = alpha_normalised - basis_vec
        self.u_hat = u/jnp.linalg.norm(u)
        self.lam = norm_alpha**2 * basis_vec


    def __matmul__(self, other):
        if other.ndim == 1:
            if self.transpose:
                other_prime = other - 2 * jnp.kron(self.u_hat, self.u_hat @ other)
                other_prime = self.diag * other_prime
            else:
                other_prime = self.diag * other
                other_prime -= 2 * jnp.kron(self.u_hat, self.u_hat @ other_prime)
        elif other.ndim == 2:
            if self.transpose:
                other_prime = other - 2 * jnp.outer(self.u_hat, self.u_hat @ other_prime)
                other_prime = jnp.einsum('i,i...->i...', self.diag, other)
            else:
                other_prime = jnp.einsum('i,i...->i...', self.diag, other)
                other_prime -= 2 * jnp.outer(self.u_hat, self.u_hat @ other_prime)
        elif other.ndim > 2:
            return vmap_for_tensors(self.__matmul__)(other)
        else:
            raise Exception("Not implemented")
        
        return other_prime

    def __rmatmul__(self, other):
        if other.ndim == 1:
            if self.transpose:
                other_prime = self.diag * other
                other_prime -= 2 * jnp.kron(other_prime @ self.u_hat, self.u_hat)
            else:
                other_prime = other - 2 * jnp.kron(other @ self.u_hat, self.u_hat)
                other_prime = self.diag * other_prime
        elif other.ndim == 2:
            if self.transpose:
                other_prime = jnp.einsum('i,i...->i...', self.diag, other)
                other_prime -= 2 * jnp.kron(other_prime @ self.u_hat, self.u_hat)
            else:
                other_prime = other - 2 * jnp.kron(other @ self.u_hat, self.u_hat)
                other_prime = jnp.einsum('i,i...->i...', self.diag, other_prime)
        else:
            raise Exception("Not implemented")

    def __repr__(self):
        dense_mat = self.to_dense()
        return f"HouseholderandDiagonalTransform({dense_mat})"

    def solve(self, R):
        inv_mat = HouseholderandDiagonalTransform(vec = self.vec, diag = 1/self.diag, transpose = 1-self.transpose)
        return inv_mat @ R
    
    @property
    def T(self):
        return HouseholderandDiagonalTransform(vec = self.vec, diag = self.diag, transpose = 1-self.transpose)

    def to_dense(self):
        H_mat = jnp.eye(self.u_hat.size) - 2 * jnp.outer(self.u_hat, self.u_hat)
        D = jnp.diag(self.diag)

        if self.transpose:
            self.dense = D @ H_mat
        else:
            self.dense = H_mat @ D

        return self.dense
        
    # Required for JAX
    def tree_flatten(self):
        return (self.__repr__(),), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)



@tree_util.register_pytree_node_class
class HouseholderTransform:
    def __init__(self, vec, i = -1):
        self.vec = vec
        self.T = self
        basis_vec = jnp.zeros_like(vec)
        basis_vec = basis_vec.at[i].set(1)

        norm_alpha = jnp.linalg.norm(self.vec)

        alpha_normalised = self.vec/norm_alpha
        u = alpha_normalised - basis_vec
        self.u_hat = u/jnp.linalg.norm(u)
        self.lam = norm_alpha**2 * basis_vec


    def __matmul__(self, other):
        if other.ndim == 1:
            return other - 2 * jnp.kron(self.u_hat, self.u_hat @ other)
        elif other.ndim == 2:
            return other - 2 * jnp.outer(self.u_hat, self.u_hat @ other)
        elif other.ndim > 2:
            return vmap_for_tensors(self.__matmul__)(other)
        else:
            raise Exception("Not implemented")

    def __rmatmul__(self, other):
        if other.ndim == 1:
            return other - 2 * jnp.kron(other @ self.u_hat, self.u_hat)
        elif other.ndim == 2:
            return other - 2 * jnp.outer(other @ self.u_hat, self.u_hat @ other)
        else:
            raise Exception("Not implemented")

    def __repr__(self):
        dense_mat = self.to_dense()
        return f"HouseholderTransform({dense_mat})"

    def to_dense(self):
        self.dense = jnp.eye(self.u_hat.size) - 2 * jnp.outer(self.u_hat, self.u_hat)
        return self.dense
        
    # Required for JAX
    def tree_flatten(self):
        return (self.__repr__(),), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)



def householder_qr_and_orthogonalise(A, reverse = False):
    
    N_alpha = A.shape[1]

    # Compute the QR decomposition of A (produces orthonormal basis for the column space of A)
    # More numerically stable than the Gram-Schmidt process used in orthonormal_nullspace_gen_unstable
    # V @ J = A, if A is (N_l, N_alpha) then V is (N_l, N_alpha) and J is (N_alpha, N_alpha)
    # V guaranteed orthonormal, V forms first N_alpha columns of Q, J is top N_alpha rows of R, where Q @ R = A
    V, J = jnp.linalg.qr(A, mode = 'reduced')
    
    # Create Householder vectors from V, flipping signs when needed for numerical stability
    U = jnp.zeros_like(A)
    for i in range(N_alpha):
        w_i = V[:, i]

        for j in range(i):
            w_i -= 2 * jnp.dot(U[:, j], w_i) * U[:, j]

        idx = -N_alpha + i if reverse else i

        # sign chosen so u = w + s*e_i never cancels; define sign(0) = +1
        s = jnp.where(w_i[idx] >= 0, 1.0, -1.0)

        # Must flip the sign of corresponding row of J to construct the correct vector from flipped orthogonal vector
        J = J.at[idx, :].multiply(s)

        u_i = w_i.at[idx].add(s)
        u_i /= jnp.linalg.norm(u_i)

        U = U.at[:, i].set(u_i)

    # Create fast H @ R function where H is a product of Householder matrices
    # Equivalent to multiplying by Q where A = Q @ R
    def householder_transform(R, transpose = 0):
        R_prime = R.copy()

        if transpose:
            for i in range(N_alpha):
                u_R_prime = U[:, -i-1].T @ R_prime
                R_prime -= jnp.outer(2*U[:, -i-1], u_R_prime)
        else:
            for i in range(N_alpha):
                u_R_prime = U[:, i].T @ R_prime
                R_prime -= jnp.outer(2*U[:, i], u_R_prime)

        return R_prime

    J = jnp.diag(jnp.diag(J))
    A_ortho = V @ J
    
    return J, U, householder_transform, A_ortho



def householder_qr(A, reverse = False):
    
    N_alpha = A.shape[1]

    # Compute the QR decomposition of A (produces orthonormal basis for the column space of A)
    # More numerically stable than the Gram-Schmidt process used in orthonormal_nullspace_gen_unstable
    # V @ J = A, if A is (N_l, N_alpha) then V is (N_l, N_alpha) and J is (N_alpha, N_alpha)
    # V guaranteed orthonormal, V forms first N_alpha columns of Q, J is top N_alpha rows of R, where Q @ R = A
    V, J = jnp.linalg.qr(A, mode = 'reduced')
    
    # Create Householder vectors from V, flipping signs when needed for numerical stability
    U = jnp.zeros_like(A)
    for i in range(N_alpha):
        w_i = V[:, i]

        for j in range(i):
            w_i -= 2 * jnp.dot(U[:, j], w_i) * U[:, j]

        idx = -N_alpha + i if reverse else i

        # sign chosen so u = w + s*e_i never cancels; define sign(0) = +1
        s = jnp.where(w_i[idx] >= 0, 1.0, -1.0)

        # Must flip the sign of corresponding row of J to construct the correct vector from flipped orthogonal vector
        J = J.at[idx, :].multiply(s)

        u_i = w_i.at[idx].add(s)
        u_i /= jnp.linalg.norm(u_i)

        U = U.at[:, i].set(u_i)

    # Create fast H @ R function where H is a product of Householder matrices
    # Equivalent to multiplying by Q where A = Q @ R
    def householder_transform(R, transpose = 0):
        R_prime = R.copy()

        if transpose:
            for i in range(N_alpha):
                u_R_prime = U[:, -i-1].T @ R_prime
                R_prime -= jnp.outer(2*U[:, -i-1], u_R_prime)
        else:
            for i in range(N_alpha):
                u_R_prime = U[:, i].T @ R_prime
                R_prime -= jnp.outer(2*U[:, i], u_R_prime)

        return R_prime
    
    return J, U, householder_transform
