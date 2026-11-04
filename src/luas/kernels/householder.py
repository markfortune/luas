from typing import Optional, Callable, Tuple, Any, Union
from jax import tree_util
import jax.numpy as jnp

from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
from luas.kronecker_fns import vmap_for_tensors


@tree_util.register_pytree_node_class
class HouseholderProduct:
    r"""The orthogonal matrix :math:`\mathbf{H} = \mathbf{H}_1\mathbf{H}_2\cdots\mathbf{H}_k`, a product of Householder
    reflections built from the columns of ``V``, which reflect the (orthonormal) columns of ``V`` onto the last
    :math:`k` standard basis vectors. Stored as the Householder vectors, so multiplying by it costs
    :math:`\mathcal{O}(Nk)`.

    Args:
        V (JAXArray): Matrix of shape ``(N, k)`` with orthonormal columns.
        transpose (int, optional): Whether the object represents :math:`\mathbf{H}^T`. Defaults to 0.

    """
    def __init__(self, V, transpose = 0):
        r"""Creates the object, see the class docstring for the arguments."""

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
        r"""Multiplies ``R`` by :math:`\mathbf{H}` (or :math:`\mathbf{H}^T` if ``transpose = 1``)."""
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
        r"""The transpose, as a ``HouseholderProduct``."""
        return HouseholderProduct(V = self.V, transpose = 1-self.transpose)

    def __matmul__(self, other):
        r"""Multiplies ``other`` by the matrix, ``self @ other``."""
        return self.householder_transform(other, transpose = self.transpose)
    
    def __rmatmul__(self, other):
        r"""Multiplies the matrix by ``other`` from the left, ``other @ self``."""
        house_transf = self.T
        R_prime_transpose = house_transf.__matmul__(other.T)
        return R_prime_transpose.T

    def __repr__(self):
        r"""Returns a string representation showing the Householder vectors."""
        return f"HouseholderProduct({self.U})"
    
    def to_dense(self):
        r"""Builds the matrix as a dense array."""
        return self.__matmul__(jnp.eye(self.N))
    
    # Required for JAX to be able to JIT/vmap this class
    def tree_flatten(self):
        r"""Flattens the object for ``JAX``, so it can be passed through ``jax.jit`` and ``jax.vmap``."""
        children = (self.V, self.U)   # arrays JAX can trace
        aux_data = (self.transpose,)                      # static Python values
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        r"""Rebuilds the object from its flattened form without re-running ``__init__``."""
        obj = object.__new__(cls)                                # skips __init__
        obj.V, obj.U = children
        obj.transpose = aux_data[0]
        obj.N, obj.N_alpha = obj.V.shape[0], obj.V.shape[1]
        return obj


@tree_util.register_pytree_node_class
class HouseholderandDiagonalTransform:
    r"""Represents :math:`\mathbf{H}\mathbf{D}` (or :math:`\mathbf{D}\mathbf{H}` if ``transpose = 1``), where
    :math:`\mathbf{D}` is the diagonal matrix of ``diag`` and :math:`\mathbf{H} = \mathbf{I} - 2\hat{u}\hat{u}^T` is
    the Householder reflection which maps :math:`\vec{v}/|\vec{v}|` onto the ``i``-th standard basis vector.
    Used as the matrix square root of ``OuterPlusScaledIdentity``.

    Args:
        vec (JAXArray): The vector :math:`\vec{v}` defining the reflection.
        diag (JAXArray): The diagonal of :math:`\mathbf{D}`.
        i (int, optional): Index of the basis vector to map onto. Defaults to -1 (the last).
        transpose (int, optional): Whether the object represents the transpose. Defaults to 0.

    """
    def __init__(self, vec, diag, i = -1, transpose = 0):
        r"""Creates the object, see the class docstring for the arguments."""
        self.vec = vec
        self.diag = diag
        self.transpose = transpose
        self.i = i
        basis_vec = jnp.zeros_like(vec)
        basis_vec = basis_vec.at[self.i].set(1)

        norm_alpha = jnp.linalg.norm(self.vec)

        alpha_normalised = self.vec/norm_alpha
        u = alpha_normalised - basis_vec
        self.u_hat = u/jnp.linalg.norm(u)
        self.lam = norm_alpha**2 * basis_vec


    def __matmul__(self, other):
        r"""Multiplies ``other`` by the matrix, ``self @ other``. Arrays with more than 2 dimensions are handled with ``jax.vmap``."""
        if other.ndim == 1:
            if self.transpose:
                other_prime = other - 2 * jnp.kron(self.u_hat, self.u_hat @ other)
                other_prime = self.diag * other_prime
            else:
                other_prime = self.diag * other
                other_prime -= 2 * jnp.kron(self.u_hat, self.u_hat @ other_prime)
        elif other.ndim == 2:
            if self.transpose:
                other_prime = other - 2 * jnp.outer(self.u_hat, self.u_hat @ other)
                other_prime = jnp.einsum('i,i...->i...', self.diag, other_prime)
            else:
                other_prime = jnp.einsum('i,i...->i...', self.diag, other)
                other_prime -= 2 * jnp.outer(self.u_hat, self.u_hat @ other_prime)
        elif other.ndim > 2:
            return vmap_for_tensors(self.__matmul__)(other)
        else:
            raise NotImplementedError(f"Matrix multiplication not implemented for tensors with ndim {other.ndim}")

        return other_prime

    def __rmatmul__(self, other):
        r"""Multiplies the matrix by ``other`` from the left, ``other @ self``."""
        if other.ndim == 1:
            if self.transpose:
                other_prime = self.diag * other
                other_prime -= 2 * jnp.kron(other_prime @ self.u_hat, self.u_hat)
            else:
                other_prime = other - 2 * jnp.kron(other @ self.u_hat, self.u_hat)
                other_prime = self.diag * other_prime
        elif other.ndim == 2:
            if self.transpose:
                other_prime = other * self.diag
                other_prime -= 2 * jnp.outer(other_prime @ self.u_hat, self.u_hat)
            else:
                other_prime = other - 2 * jnp.outer(other @ self.u_hat, self.u_hat)
                other_prime = other_prime * self.diag
        elif other.ndim > 2:
            return vmap_for_tensors(self.__rmatmul__)(other)
        else:
            raise NotImplementedError(f"Matrix multiplication not implemented for tensors with ndim {other.ndim}")

        return other_prime

    def __repr__(self):
        r"""Returns a string representation showing the shapes of ``vec`` and ``diag``."""
        return f"HouseholderandDiagonalTransform({self.vec.shape}, {self.diag.shape})"

    def solve(self, R):
        r"""Multiplies ``R`` by the inverse of the matrix, using :math:`(\mathbf{H}\mathbf{D})^{-1} = \mathbf{D}^{-1}\mathbf{H}`."""
        inv_mat = HouseholderandDiagonalTransform(vec = self.vec, diag = 1/self.diag, transpose = 1-self.transpose)
        return inv_mat @ R
    
    @property
    def T(self):
        r"""The transpose, as a ``HouseholderandDiagonalTransform``."""
        return HouseholderandDiagonalTransform(vec = self.vec, diag = self.diag, transpose = 1-self.transpose)

    def to_dense(self):
        r"""Builds the matrix as a dense array."""
        H_mat = jnp.eye(self.u_hat.size) - 2 * jnp.outer(self.u_hat, self.u_hat)
        D = jnp.diag(self.diag)

        if self.transpose:
            self.dense = D @ H_mat
        else:
            self.dense = H_mat @ D

        return self.dense


    # Required for JAX to be able to JIT/vmap this class
    def tree_flatten(self):
        r"""Flattens the object for ``JAX``, so it can be passed through ``jax.jit`` and ``jax.vmap``."""
        children = (self.vec, self.diag, self.u_hat, self.lam)   # arrays JAX can trace
        aux_data = (self.i, self.transpose)                      # static Python values
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        r"""Rebuilds the object from its flattened form without re-running ``__init__``."""
        obj = object.__new__(cls)                                # skips __init__
        obj.vec, obj.diag, obj.u_hat, obj.lam = children
        obj.i, obj.transpose = aux_data
        return obj



@tree_util.register_pytree_node_class
class HouseholderTransform:
    r"""The Householder reflection :math:`\mathbf{H} = \mathbf{I} - 2\hat{u}\hat{u}^T` which maps
    :math:`\vec{v}/|\vec{v}|` onto the ``i``-th standard basis vector.

    This gives the eigendecomposition of the rank-one matrix :math:`\vec{v}\vec{v}^T`: the eigenvectors are the
    columns of :math:`\mathbf{H}`, and the eigenvalues are stored in ``lam`` (:math:`|\vec{v}|^2` at index ``i``,
    zero elsewhere). :math:`\mathbf{H}` is symmetric, so ``T`` is the object itself.

    Args:
        vec (JAXArray): The vector :math:`\vec{v}`.
        i (int, optional): Index of the basis vector to map onto. Defaults to -1 (the last).

    """
    def __init__(self, vec, i = -1):
        r"""Creates the object, see the class docstring for the arguments."""
        self.vec = vec
        self.T = self
        self.i = i
        basis_vec = jnp.zeros_like(vec)
        basis_vec = basis_vec.at[self.i].set(1)

        norm_alpha = jnp.linalg.norm(self.vec)

        alpha_normalised = self.vec/norm_alpha
        u = alpha_normalised - basis_vec
        self.u_hat = u/jnp.linalg.norm(u)
        self.lam = norm_alpha**2 * basis_vec


    def __matmul__(self, other):
        r"""Multiplies ``other`` by the reflection, ``self @ other``. Arrays with more than 2 dimensions are handled with ``jax.vmap``."""
        if other.ndim == 1:
            return other - 2 * jnp.kron(self.u_hat, self.u_hat @ other)
        elif other.ndim == 2:
            return other - 2 * jnp.outer(self.u_hat, self.u_hat @ other)
        elif other.ndim > 2:
            return vmap_for_tensors(self.__matmul__)(other)
        else:
            raise NotImplementedError(f"Matrix multiplication not implemented for tensors with ndim {other.ndim}")

    def __rmatmul__(self, other):
        r"""Multiplies the reflection by ``other`` from the left, ``other @ self``."""
        if other.ndim == 1:
            return other - 2 * jnp.kron(other @ self.u_hat, self.u_hat)
        elif other.ndim == 2:
            return other - 2 * jnp.outer(other @ self.u_hat, self.u_hat)
        elif other.ndim > 2:
            return vmap_for_tensors(self.__rmatmul__)(other)
        else:
            raise NotImplementedError(f"Matrix multiplication not implemented for tensors with ndim {other.ndim}")

    def __repr__(self):
        r"""Returns a string representation showing the shape of ``vec``."""
        return f"HouseholderTransform({self.vec.shape})"

    def to_dense(self):
        r"""Builds the reflection as a dense array."""
        self.dense = jnp.eye(self.u_hat.size) - 2 * jnp.outer(self.u_hat, self.u_hat)
        return self.dense
        
    # Required for JAX to be able to JIT/vmap this class
    def tree_flatten(self):
        r"""Flattens the object for ``JAX``, so it can be passed through ``jax.jit`` and ``jax.vmap``."""
        children = (self.vec, self.u_hat, self.lam)   # arrays JAX can trace
        aux_data = (self.i,)            # static Python values
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        r"""Rebuilds the object from its flattened form without re-running ``__init__``."""
        obj = object.__new__(cls)                # skips __init__
        obj.vec, obj.u_hat, obj.lam = children
        obj.i = aux_data[0]
        obj.T = obj
        return obj



def householder_qr_and_orthogonalise(A, reverse = False):
    r"""QR decomposition of ``A`` with :math:`\mathbf{Q}` stored as a product of Householder reflections, which also
    removes the non-orthogonal part of the columns of ``A``.

    Args:
        A (JAXArray): Matrix of shape ``(N, k)`` with :math:`k \leq N`.
        reverse (bool, optional): Whether the reflections map onto the last :math:`k` basis vectors rather than
            the first :math:`k`. Defaults to ``False``.

    Returns:
        (JAXArray, JAXArray, Callable, JAXArray): A tuple of four elements: the diagonal of the
        :math:`(k, k)` R factor as a diagonal matrix; the Householder vectors, of shape ``(N, k)``; a function
        ``householder_transform(R, transpose = 0)`` which multiplies by the full :math:`(N, N)` matrix
        :math:`\mathbf{Q}` (or :math:`\mathbf{Q}^T`); and ``A`` with each column's components along the earlier
        columns removed.

    """
    
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
    r"""QR decomposition of ``A`` with :math:`\mathbf{Q}` stored as a product of Householder reflections, which is
    more numerically stable than Gram-Schmidt orthogonalisation.

    Args:
        A (JAXArray): Matrix of shape ``(N, k)`` with :math:`k \leq N`.
        reverse (bool, optional): Whether the reflections map onto the last :math:`k` basis vectors rather than
            the first :math:`k`. Defaults to ``False``.

    Returns:
        (JAXArray, JAXArray, Callable): A tuple of three elements: the :math:`(k, k)` R factor (with row signs
        matching the Householder vectors); the Householder vectors, of shape ``(N, k)``; and a function
        ``householder_transform(R, transpose = 0)`` which multiplies by the full :math:`(N, N)` matrix
        :math:`\mathbf{Q}` (or :math:`\mathbf{Q}^T`).

    """
    
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
