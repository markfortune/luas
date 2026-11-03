import numpy as np
import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional

from luas.kernels.covtype import CovType, Outer
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import kron_prod, calc_total_size, cyclic_transpose, vmap_for_tensors, tensor_mult
from luas.kernels.WhiteNoiseKernel import WhiteNoiseKernel

__all__ = [
    "LuasKernel (n-dimensional)",
]

class LuasKernelND(CovType):
    r"""Kronecker-structured covariance class for multi-dimensional inputs.

    This class models a covariance matrix of the form

    .. math::
        C = \bigotimes_d \Sigma_d + \bigotimes_d K_d,

    where each ``\Sigma_d`` and ``K_d`` is a per-dimension covariance operator.
    The implementation exploits Kronecker structure to perform decompositions and
    linear algebra without explicitly constructing dense full matrices in many
    internal operations.

    Args:
        Sigma: Sequence of per-dimension covariance objects for the first Kronecker term.
        K: Sequence of per-dimension covariance objects for the second Kronecker term.
        use_stored_values (bool, optional): Whether to reuse cached decompositions.
            Currently not implemented in this class version.
    """
    
    def __init__(
        self,
        Sigma,
        K,
        use_stored_values: Optional[bool] = False,
        **kwargs,
    ):
        
        self.Sigma = Sigma
        self.K = K
        self.dim = len(Sigma)
        self.opt_name = "LuasKernelND"
        
        self.K_list = [K]
           
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            raise NotImplementedError("Use of pre-calculated stored values not yet implemented.")
        else:
            self.decompose = self.decompose_no_stored_values
    
    def evaluate(self, X, **kwargs):

        dim = len(X)
        Sigma = self.Sigma[0].evaluate(X[0], X[0], **kwargs)
        
        for d in range(1, dim):
            Sigma = jnp.kron(Sigma, self.Sigma[d].evaluate(X[d], X[d], **kwargs))
        
        K = self.K[0].evaluate(X[0], X[0], **kwargs)
        for d in range(1, dim):
            K = jnp.kron(K, self.K[d].evaluate(X[d], X[d], **kwargs))
        
        Sigma += K
        
        return Sigma
    
    def decompose_no_stored_values(
        self,
        X: Tuple,
        stored_values: Optional[PyTree] = {},
        full = True,
        idx = None,
        **kwargs,
    ) -> PyTree:

        total_size = calc_total_size(X)

        if idx is None:
            idx = (None,)*self.dim
        
        gp_dim = len(X)

        self.sigma_decomp_mats = ()
        self.eigen_decomp_mats = ()
        all_lam = jnp.ones(1)
        all_lam_shape = ()
        stored_values["logdet"] = 0.

        for d in range(gp_dim):
            Sigma_d_new, stored_values_d = self.Sigma[d].decompose(X[d], full = full, idx = idx[d])

            if isinstance(self.K[d], Outer):
                K_d, _ = self.K[d].decompose(X[d], full = full, idx = idx[d])
            else:
                K_d = self.K[d]
            
            K_d_new = Sigma_d_new.inv_sqrt_transform(K_d, X[d], full = full, idx = idx[d])
            
            if gp_dim > 2:
                Sigma_d_new.matrix_sqrt = vmap_for_tensors(Sigma_d_new.matrix_sqrt)
                Sigma_d_new.matrix_inv_sqrt = vmap_for_tensors(Sigma_d_new.matrix_inv_sqrt)
                K_d_new.matrix_sqrt = vmap_for_tensors(K_d_new.matrix_sqrt)
                K_d_new.matrix_inv_sqrt = vmap_for_tensors(K_d_new.matrix_inv_sqrt)
            
            stored_values[f"lam_{d}"], stored_values[f"Q_{d}"] = K_d_new.eigendecomp(X[d], full = full, idx = idx[d])
            
            all_lam = jnp.kron(all_lam.reshape(all_lam_shape + (1,)), stored_values[f"lam_{d}"])
            all_lam_shape = all_lam.shape
            
            self.sigma_decomp_mats += (Sigma_d_new,)
            self.eigen_decomp_mats += (stored_values[f"Q_{d}"],)
            stored_values["logdet"] += (total_size/X[d].shape[-1])*stored_values_d["logdet"]

        K_tilde_diag = all_lam + 1
        
        self.kf_tilde = WhiteNoiseKernel(diag = K_tilde_diag, wn_diag = 0.)
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose(X)

        stored_values["logdet"] += stored_values["kf_tilde_stored"]["logdet"]
        self.logdet = stored_values["logdet"]

        return self, stored_values


    def transform_fn(self, R, transpose = 0):
        
        R_prime = cyclic_transpose(R, 2)

        if transpose:
            for d in range(self.dim):
                R_prime = self.sigma_decomp_mats[d].matrix_sqrt(R_prime, transpose = 1)
                R_prime = self.eigen_decomp_mats[d].T @ R_prime
                R_prime = cyclic_transpose(R_prime, 1)
        else:
            for d in range(self.dim):
                R_prime = self.eigen_decomp_mats[d] @ R_prime
                R_prime = self.sigma_decomp_mats[d].matrix_sqrt(R_prime, transpose = 0)
                R_prime = cyclic_transpose(R_prime, 1)

        R_prime = cyclic_transpose(R_prime, -2)
            
        return R_prime


    def inv_transform_fn(self, R, transpose = 0):
        R_prime = cyclic_transpose(R, 2)

        if transpose:
            for d in range(self.dim):
                R_prime = self.eigen_decomp_mats[d] @ R_prime
                R_prime = self.sigma_decomp_mats[d].matrix_inv_sqrt(R_prime, transpose = 1)
                R_prime = cyclic_transpose(R_prime, 1)
        else:
            for d in range(self.dim):
                R_prime = self.sigma_decomp_mats[d].matrix_inv_sqrt(R_prime, transpose = 0)
                R_prime = self.eigen_decomp_mats[d].T @ R_prime
                R_prime = cyclic_transpose(R_prime, 1)

        R_prime = cyclic_transpose(R_prime, -2)
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
        K_R += tensor_mult(self.K, X1, X2, R, **kwargs)
        
        return K_R
