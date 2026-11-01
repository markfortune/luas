import numpy as np
from typing import Optional, Callable, Tuple, Any, Union
import scipy

import jax
from jax import tree_util
import jax.numpy as jnp
import jax.scipy.linalg as JLA

import tinygp
from tinygp.solvers.quasisep.core import SymmQSM, StrictLowerTriQSM, DiagQSM
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
import luas.kernels.tinygp_ext
from luas.kernels.tinygp_ext import ScaledKernel, HandleIdx
from luas.kernels.householder import HouseholderTransform, HouseholderandDiagonalTransform
from luas.stable_gradients import stable_eigh


class CovType():
    K_list = []
    params = None
    use_inv = False
    diag = 0.
    wn_diag = 0.

    def evaluate(self, x1, x2, **kwargs):
        raise Exception("Not implemented")
    
    def decompose(self, x, **kwargs):
        raise Exception("Not implemented")

    def matrix_inv_sqrt(self, R, transpose=0):
        raise Exception("Not implemented")

    def matrix_sqrt(self, R, transpose=0):
        raise Exception("Not implemented")

    def scale(self, c):
        raise Exception("Not implemented")
    
    def eigendecomp(self, x, wn = True, full = True, idx = None, **kwargs):

        if not full:
            K = self.evaluate(x, x, wn = wn, full = False,
                              row_idx = idx, col_idx = idx, **kwargs)
        else:
            K = self.evaluate(x, x, wn = wn, full = True, **kwargs)
        return stable_eigh(K)
    
    def inverse(self, R):
        R_prime = self.matrix_inv_sqrt(R, transpose=0)
        K_inv_R = self.matrix_inv_sqrt(R_prime, transpose=1)

        return K_inv_R
    
    def dot_solve(self, R):
        if self.use_inv:
            # For some optimisations its easier to get the inverse then the matrix inverse sqrt
            K_inv_R = self.inverse(R)
            return (R * K_inv_R).sum()
        else:
            L_inv_R = self.matrix_inv_sqrt(R, transpose = 0)
            return jnp.square(L_inv_R).sum()
    
    # def logdet_calc(self):
    #     # Defining a function makes it easier to define custom derivatives to avoid numerical stability issues
    #     return self.logdet

    def logL(self, R, **kwargs):
        return - 0.5 * self.dot_solve(R) - 0.5 * self.logdet - 0.5 * R.size * jnp.log(2*jnp.pi)
    
    def logL_hessianable(self, R, **kwargs):
        return - 0.5 * self.dot_solve(R) - 0.5 * self.logdet - 0.5 * R.size * jnp.log(2*jnp.pi)

    def inv_sqrt_transform(self, K, x, **K_kwargs):
        
        if isinstance(K, Outer):
            K, _ = K.decompose(x, **K_kwargs)
            alpha_tilde = self.matrix_inv_sqrt(K.alpha, transpose=0)
            K_tilde = Outer(alpha = alpha_tilde)

        elif isinstance(K, (General, GeneralQuasisep, GeneralQuasisepPlusNoise, Identity,
                            ScaledIdentity, Diagonal, OuterPlusScaledIdentity)):
            # Define a new General covtype for K where it's kernel function transforms it
            K_eval = K.evaluate(x, x, row_idx = K_kwargs.get("idx"), col_idx = K_kwargs.get("idx"), full = K_kwargs.get("full", True))
            K_prime = self.matrix_inv_sqrt(K_eval, transpose=0)
            K_tilde_eval = self.matrix_inv_sqrt(K_prime.T, transpose=0)

            def kf_transf(hp, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs):
                if full:
                    return K_tilde_eval
                else:                   
                    return K_tilde_eval[jnp.ix_(row_idx, col_idx)]

            K_tilde = General(kf_transf)

        else:
            try:
                K_tilde = K.transform_with_inv_sqrt(self.matrix_inv_sqrt, x, **K_kwargs)
            except:
                raise Exception(f"Don't know how to transform this matrix type {type(K)} to K_tilde = L^-1 K L^-T")
        
        return K_tilde


    def general_transf(self, mat, x, **K_kwargs):
        K_eval = mat @ self.evaluate(x, x, **K_kwargs) @ mat.T
        def kf_transf(hp, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs):
            if full:
                return K_eval
            else:
                return K_eval[jnp.ix_(row_idx, col_idx)]
            
        return General(kf_transf)


    def matmul(self, x1, x2, R, **kwargs):
        return self.evaluate(x1, x2, **kwargs) @ R
    
    def rank(self, x, **kwargs):
        return x.shape[-1]
        
    def __call__(self, x1, x2, **kwargs):
        return self.evaluate(x1, x2, **kwargs)
    
    def __add__(self, other):
        raise Exception("Not implemented")

    def __mul__(self, other):
        raise Exception("Not implemented")
    
    def __radd__(self, other):
        return self.__add__(other)
        
    def __rmul__(self, other):
        return self.__mul__(other)



class Conditional(CovType):
    def __init__(self, kf, observed_ind, hp = {}, params = None):
        self.kf = kf
        self.hp = hp
        self.params = params
        self.opt_name = "General (1D)"
        
    def evaluate(self, x1, x2, **kwargs):
        K = self.kf(self.hp, x1, x2, **kwargs)

        return K

    def decompose(self, x, full = True, idx = None, **kwargs):
        # By default returns the lower triangular Cholesky factor
        # i.e. K = L @ L.T
        # Matches with tinygp default but not scipy or jax.scipy default
        
        K = self.evaluate(x, x, full = full, row_idx = idx, col_idx = idx, **kwargs)
        
        self.factor = JLA.cholesky(K, lower=True)
        self.logdet = 2*jnp.log(jnp.diag(self.factor)).sum()
        
        return self, {"logdet":self.logdet, "factor":self.factor, "hp":self.hp}

    def matrix_inv_sqrt(self, R, transpose=0, **kwargs):

        R_prime = jax.scipy.linalg.solve_triangular(self.factor, R, trans=transpose, lower=True)

        return R_prime
    
    def matrix_sqrt(self, R, transpose=0, **kwargs):

        if transpose:
            return self.factor.transpose() @ R
        else:
            return self.factor @ R
        
    def scale(self, c):
        return General(kf = lambda hp, x1, x2, **kwargs: c*self.evaluate(x1, x2, **kwargs))
    
    
    def householder_transform(self, x, u):
        w_i = self.matmul(x, x, u, full = True)
        u_dot_w = jnp.dot(u, w_i)

        def new_kf(hp, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs):
            K = self.kf(hp, x1, x2, full = full,
                        row_idx = row_idx, col_idx = col_idx, **kwargs)

            if row_idx is not None and col_idx is not None:
                u_row, u_col = u[row_idx], u[col_idx]
                w_i_row, w_i_col = w_i[row_idx], w_i[col_idx]
            elif full == True:
                u_row, u_col = u.copy(), u.copy()
                w_i_row, w_i_col = w_i.copy(), w_i.copy()
            else:
                raise Exception("Must specify row and col indices or else set full = True to evaluate the full covariance matrix")
            
            K += 4 * u_dot_w * jnp.outer(u_row, u_col)
            K -= 2 * (jnp.outer(u_row, w_i_col) + jnp.outer(w_i_row, u_col))
            return K
        
        return General(new_kf)
    
    def __add__(self, other):
        if isinstance(other, CovType):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))
        elif isinstance(other, (jax.Array, np.ndarray)) or is_scalar(other):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other)
        else:
            raise Exception("Not implemented")

    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other.evaluate(x1, x2, **kwargs))
        elif is_scalar(other):
            return self.scale(other)
        elif isinstance(other, jax.Array) or isinstance(other, np.ndarray):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other)
        else:
            raise Exception("Not implemented")


class General(CovType):
    def __init__(self, kf, hp = {}, params = None):
        self.kf = kf
        self.hp = hp
        self.params = params
        self.opt_name = "General (1D)"
        
    def evaluate(self, x1, x2, **kwargs):
        K = self.kf(self.hp, x1, x2, **kwargs)

        return K

    def decompose(self, x, full = True, idx = None, **kwargs):
        # By default returns the lower triangular Cholesky factor
        # i.e. K = L @ L.T
        # Matches with tinygp default but not scipy or jax.scipy default
        
        K = self.evaluate(x, x, full = full, row_idx = idx, col_idx = idx, **kwargs)
        
        self.factor = JLA.cholesky(K, lower=True)
        self.logdet = 2*jnp.log(jnp.diag(self.factor)).sum()
        
        return self, {"logdet":self.logdet, "factor":self.factor, "hp":self.hp}

    def matrix_inv_sqrt(self, R, transpose=0, **kwargs):

        R_prime = jax.scipy.linalg.solve_triangular(self.factor, R, trans=transpose, lower=True)

        return R_prime
    
    def matrix_sqrt(self, R, transpose=0, **kwargs):

        if transpose:
            return self.factor.transpose() @ R
        else:
            return self.factor @ R
        
    def scale(self, c):
        return General(kf = lambda hp, x1, x2, **kwargs: c*self.evaluate(x1, x2, **kwargs))
    
    
    def householder_transform(self, x, u):
        w_i = self.matmul(x, x, u, full = True)
        u_dot_w = jnp.dot(u, w_i)

        def new_kf(hp, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs):
            K = self.kf(hp, x1, x2, full = full,
                        row_idx = row_idx, col_idx = col_idx, **kwargs)

            if row_idx is not None and col_idx is not None:
                u_row, u_col = u[row_idx], u[col_idx]
                w_i_row, w_i_col = w_i[row_idx], w_i[col_idx]
            elif full == True:
                u_row, u_col = u.copy(), u.copy()
                w_i_row, w_i_col = w_i.copy(), w_i.copy()
            else:
                raise Exception("Must specify row and col indices or else set full = True to evaluate the full covariance matrix")
            
            K += 4 * u_dot_w * jnp.outer(u_row, u_col)
            K -= 2 * (jnp.outer(u_row, w_i_col) + jnp.outer(w_i_row, u_col))
            return K
        
        return General(new_kf)
    
    def __add__(self, other):
        if isinstance(other, CovType):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))
        elif isinstance(other, (jax.Array, np.ndarray)) or is_scalar(other):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other)
        else:
            raise Exception("Not implemented")

    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other.evaluate(x1, x2, **kwargs))
        elif is_scalar(other):
            return self.scale(other)
        elif isinstance(other, jax.Array) or isinstance(other, np.ndarray):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other)
        else:
            raise Exception("Not implemented")


class Identity(CovType):

    def __init__(self):
        self.diag = 1.
        self.wn_diag = 0.
        self.logdet = 0.
        self.params = []
        self.opt_name = "WhiteNoiseKernel (1D)"

    def evaluate(self, x1, x2, row_idx = None, col_idx = None, full = True, **kwargs):

        if full:
            mat = jnp.eye(x1.shape[-1])
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if idx_specified:
                mask = row_idx[:, None] == col_idx[None, :]
                mat = jnp.where(mask, 1.0, 0.0)
            else:
                raise Exception("""Cannot evaluate non-stationary diagonal covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                            )
        
        return mat
    
    def decompose(self, x, **kwargs):
        return self, {"logdet":0.}

    def matrix_inv_sqrt(self, R, transpose=0):
        return R
        
    def matrix_sqrt(self, R, **kwargs):
        return R
    
    def eigendecomp(self, x, **kwargs):
        return jnp.ones(x.shape[-1]), jnp.eye(x.shape[-1])
    
    def inverse(self, R, **kwargs):
        return R
        
    def dot_solve(self, R):
        return jnp.square(R).sum()
        
    def inv_sqrt_transform(self, K, x, **kwargs):
        return K
    
    def householder_transform(self, x, u):
        return self

    def scale(self, c):
        return ScaledIdentity(diag = c)

    def matmul(self, x1, x2, R, full = True, **kwargs):
        if full:
            return R
        else:
            mat = self.evaluate(x1, x2, full = False, **kwargs)
            return mat @ R
    
    def __add__(self, K):
        if isinstance(K, (GeneralQuasisep, GeneralQuasisepPlusNoise)):
            K_sum = GeneralQuasisepPlusNoise(K.tinygp_kf, diag = K.diag + 1., wn_diag = K.wn_diag, use_block = K.use_block, noise_model = K.noise_model)

        elif isinstance(K, Outer):
            K_sum = OuterPlusScaledIdentity(K.alpha_init, diag = 1.)

        elif isinstance(K, OuterPlusScaledIdentity):
            K_sum = OuterPlusScaledIdentity(K.alpha, diag = 1. + K.diag, wn_diag = K.wn_diag)

        elif isinstance(K, Identity):
            K_sum = ScaledIdentity(diag = 2.)

        elif isinstance(K, (ScaledIdentity, Diagonal)):
            K_sum = type(K)(diag = K.diag + 1., wn_diag = K.wn_diag)

        elif isinstance(K, General):
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))

        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum
    
    def __mul__(self, other) -> Kernel:
        if is_scalar(other):
            return self.scale(other)
        else:
            raise Exception("Not implemented")


        
class ScaledIdentity(CovType):
    def __init__(self, diag = 0., wn_diag = 0., params = None):
        self.diag = diag
        self.wn_diag = wn_diag
        self.params = params
        self.opt_name = "WhiteNoiseKernel (1D)"

        assert is_scalar(self.diag)
        assert is_scalar(self.wn_diag)
        
    def evaluate(self, x1, x2, wn = True, row_idx = None, col_idx = None, full = True, **kwargs):

        if full:
            mat = (self.diag + wn*self.wn_diag) * jnp.eye(x1.shape[-1])
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if idx_specified:
                mask = row_idx[:, None] == col_idx[None, :]
                d = self.diag + wn*self.wn_diag
                mat = jnp.where(mask, d, 0.0)
            else:
                raise Exception("""Cannot evaluate non-stationary diagonal covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                            )
        return mat

    def decompose(self, x, wn = True, **kwargs):

        self.D = self.diag + wn*self.wn_diag
        self.factor = jnp.sqrt(self.D)
        self.logdet = x.shape[-1]*jnp.log(self.D)
        
        return self, {"logdet":self.logdet}

    def matrix_inv_sqrt(self, R, **kwargs):

        return R/self.factor

    def matrix_sqrt(self, R, **kwargs):

        return self.factor * R
        
    def eigendecomp(self, x, wn = True, **kwargs):

        const_diag = self.diag + wn*self.wn_diag
        
        return const_diag*jnp.ones(x.shape[-1]), jnp.eye(x.shape[-1])

    def inv_sqrt_transform(self, K, x, **K_kwargs):

        K_tilde = K.scale(1/self.D)

        return K_tilde

    def matmul(self, x1, x2, R, wn = True, full = True, **kwargs):

        if full:
            assert x2.shape[-1] == R.shape[0] # Multiplication is only allowed in this case
            const = self.diag + wn*self.wn_diag
            return const * R
        else:
            # Could do something more efficient, return certain elements of R after scaling
            mat = self.evaluate(x1, x2, wn = wn, full = False, **kwargs)
            return mat @ R
        
    def householder_transform(self, x, u):
        return self

    def scale(self, c):
        return ScaledIdentity(diag = self.diag * c, wn_diag = self.wn_diag * c)

    def __add__(self, K):
        if isinstance(K, (GeneralQuasisep, GeneralQuasisepPlusNoise)):
            K_sum = GeneralQuasisepPlusNoise(K.tinygp_kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag,
                                             noise_model = K.noise_model, use_block = K.use_block)
            
        elif isinstance(K, (Identity, ScaledIdentity)):
            K_sum = ScaledIdentity(diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)

        elif isinstance(K, Diagonal):
            K_sum = Diagonal(diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)

        elif isinstance(K, (Outer, OuterPlusScaledIdentity)):
            K_sum = OuterPlusScaledIdentity(K.alpha_init, diag = self.diag, wn_diag = self.wn_diag)

        elif isinstance(K, General):
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))

        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum
    
    def __mul__(self, other) -> Kernel:
        if is_scalar(other):
            return self.scale(other)
        else:
            raise Exception("Not implemented")


class Diagonal(CovType):
    def __init__(self, diag = 0., wn_diag = 0.):

        self.N = (diag + wn_diag).size
        self.diag = diag * jnp.ones(self.N)
        self.wn_diag = wn_diag * jnp.ones(self.N)
        self.opt_name = "WhiteNoiseKernel (1D)"

        assert self.diag.ndim == 1
        assert self.wn_diag.ndim == 1
    
    def evaluate(self, x1, x2, wn = True, row_idx = None, col_idx = None, full = True, **kwargs):

        if full:
            diag_mat = jnp.diag(self.diag + wn*self.wn_diag)
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if idx_specified:
                mask = row_idx[:, None] == col_idx[None, :]  # (n1, n2) bool matrix

                # Pick the diagonal values where they match
                diag_vals = self.diag + wn*self.wn_diag
                diag_mat = jnp.where(mask, diag_vals[row_idx[:, None] * mask], 0.0)
                
            else:
                raise Exception("""Cannot evaluate non-stationary diagonal covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                            )
        return diag_mat
    
    def decompose(self, x, wn = True, **kwargs):

        self.D = self.diag + wn*self.wn_diag
        self.factor = jnp.sqrt(self.D)
        self.logdet = jnp.log(self.D).sum()
        
        return self, {"logdet":self.logdet, "hp":{"diag":self.diag, "wn_diag":self.wn_diag}}
    
    def matrix_sqrt(self, R, **kwargs):
        
        return jnp.einsum('i,i...->i...', self.factor, R)
        
    def matrix_inv_sqrt(self, R, transpose=0):
        
        D_inv_sqrt = 1/self.factor

        return jnp.einsum('i,i...->i...', D_inv_sqrt, R)
    
    def eigendecomp(self, x, wn = True, **kwargs):

        D = self.diag + wn*self.wn_diag
        
        # Not very memory efficient building a full identity matrix...
        return D, jnp.eye(x.shape[-1])
    
    def inv_sqrt_transform(self, K, x, **K_kwargs):

        if isinstance(K, (ScaledIdentity, Diagonal)):
            K_tilde = Diagonal(diag = K.diag/self.D, wn_diag = K.wn_diag/self.D)

        elif isinstance(K, Identity):
            K_tilde = Diagonal(diag = 1/self.D)

        elif isinstance(K, Outer):
            K_tilde = Outer(alpha = K.alpha_init/jnp.sqrt(self.D))
        
        elif isinstance(K, (GeneralQuasisep, GeneralQuasisepPlusNoise)):
            new_kf = ScaledKernel(kernel = K.tinygp_kf, amplitudes = 1/jnp.sqrt(self.D))
            new_diag = K.diag/self.D
            new_wn_diag = K.wn_diag/self.D

            # Yet to implement for the case of a general noise model added
            assert K.noise_model is None

            K_tilde = GeneralQuasisepPlusNoise(new_kf, diag = new_diag, wn_diag = new_wn_diag, use_block = K.use_block)

        elif isinstance(K, General):
            D_inv_sqrt = 1/jnp.sqrt(self.D)
            K_eval = jnp.outer(D_inv_sqrt, D_inv_sqrt) * K.evaluate(x, x, row_idx = K_kwargs.get("idx"), col_idx = K_kwargs.get("idx"), full = K_kwargs.get("full", True))
            # Define a new General covtype for K where it's kernel function transforms it
            def kf_transf(hp, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs):
                if full:
                    return K_eval
                else:
                    return K_eval[jnp.ix_(row_idx, col_idx)]

            K_tilde = General(kf_transf)

        else:
            try:
                K_tilde = K.transform_with_inv_sqrt(self.matrix_inv_sqrt, x)
            except:
                raise Exception(f"Don't know how to transform this matrix type {type(K)} to K_tilde = L^-1 K L^-T")
            
        return K_tilde
    
    def inverse(self, R, transpose=0):
        
        D_inv = 1/self.D

        return jnp.einsum('i,i...->i...', D_inv, R)

    def matmul(self, x1, x2, R, wn = True, full = True, **kwargs):

        if full:
            D = self.diag + wn*self.wn_diag
            return jnp.einsum('i,i...->i...', D, R)
        else:
            D_mat = self.evaluate(x1, x2, wn = wn, full = False, **kwargs)
            return D_mat @ R

    def scale(self, c):
        return Diagonal(diag = self.diag * c, wn_diag = self.wn_diag * c)
    
    def householder_transform(self, x, u):
        w_i = self.matmul(x, x, u, full = True)

        # Can this ever be super close to zero? Would be very numerically unstable but seems unlikely?
        u_dot_w = jnp.dot(u, w_i)
        alpha = 2 * u_dot_w * u - w_i

        new_tinygp_kf = luas.kernels.tinygp_ext.Linear(alpha = alpha, const = 1/u_dot_w)
        new_tinygp_kf += luas.kernels.tinygp_ext.Linear(alpha = w_i, const = -1/u_dot_w)
        
        return GeneralQuasisepPlusNoise(new_tinygp_kf, diag = self.diag, wn_diag = self.wn_diag)

    def __add__(self, K):
        if isinstance(K, (GeneralQuasisep, GeneralQuasisepPlusNoise)):
            K_sum = GeneralQuasisepPlusNoise(K.tinygp_kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag,
                                             noise_model = K.noise_model, use_block = K.use_block)

        elif isinstance(K, (Identity, ScaledIdentity, Diagonal)):
            K_sum = Diagonal(diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)

        elif isinstance(K, (Outer, OuterPlusScaledIdentity, General)):
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))

        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

    def __mul__(self, other) -> Kernel:
        if is_scalar(other):
            return self.scale(other)
        else:
            raise Exception("Not implemented")


class GeneralQuasisepPlusNoise(CovType):
    def __init__(self, tinygp_kf, diag = 0., wn_diag = 0., noise_model = None, params = None, use_block = True):
        self.tinygp_kf = tinygp_kf
        self.diag = diag
        self.wn_diag = wn_diag
        self.noise_model = noise_model
        self.use_block = use_block
        self.params = params
        self.opt_name = "Quasiseparable (1D)"


    def _tinygp_coords(self, x1, x2, row_idx = None, col_idx = None, full = True):

        if full:
            X1 = (x1, jnp.arange(x1.shape[-1]))
            X2 = (x2, jnp.arange(x2.shape[-1]))
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if idx_specified:
                # Potentially inefficient to build full diag_mat but shouldn't be doing this often
                X1 = (x1, row_idx)
                X2 = (x2, col_idx)
            else:
                raise Exception("""Cannot evaluate non-stationary diagonal covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                            )
        return X1, X2


    def _to_symm_qsm(self, x, wn = True, full = True, idx = None, stored_values = {}):
        if full:
            idx = jnp.arange(x.shape[-1])
        else:
            assert idx is not None

        diag = (self.diag + wn*self.wn_diag)
        
        if is_scalar(diag):
            diag = diag*jnp.ones(x.shape[-1])
        else:
            diag = diag[idx]
        
        qsm_matrix = tinygp.noise.Diagonal(diag=diag).to_qsm()

        if self.noise_model is not None:
            qsm_matrix += self.noise_model

        if self.tinygp_kf is not None:
            qsm_matrix += self.tinygp_kf.to_symm_qsm((x, idx))
    
        return qsm_matrix
    
    def decompose(self, x, wn = True, full = True, **kwargs):

        matrix = self._to_symm_qsm(x, wn = wn, full = full, **kwargs)
        self.factor = matrix.cholesky()
        self.logdet = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
        return self, {"logdet":self.logdet}
        
    def matrix_inv_sqrt(self, R, transpose=0):

        if transpose:
            R_prime = self.factor.transpose().solve(R)
        else:
            R_prime = self.factor.solve(R)
            
        return R_prime

    def matrix_sqrt(self, R, transpose=0, **kwargs):

        if transpose:
            return self.factor.transpose() @ R
        else:
            return self.factor @ R
    
    def evaluate(self, x1, x2, wn = True, row_idx = None, col_idx = None, full = True, **kwargs):

        X1, X2 = self._tinygp_coords(x1, x2, row_idx = row_idx, col_idx = col_idx, full = full)

        kernel_model = self.tinygp_kf(X1, X2)

        if is_scalar(self.diag + self.wn_diag):
            diag_cov = ScaledIdentity(diag = self.diag, wn_diag = self.wn_diag)
        else:
            diag_cov = Diagonal(diag = self.diag, wn_diag = self.wn_diag)

        noise_model = diag_cov.evaluate(x1, x2, wn = wn, row_idx = row_idx, col_idx = col_idx, full = full)
        
        if self.noise_model is not None:
            if full:
                noise_model += self.noise_model.to_dense()
            else:
                qsm_model = self.noise_model.to_dense()
                noise_model += qsm_model[jnp.ix_(row_idx, col_idx)]
        
        return kernel_model + noise_model
    

    def matmul(self, x1, x2, R, wn = True, full = True, **kwargs):
        # Assumes matrix is evaluated at positions of diagonal, noise_model

        X1 = (x1, jnp.arange(x1.shape[-1]))
        X2 = (x2, jnp.arange(x2.shape[-1]))
        
        diag_eval = (self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1])
        R_prime = jnp.einsum('i,i...->i...', diag_eval, R)

        if self.tinygp_kf is not None:
            if full:
                R_prime += self.tinygp_kf.matmul(X1, R)
            else:
                R_prime += self.tinygp_kf.matmul(X1, X2, R)

        if self.noise_model is not None:
            R_prime += self.noise_model @ R
        
        return R_prime


    def scale(self, c):
        if self.tinygp_kf is not None: 
            scaled_kernel = luas.kernels.tinygp_ext.Scale(kernel = self.tinygp_kf, scale = c)
        else:
            scaled_kernel = None

        if self.noise_model is None:
                new_noise_model = None
        else:
            new_noise_model = self.noise_model * c

        return GeneralQuasisepPlusNoise(scaled_kernel, diag = self.diag * c, wn_diag = self.wn_diag * c,
                                        noise_model = new_noise_model, use_block = self.use_block)
    

    def householder_transform(self, x, u):
        w_i = self.matmul(x, x, u, full = True)

        # Can this ever be super close to zero? Would be very numerically unstable but seems unlikely?
        u_dot_w = jnp.dot(u, w_i)
        alpha = 2 * u_dot_w * u - w_i

        new_tinygp_kf = self.tinygp_kf + luas.kernels.tinygp_ext.Linear(alpha = alpha, const = 1/u_dot_w)
        new_tinygp_kf += luas.kernels.tinygp_ext.Linear(alpha = w_i, const = -1/u_dot_w)
        
        return GeneralQuasisepPlusNoise(new_tinygp_kf, diag = self.diag, wn_diag = self.wn_diag,
                                        noise_model = self.noise_model, use_block = self.use_block)
    

    def __add__(self, K):

        if isinstance(K, (Identity, ScaledIdentity, Diagonal)):
            K_sum = GeneralQuasisepPlusNoise(self.tinygp_kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag,
                                            noise_model = self.noise_model, use_block = self.use_block)

        elif isinstance(K, GeneralQuasisep):
            kernel_sum = luas.kernels.tinygp_ext.Sum(self.tinygp_kf, K.tinygp_kf, use_block = self.use_block)
            K_sum = GeneralQuasisepPlusNoise(kernel_sum, diag = self.diag, wn_diag = self.wn_diag, noise_model = self.noise_model)

        elif isinstance(K, (GeneralQuasisepPlusNoise)):
            if self.tinygp_kf is None:
                new_tinygp_kf = K.tinygp_kf
            elif K.tinygp_kf is None:
                new_tinygp_kf = self.tinygp_kf
            else:
                new_tinygp_kf = luas.kernels.tinygp_ext.Sum(self.tinygp_kf, K.tinygp_kf, use_block = self.use_block)

            if self.noise_model is not None and K.noise_model is not None:
                new_noise_model = self.noise_model + K.noise_model
            elif self.noise_model is not None:
                new_noise_model = self.noise_model
            elif K.noise_model is not None:
                new_noise_model = K.noise_model
            else:
                new_noise_model = None

            K_sum = GeneralQuasisepPlusNoise(new_tinygp_kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag,
                                             noise_model = new_noise_model, use_block = self.use_block)
            
        elif isinstance(K, SymmQSM):
            if self.noise_model is None:
                new_noise_model = K
            else:
                new_noise_model = K + self.noise_model

            K_sum = GeneralQuasisepPlusNoise(self.tinygp_kf, use_block = self.use_block,
                                            diag = self.diag, wn_diag = self.wn_diag, noise_model = new_noise_model)
            
        elif isinstance(K, (Outer, OuterPlusScaledIdentity)):
            if self.tinygp_kf is None:
                new_tinygp_kf = K.tinygp_kf
            else:
                new_tinygp_kf = luas.kernels.tinygp_ext.Sum(self.tinygp_kf, K.tinygp_kf, use_block = self.use_block)

            K_sum = GeneralQuasisepPlusNoise(new_tinygp_kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag,
                                             use_block = self.use_block, noise_model = self.noise_model)

        elif isinstance(K, (General)):
            # Should update as don't lose quasiseparability for Outer
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))

        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum
        
    # Should implement multiplying celerite kernels together but noise model terms need separate calc
    def __mul__(self, other) -> Kernel:
        if is_scalar(other):
            K_mult = self.scale(other)

        elif isinstance(other, GeneralQuasisep) and self.noise_model is None:

            K_mult = luas.kernels.tinygp_ext.Product(self.tinygp_kf, other.tinygp_kf)
            new_diag = self.diag * other.tinygp_kf.evaluate_diag((jnp.zeros(1), 0))
            new_wn_diag = self.wn_diag * other.tinygp_kf.evaluate_diag((jnp.zeros(1), 0))
            
            K_mult = GeneralQuasisepPlusNoise(K_mult, diag = new_diag, wn_diag = new_wn_diag,
                                            use_block = self.use_block, noise_model = None)
            
        elif isinstance(other, Outer):
            if is_scalar(other.alpha_init):
                K_mult = self.scale(other.alpha_init**2)
            else:
                new_kf = ScaledKernel(kernel = self.tinygp_kf, amplitudes = other.alpha_init)

                if self.noise_model is not None:
                    diag_term = DiagQSM(d=self.noise_model.diag.d * other.alpha_init**2)

                    p_new = jnp.einsum('i,i...->i...', diag_term, self.noise_model.lower.p)
                    q_new = jnp.einsum('i,i...->i...', diag_term, self.noise_model.lower.q)
                    lower_term = StrictLowerTriQSM(p=p_new, q=q_new, a=self.noise_model.lower.a)

                    new_noise_model = SymmQSM(diag=diag_term, lower=lower_term)
                else:
                    new_noise_model = None

                K_mult = GeneralQuasisepPlusNoise(new_kf, diag = self.diag * other.alpha_init**2, wn_diag = self.wn_diag * other.alpha_init**2,
                                                use_block = self.use_block, noise_model = new_noise_model)
        
        else:
            raise Exception("Multiplication of Quasisep kernels which are non-stationary or include a noise model not yet implemented")
        
        return K_mult


class GeneralQuasisep(CovType):
    def __init__(self, tinygp_kf, params = None, use_block = True):
        self.tinygp_kf = tinygp_kf
        self.diag = 0.
        self.wn_diag = 0.
        self.use_block = use_block
        self.noise_model = None
        self.params = params
        self.opt_name = "Quasiseparable (1D)"

    def _to_symm_qsm(self, x, wn = True, idx = None, stored_values = {}):

        if idx is None:
            idx = jnp.arange(x.shape[-1])

        matrix = self.tinygp_kf.to_symm_qsm((x, idx))
        return matrix
    
    def _tinygp_coords(self, x1, x2, row_idx = None, col_idx = None, full = True):

        X1 = (x1, jnp.arange(x1.shape[-1]))
        X2 = (x2, jnp.arange(x2.shape[-1]))

        return X1, X2

    def evaluate(self, x1, x2, wn = True, **kwargs):

        X1, X2 = self._tinygp_coords(x1, x2, **kwargs)
        return self.tinygp_kf(X1, X2)
    
    def decompose(self, x, wn = True, **kwargs):

        matrix = self._to_symm_qsm(x, wn = wn, **kwargs)
        self.factor = matrix.cholesky()
        self.logdet = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
        return self, {"logdet":self.logdet}
        
    def matrix_inv_sqrt(self, R, transpose=0):

        if transpose:
            R_prime = self.factor.transpose().solve(R)
        else:
            R_prime = self.factor.solve(R)
            
        return R_prime

    def matrix_sqrt(self, R, transpose=0, **kwargs):

        if transpose:
            return self.factor.transpose() @ R
        else:
            return self.factor @ R

    def scale(self, c):
        scaled_kernel = luas.kernels.tinygp_ext.Scale(kernel = self.tinygp_kf, scale = c)
        return GeneralQuasisep(scaled_kernel, use_block = self.use_block)
    
    def matmul(self, x1, x2, R, wn = True, full = True, **kwargs):

        X1, X2 = self._tinygp_coords(x1, x2, full = full, **kwargs)

        if full:
            R_prime = self.tinygp_kf.matmul(X1, R)
        else:
            R_prime = self.tinygp_kf.matmul(X1, X2, R)
        
        return R_prime
    
    def householder_transform(self, x, u):
        w_i = self.matmul(x, x, u, full = True)

        # Can this ever be super close to zero? Would be very numerically unstable but seems unlikely?
        u_dot_w = jnp.dot(u, w_i)
        alpha = 2 * u_dot_w * u - w_i

        new_tinygp_kf = self.tinygp_kf + luas.kernels.tinygp_ext.Linear(alpha = alpha, const = 1/u_dot_w)
        new_tinygp_kf += luas.kernels.tinygp_ext.Linear(alpha = w_i, const = -1/u_dot_w)
        
        return GeneralQuasisepPlusNoise(new_tinygp_kf, use_block = self.use_block)

    def __add__(self, K):

        if isinstance(K, Identity) or isinstance(K, ScaledIdentity) or isinstance(K, Diagonal):
            K_sum = GeneralQuasisepPlusNoise(self.tinygp_kf, diag = K.diag, wn_diag = K.wn_diag, use_block = self.use_block)

        elif isinstance(K, GeneralQuasisep):
            sum_kernel = luas.kernels.tinygp_ext.Sum(self.tinygp_kf, K.tinygp_kf, use_block = self.use_block)
            K_sum = GeneralQuasisep(sum_kernel, use_block = self.use_block)

        elif isinstance(K, GeneralQuasisepPlusNoise):
            if K.tinygp_kf is not None:
                new_tinygp_kf = luas.kernels.tinygp_ext.Sum(self.tinygp_kf, K.tinygp_kf, use_block = self.use_block)
            else:
                new_tinygp_kf = self.tinygp_kf
            K_sum = GeneralQuasisepPlusNoise(new_tinygp_kf, diag = K.diag, wn_diag = K.wn_diag, noise_model = K.noise_model, use_block = self.use_block)

        elif isinstance(K, SymmQSM):
            K_sum = GeneralQuasisepPlusNoise(self.tinygp_kf, use_block = self.use_block, noise_model = K)

        elif isinstance(K, (Outer, OuterPlusScaledIdentity)):
            K_sum = GeneralQuasisepPlusNoise(self.tinygp_kf + K.tinygp_kf, diag = K.diag, wn_diag = K.wn_diag, use_block = self.use_block)

        elif isinstance(K, (General)):
            # Should update as don't lose quasiseparability
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))

        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum
    
    def __mul__(self, other) -> Kernel:
        if isinstance(other, GeneralQuasisep):
            product_kernel = luas.kernels.tinygp_ext.Product(kernel1 = self.tinygp_kf, kernel2 = other.tinygp_kf)
            K_mult = GeneralQuasisep(product_kernel, use_block = self.use_block)
        
        elif is_scalar(other):
            K_mult = self.scale(other)
        
        elif isinstance(other, GeneralQuasisepPlusNoise) and other.noise_model is None:
            K_mult = luas.kernels.tinygp_ext.Product(self.tinygp_kf, other.tinygp_kf)
            new_diag = other.diag * self.tinygp_kf.evaluate_diag((jnp.zeros(1), 0))
            new_wn_diag = other.wn_diag * self.tinygp_kf.evaluate_diag((jnp.zeros(1), 0))
            
            K_mult = GeneralQuasisepPlusNoise(K_mult, diag = new_diag, wn_diag = new_wn_diag,
                                            use_block = self.use_block, noise_model = None)
            
        elif isinstance(other, Outer):
            if is_scalar(other.alpha_init):
                K_mult = self.scale(other.alpha_init**2)
            else:
                new_kf = ScaledKernel(kernel = self.tinygp_kf, amplitudes = other.alpha_init)
                K_mult = GeneralQuasisepPlusNoise(new_kf, use_block = self.use_block)
            
        else:
            raise Exception(f"Multiplication between Quasisep kernels and non Quasisep kernels not implemented")
        
        return K_mult
    

class PeriodicQuasisep(GeneralQuasisep):
    def scale(self, c):
        scaled_kernel = luas.kernels.tinygp_ext.Scale(kernel = self.tinygp_kf, scale = c)
        return PeriodicQuasisep(scaled_kernel, use_block = self.use_block)


class Exp(GeneralQuasisep):
    def __init__(self, tinygp_kf, scale, sigma = 1., params = None, use_block = True, fast_eigen = False):
        self.len_scale = scale
        self.sigma = sigma
        self.tinygp_kf = tinygp_kf
        self.diag = 0.
        self.wn_diag = 0.
        self.noise_model = None
        self.use_block = use_block
        self.params = params
        self.fast_eigen = fast_eigen
        self.opt_name = "Quasiseparable (1D)"

        if self.fast_eigen:
            # Note gradients not implemented yet for this method
            self.eigendecomp = self.eigendecomp_fast

    def scale(self, c):
        scaled_kernel = luas.kernels.tinygp_ext.Scale(kernel = self.tinygp_kf, scale = c)
        return Exp(scaled_kernel, self.len_scale, sigma = self.sigma * jnp.sqrt(c),
                   use_block = self.use_block, fast_eigen = self.fast_eigen)

    def eigendecomp_fast(self, x, **kwargs):
        N = x.shape[-1]
        result_shape = (jax.ShapeDtypeStruct((N,), x.dtype), jax.ShapeDtypeStruct((N, N), x.dtype))
        lam, Q = jax.pure_callback(self.fast_exp_eigh_scipy, result_shape, x, self.len_scale)
        return self.sigma**2 * lam, Q
        
    def fast_exp_eigh_scipy(self, x1, scale):
    
        if x1.ndim == 2:
            x1 = x1[0, :]
            
        r = np.exp(-np.diff(x1/scale))
        e = 1/(1/r - r)
        
        arr_diag = np.zeros(x1.shape[-1])
        arr_diag[:-1] = 1 + r*e
        arr_diag[1:] += r*e
        arr_diag[-1] += 1
        
        lam, Q = scipy.linalg.eigh_tridiagonal(arr_diag, -e)
        
        return 1/lam, Q
    


class Outer(CovType):
    def __init__(self, alpha, params = None):
        self.alpha_init = alpha
        self.diag = 0.
        self.wn_diag = 0.
        self.params = params
        self.opt_name = "Not a valid kernel!"

        if is_scalar(alpha):
            self.tinygp_kf = HandleIdx(luas.kernels.tinygp_ext.Constant(const = alpha**2))
        else:
            self.tinygp_kf = luas.kernels.tinygp_ext.Linear(alpha = alpha)

        assert is_scalar(self.alpha_init) or self.alpha_init.ndim == 1

    def rank(self, x):
        return 1

    def evaluate(self, x1, x2, wn = True, row_idx = None, col_idx = None, full = True, **kwargs):

        if is_scalar(self.alpha_init):
            mat = self.alpha_init**2 * jnp.ones((x1.shape[-1], x2.shape[-1]))
        elif full:
            mat = jnp.outer(self.alpha_init, self.alpha_init)
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if idx_specified:
                mat = jnp.outer(self.alpha_init[row_idx], self.alpha_init[col_idx])
            else:
                raise Exception("""Cannot evaluate non-stationary outer product covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                                )
        return mat

    def decompose(self, x, **kwargs):
        # Matrix is decomposed by definition
        # Except need to handle ConstantKernel where self.alpha_init is a float

        if is_scalar(self.alpha_init):
            self.alpha = self.alpha_init * jnp.ones(x.shape[-1])
        else:
            self.alpha = self.alpha_init

        return self, {"logdet":-jnp.inf}
    
    def matrix_sqrt(self, R, **kwargs):

        if R.ndim == 1:
            return jnp.kron(self.alpha, self.alpha @ R)
        elif R.ndim == 2:
            return jnp.outer(self.alpha, self.alpha @ R)
        else:
            raise Exception("Not implemented")

    def matrix_inv_sqrt(self, R, **kwargs):

        raise Exception("Outer product matrix is not invertible!")
    
    def inv_sqrt_transform(self, K, x, **kwargs):

        raise Exception("Outer product matrix is not invertible!")

    def scale(self, c):
        return Outer(self.alpha_init * jnp.sqrt(c))

    def eigendecomp(self, x, i = -1, **kwargs):
        # Calculates the Householder transformation of alpha for the eigenvector matrix
        # And the matrix's sole non-zero eigenvalue which is the squared norm of alpha

        H = HouseholderTransform(self.alpha_init * jnp.ones(x.shape[-1]), i = i)
        
        return H.lam, H
    
    def matmul(self, x1, x2, R, **kwargs):

        if is_scalar(self.alpha_init):
            return jnp.outer(jnp.ones(x1.shape[-1]) * self.alpha_init**2, R.sum(0))
        else:
            self.vec1 = self.alpha_init * jnp.ones(x1.shape[-1])
            self.vec2 = self.alpha_init * jnp.ones(x2.shape[-1])
            
            if R.ndim == 1:
                return jnp.kron(self.vec1, self.vec2 @ R)
            elif R.ndim == 2:
                return jnp.outer(self.vec1, self.vec2 @ R)
            else:
                raise Exception("Not implemented")
        
            

    def __add__(self, K):

        if isinstance(K, (Identity, ScaledIdentity)):
            K_sum = OuterPlusScaledIdentity(self.alpha_init, diag = K.diag, wn_diag = K.wn_diag)

        elif isinstance(K, Outer):
            # Could do something better here
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))

        elif isinstance(K, (GeneralQuasisepPlusNoise, GeneralQuasisep, Exp)):
            if K.tinygp_kf is not None:
                new_tinygp_kf = luas.kernels.tinygp_ext.Sum(self.tinygp_kf, K.tinygp_kf, use_block = K.use_block)
            else:
                new_tinygp_kf = self.tinygp_kf

            K_sum = GeneralQuasisepPlusNoise(new_tinygp_kf, diag = K.diag, wn_diag = K.wn_diag,
                                             noise_model = K.noise_model,  use_block = K.use_block)

        elif isinstance(K, (Diagonal, OuterPlusScaledIdentity, General)):
            # Should update as you can always add outer products without losing quasiseparability
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))

        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

    # Could implement an efficient Quasisep multiplication
    def __mul__(self, K) -> Kernel:
        if is_scalar(K):
            K_mul = self.scale(K)

        elif is_scalar(self.alpha_init) and isinstance(K, CovType):
            K_mul = K.scale(self.alpha_init**2)
        
        elif isinstance(K, GeneralQuasisep):
            new_kf = ScaledKernel(kernel = K.tinygp_kf, amplitudes = self.alpha_init)
            K_mul = GeneralQuasisepPlusNoise(new_kf, use_block = K.use_block)
        
        elif isinstance(K, GeneralQuasisepPlusNoise):
            new_kf = ScaledKernel(kernel = K.tinygp_kf, amplitudes = self.alpha_init)

            if K.noise_model is not None:
                diag_term = DiagQSM(d=K.noise_model.diag.d * self.alpha_init**2)

                p_new = jnp.einsum('i,i...->i...', diag_term, K.noise_model.lower.p)
                q_new = jnp.einsum('i,i...->i...', diag_term, K.noise_model.lower.q)
                lower_term = StrictLowerTriQSM(p=p_new, q=q_new, a=K.noise_model.lower.a)

                new_noise_model = SymmQSM(diag=diag_term, lower=lower_term)
            else:
                new_noise_model = None

            K_mul = GeneralQuasisepPlusNoise(new_kf, diag = K.diag * self.alpha_init**2, wn_diag = K.wn_diag * self.alpha_init**2,
                                            use_block = K.use_block, noise_model = new_noise_model)
        
        elif isinstance(K, (Identity, ScaledIdentity, Diagonal)):
            K_mul = Diagonal(diag = K.diag * self.alpha_init**2, wn_diag = K.wn_diag * self.alpha_init**2)
        
        elif isinstance(K, Outer):
            K_mul = Outer(alpha = self.alpha_init * K.alpha_init)
        
        elif isinstance(K, General):
            K_mul = General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * K.evaluate(x1, x2, **kwargs))
        
        else:
            raise Exception(f"Multiplication of Outer with type {type(K)} not implemented")
        
        return K_mul


class OuterPlusScaledIdentity(CovType):
    # TO DO: Surely better to do a Sherman-Morrison CovType instead...
    def __init__(self, alpha, diag = 0., wn_diag = 0., params = None):
        self.alpha_init = alpha
        self.diag = diag
        self.wn_diag = wn_diag
        self.params = params
        self.opt_name = "Rank 1 matrix plus noise"

        if is_scalar(self.alpha_init):
            self.tinygp_kf = HandleIdx(luas.kernels.tinygp_ext.Constant(const = self.alpha_init**2))
        else:
            const_kf = HandleIdx(luas.kernels.tinygp_ext.Constant(1.))
            self.tinygp_kf = ScaledKernel(kernel = const_kf, amplitudes = self.alpha_init)

        assert is_scalar(self.alpha_init) or self.alpha_init.ndim == 1
        assert is_scalar(self.diag)
        assert is_scalar(self.wn_diag)
    
    def evaluate(self, x1, x2, wn = True, row_idx = None, col_idx = None, full = True, **kwargs):

        if full:
            diag_mat = (self.diag + wn*self.wn_diag) * jnp.eye(x1.shape[0])
        else:
            idx_specified = (row_idx is not None) and (col_idx is not None)

            if idx_specified:
                mask = row_idx[:, None] == col_idx[None, :]  # (n1, n2) bool matrix

                # Pick the diagonal values where they match
                diag_val = self.diag + wn*self.wn_diag
                diag_mat = jnp.where(mask, diag_val, 0.0)
                
            else:
                raise Exception("""Cannot evaluate non-stationary diagonal covariance without specifying matrix indices to evaluate.
                                Either specify the row and col indices by the keyword arguments row_idx and col_idx
                                or specify that the full covariance matrix is being evaluated by setting full = True"""
                            )
            
        rank1_mat = jnp.outer(self.alpha_init * jnp.ones(x1.shape[-1]), self.alpha_init * jnp.ones(x2.shape[-1]))
        
        return rank1_mat + diag_mat

    def decompose(self, x, wn = True, i = -1, **kwargs):

        self.alpha = self.alpha_init * jnp.ones(x.shape[-1])
        self.H = HouseholderTransform(self.alpha, i = i)

        # Shift eigenvalues by constant added to diagonal
        self.lam = self.H.lam + (self.diag + wn * self.wn_diag)

        self.factor = HouseholderandDiagonalTransform(self.alpha, jnp.sqrt(self.lam), i = i, transpose = 0)
        
        self.logdet = jnp.log(self.lam).sum()
        return self, {"logdet":self.logdet}

    def matrix_inv_sqrt(self, R, transpose=0, **kwargs):

        if transpose:
            R_prime = self.factor.T.solve(R)
        else:
            R_prime = self.factor.solve(R)
    
        return R_prime

    
    def matrix_sqrt(self, R, transpose=0, **kwargs):

        if transpose:
            R_prime = self.factor.T @ R
        else:
            R_prime = self.factor @ R

        return R_prime

    def eigendecompose(self, x, wn = True, i = -1, **kwargs):

        self.H = HouseholderTransform(self.alpha, i = i)
        self.lam = self.H.lam + (self.diag + wn * self.wn_diag)
        self.logdet = jnp.log(self.lam).sum()
            
        return self.lam, self.H
    

    def householder_transform(self, x, u):
        K_general = General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs))
        
        return K_general.householder_transform(x, u)


    def matmul(self, x1, x2, other, wn = True, **kwargs):
        self.vec1 = self.alpha_init * jnp.ones(x1.shape[-1])
        self.vec2 = self.alpha_init * jnp.ones(x2.shape[-1])

        D = self.diag + wn * self.wn_diag
        
        if other.ndim == 1:
            return D * other + jnp.kron(self.vec1, self.vec2 @ other)
        elif other.ndim == 2:
            return D * other + jnp.outer(self.vec1, self.vec2 @ other)
        else:
            raise Exception("Not implemented")
        
    def scale(self, c):
        return OuterPlusScaledIdentity(alpha = self.alpha_init * jnp.sqrt(c), diag = self.diag * c, wn_diag = self.wn_diag * c)

    def __add__(self, K):
        if isinstance(K, (Identity, ScaledIdentity)):
            K_sum = OuterPlusScaledIdentity(self.alpha_init, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)

        elif isinstance(K, (GeneralQuasisepPlusNoise, GeneralQuasisep)):
            if is_scalar(self.alpha_init):
                outer_kf = luas.kernels.tinygp_ext.Constant(self.alpha_init**2)
            else:
                outer_kf = luas.kernels.tinygp_ext.Linear(self.alpha_init)

            if K.tinygp_kf is not None:
                new_tinygp_kf = luas.kernels.tinygp_ext.Sum(outer_kf, K.tinygp_kf, use_block = K.use_block)
            else:
                new_tinygp_kf = outer_kf

            K_sum = GeneralQuasisepPlusNoise(new_tinygp_kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag,
                                             noise_model = K.noise_model,  use_block = K.use_block)
            
        elif isinstance(K, (Diagonal, Outer, OuterPlusScaledIdentity, General)):
            K_sum = General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))

        else:
            raise Exception("Addition of kernels not implemented")
        return K_sum

    def __mul__(self, other) -> Kernel:
        if is_scalar(other):
            return self.scale(other)
        else:
            raise Exception("Not implemented")