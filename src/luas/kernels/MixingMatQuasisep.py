from luas.kernels.tinygp_ext import Multiband
from luas.kernels.covtype import CovType
import luas.kernels.quasisep
from luas.luas_types import is_scalar
import jax.numpy as jnp
import tinygp

class MixingMatQuasisep(CovType):
    def __init__(self, mixing_mat, kernel_list, noise_model = None,
                 diag = 0., wn_diag = 0., params = None, fast_dim = 1, use_block = True, **kwargs):
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
        if self.fast_dim == 0:
            return R.ravel("C")
        else:
            return R.ravel("F")

    def _reshape_R(self, r, R_shape):
        if self.fast_dim == 0:
            return r.reshape(R_shape, order='C')
        else:
            return r.reshape(R_shape, order='F')
        

    def _tinygp_coords(self, X, full = False, idx = (None, None)):
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

        self.quasi_mat = self._to_symm_qsm(X, wn = wn, **kwargs)
        self.factor = self.quasi_mat.cholesky()
        self.logdet = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
        return self, {"logdet":self.logdet}
        

    def evaluate(self, X1, X2, row_idx = None, col_idx = None, full = False, wn = True, flip_kron = False, calc_diag = True):

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
                raise Exception("Not Implemented!")

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
        
        R_shape = R.shape
        r = self._flatten_R(R)

        if transpose:
            r_prime = self.factor.transpose().solve(r)
        else:
            r_prime = self.factor.solve(r)

        R_prime = self._reshape_R(r_prime, R_shape)
            
        return R_prime

    def matrix_sqrt(self, R, transpose=0, **kwargs):
        
        R_shape = R.shape
        r = self._flatten_R(R)

        if transpose:
            r_prime = self.factor.transpose() @ r
        else:
            r_prime = self.factor @ r

        R_prime = self._reshape_R(r_prime, R_shape)
            
        return R_prime

    def inv_sqrt_transform(self, K):
        raise Exception("Not implemented!")

    def scale(self, c):
        
        if self.noise_model is not None:
            new_noise_model = self.noise_model * c
        else:
            new_noise_model = None

        # c should be positive!
        
        return MixingMatQuasisep(self.mixing_mat * jnp.sqrt(c), self.kernel_list, diag = self.diag * c, wn_diag = self.wn_diag * c,
                                 fast_dim = self.fast_dim, noise_model = new_noise_model)

    def matmul(self, x1, x2, R, wn = True, **kwargs):

        R_shape = R.shape
        r = self._flatten_R(R)

        self.quasi_mat = self._to_symm_qsm(x1, wn = wn)
        r_prime = self.quasi_mat @ r
        
        R_prime = self._reshape_R(r_prime, R_shape)
            
        return R_prime
    

    def householder_transform(self, X, u):
        
        for i in range(self.rank):
            w_i = self.kernel_list[i].matmul(X[self.fast_dim], X[self.fast_dim], u, full = True)

            # Can this ever be super close to zero? Seems unlikely?
            u_dot_w = jnp.dot(u, w_i)
            alpha = 2 * u_dot_w * u - w_i
            
            self.kernel_list[i] += luas.kernels.quasisep.Amplitude(alpha, const = 1/u_dot_w)
            self.kernel_list[i] += luas.kernels.quasisep.Amplitude(w_i, const = -1/u_dot_w)
        
        return MixingMatQuasisep(self.mixing_mat, self.kernel_list, diag = self.diag, wn_diag = self.wn_diag,
                                 fast_dim = self.fast_dim, noise_model = self.noise_model)
    

    def __add__(self, K):

        if isinstance(K, MixingMatQuasisep):
            assert self.mixing_mat.shape[0] == K.mixing_mat.shape[0]

            new_mixing_mat = jnp.concatenate([self.mixing_mat, K.mixing_mat], axis = 1)
            new_kernel_list = self.kernel_list + K.kernel_list

            if self.noise_model is not None and K.noise_model is not None:
                new_noise_model = self.noise_model + K.noise_model
            elif self.noise_model is not None:
                new_noise_model = self.noise_model
            elif K.noise_model is not None:
                new_noise_model = K.noise_model
            else:
                new_noise_model = None
                
            K_sum = MixingMatQuasisep(new_mixing_mat, new_kernel_list, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag,
                                      fast_dim = self.fast_dim, noise_model = new_noise_model)
            
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported")
            
        return K_sum
    

