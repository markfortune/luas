import numpy as np
import jax.numpy as jnp
from luas.kernels.covtype import CovType

__all__ = ["ConditionalKernel"]

class ConditionalKernel(CovType):
    def __init__(self, kf_total, N_total = None, predicted_ind = None, logdet_K_train = None):

        self.kf_total = kf_total
        self.N_total = N_total
        self.predicted_ind = predicted_ind
        self.logdet_K_train = logdet_K_train
        self.opt_name = f"GP conditional object with {self.kf_total.opt_name}"

    def evaluate(self, X1, X2):
        raise NotImplementedError("Building the GP predictive covariance matrix not currently implemented.")


    def decompose(self, X_total, **kwargs):

        self.kf_total, stored_values_total = self.kf_total.decompose(X_total)
        self.logdet = stored_values_total["logdet"] - self.logdet_K_train

        stored_values_total["logdet"] = self.logdet
        return self, stored_values_total


    def matrix_sqrt(self, R, **kwargs):

        R_pred = jnp.zeros(self.N_total)
        R_pred = R_pred.at[self.predicted_ind].set(R)
        return self.kf_total.matrix_sqrt(R_pred, **kwargs)[self.predicted_ind]


    def matrix_inv_sqrt(self, R, **kwargs):
    
        R_pred = jnp.zeros(self.N_total)
        R_pred = R_pred.at[self.predicted_ind].set(R)
        return self.kf_total.matrix_inv_sqrt(R_pred, **kwargs)[self.predicted_ind]


    def matmul(self, x1, x2, R, full = True, wn = True, **kwargs):
        if not full:
            raise NotImplementedError("")

        if wn == False:
            R_pred = jnp.zeros(self.N_total)
            R_pred = R_pred.at[self.predicted_ind].set(R)
            R_prime = self.kf_total.matmul(x1, x2, R_pred, full = True, wn = False)[self.predicted_ind]

            self, _ = self.decompose(x1, full = full, **kwargs)

            min_K_inv_K_s_R = self.kf_total.matrix_sqrt(R_pred, transpose = 1)
            min_K_inv_K_s_R = min_K_inv_K_s_R.at[self.predicted_ind].set(0.)
            R_prime += self.kf_total.matmul(x1, x2, min_K_inv_K_s_R, full = True, wn = True)[self.predicted_ind]
        else:
            self, _ = self.decompose(x1, full = full, **kwargs)
            
            R_prime = self.matrix_sqrt(R, transpose = 1)
            R_prime = self.matrix_sqrt(R_prime, transpose = 0)
        return R_prime