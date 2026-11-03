import jax.numpy as jnp
from luas.kernels.covtype import CovType

__all__ = ["Conditional"]

class Conditional(CovType):
    def __init__(self, kf_full, kf_train = None, X_full = None, 
                 observed_ind = None, predicted_ind = None, logdet_K_train = None):

        self.kf_full = kf_full
        self.kf_train = kf_train
        self.X_full = X_full
        self.N_full = (X_full[0].shape[-1], X_full[1].shape[-1])
        self.observed_ind = observed_ind
        self.predicted_ind = predicted_ind
        self.logdet_K_train = logdet_K_train
        self.opt_name = f"GP conditional object with {self.kf_full.opt_name}"

        # Shape of the random normal vector matrix_sqrt expects (used by GP.sample)
        self.sqrt_input_shape = tuple(self.N_full)


    def evaluate(self, X1, X2, **kwargs):
        raise NotImplementedError("Building the GP predictive covariance matrix not currently implemented.")

    def decompose(self, X_full, **kwargs):
        self.kf_full, stored_values_full = self.kf_full.decompose(X_full)
        self.logdet = stored_values_full["logdet"] - self.logdet_K_train

        stored_values_full["logdet"] = self.logdet
        return self, stored_values_full

    def inverse(self, R_pred):
        R_full = jnp.zeros(self.N_full)
        R_full = R_full.at[self.predicted_ind].set(R_pred)
        return self.kf_full.inverse(R_full)[self.predicted_ind]

    def dot_solve(self, R_pred):
        K_inv_R = self.inverse(R_pred)
        return (R_pred * K_inv_R).sum()
    
    def matrix_inv_sqrt(self, R, transpose = 0):
        if transpose:
            # require R to be of size (N_train + N_pred,) but returns size (N_pred,)
            return self.kf_full.matrix_inv_sqrt(R, transpose = 1)[self.predicted_ind]
        else:
            # Takes R as size (N_pred,) but returns size (N_train + N_pred,)
            R_full = jnp.zeros(self.N_full)
            R_full = R_full.at[self.predicted_ind].set(R)
            return self.kf_full.matrix_inv_sqrt(R_full)

    def _Ks_matmul(self, R_pred):
    
        R_full = jnp.zeros(self.N_full)
        R_full = R_full.at[self.predicted_ind].set(R_pred)
        return self.kf_full.matmul(self.X_full, self.X_full, R_full)[self.observed_ind]

    def _Ks_transp_matmul(self, R_obs):

        R_full = jnp.zeros(self.N_full)
        R_full = R_full.at[self.observed_ind].set(R_obs)
        return self.kf_full.matmul(self.X_full, self.X_full, R_full)[self.predicted_ind]

    def matrix_sqrt(self, R, transpose = 0):
        if transpose:
            # require R to be of size (N_pred,) but returns size (N_train + N_pred,)
            R_full = jnp.zeros(self.N_full)
            R_full = R_full.at[self.predicted_ind].set(R)

            Ks_R_pred = self._Ks_matmul(R)
            K_inv_Ks_R_pred = self.kf_train.inverse(Ks_R_pred)
            R_full = R_full.at[self.observed_ind].set(-K_inv_Ks_R_pred)

            return self.kf_full.matrix_sqrt(R_full, transpose = 1)
        else:
            # Takes R as size (N_train + N_pred,) but returns size (N_pred,)
            R_full = self.kf_full.matrix_sqrt(R, transpose = 0)

            K_inv_R_obs = self.kf_train.inverse(R_full[self.observed_ind])
            Ks_transp_K_inv_R_obs = self._Ks_transp_matmul(K_inv_R_obs)

            return R_full[self.predicted_ind] - Ks_transp_K_inv_R_obs

    def matmul(self, X1, X2, R_pred, full = True, **kwargs):
        if not full:
            raise Exception("matmul for Conditional covariance matrix only defined for full = True\
                            i.e. multiplying by the full predictive covariance matrix")

        R_full = jnp.zeros(self.N_full)
        R_full = R_full.at[self.predicted_ind].set(R_pred)

        Ks_R_pred = self._Ks_matmul(R_pred)
        K_inv_Ks_R_pred = self.kf_train.inverse(Ks_R_pred)
        R_full = R_full.at[self.observed_ind].set(-K_inv_Ks_R_pred)

        return self.kf_full.matmul(self.X_full, self.X_full, R_full, **kwargs)[self.predicted_ind]