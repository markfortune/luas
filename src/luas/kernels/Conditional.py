import jax.numpy as jnp
from luas.kernels.covtype import CovType
from luas.kronecker_fns import calc_data_shape

__all__ = ["Conditional"]

class Conditional(CovType):
    r"""Covariance of the GP conditional (predictive) distribution at the prediction locations, given the data at
    the observed locations. Created by :meth:`luas.GP.predict`.

    The observed and prediction locations are stacked into one full set of locations, with covariance matrix
    ``kf_full``. The conditional covariance at the prediction locations is
    :math:`\mathbf{K}_{**} - \mathbf{K}_*^T\mathbf{K}^{-1}\mathbf{K}_*`, and its inverse is the block of the
    inverse of the full covariance matrix at the prediction locations, so everything is calculated from the
    decompositions of ``kf_full`` and ``kf_train`` without building the conditional covariance matrix.

    The matrix square root used here is not square: it maps an array with the shape of the full set of
    locations (``sqrt_input_shape``) to the prediction locations, which is all that is needed to draw samples.

    Warning:
        Internal to ``luas`` and not intended to be used directly. The interface is not stable and may change
        without notice.

    Args:
        kf_full (CovType): The covariance object for the full set of observed and prediction locations.
        kf_train (CovType, optional): The decomposed covariance object for the observed locations.
        X_full (JAXArray | tuple, optional): The full set of input locations.
        observed_ind (optional): Indices of the observed locations in the full data array (from ``jnp.ix_``
            for 2D data).
        predicted_ind (optional): Indices of the prediction locations in the full data array.
        logdet_K_train (Scalar, optional): Log determinant of the covariance matrix at the observed locations.

    """
    def __init__(self, kf_full, kf_train = None, X_full = None, 
                 observed_ind = None, predicted_ind = None, logdet_K_train = None):
        r"""Creates the covariance object, see the class docstring for the arguments."""

        self.kf_full = kf_full
        self.kf_train = kf_train
        self.X_full = X_full
        self.N_full = calc_data_shape(X_full)
        self.observed_ind = observed_ind
        self.predicted_ind = predicted_ind
        self.logdet_K_train = logdet_K_train
        self.opt_name = f"GP conditional object with {self.kf_full.opt_name}"

        # Shape of the random normal vector matrix_sqrt expects (used by GP.sample)
        self.sqrt_input_shape = tuple(self.N_full)


    def evaluate(self, X1, X2, **kwargs):
        r"""Building the predictive covariance matrix, which is not implemented.

        Raises:
            NotImplementedError: Always.

        """
        raise NotImplementedError("Building the GP predictive covariance matrix not currently implemented.")

    def decompose(self, X_full, **kwargs):
        r"""Decomposes ``kf_full``. The log determinant of the conditional covariance matrix is
        :math:`\log|\mathbf{K}_{\mathrm{full}}| - \log|\mathbf{K}|`.

        Args:
            X_full (JAXArray | tuple): The full set of input locations.

        Returns:
            (Conditional, dict): Itself and the stored values of ``kf_full``, with ``"logdet"`` replaced by the
            conditional log determinant.

        """
        self.kf_full, stored_values_full = self.kf_full.decompose(X_full)
        self.logdet = stored_values_full["logdet"] - self.logdet_K_train

        stored_values_full["logdet"] = self.logdet
        return self, stored_values_full

    def inverse(self, R_pred):
        r"""Multiplies ``R_pred`` by the inverse of the conditional covariance matrix, by padding it with zeros at the
        observed locations, multiplying by the inverse of the full covariance matrix and keeping the prediction
        locations.

        """
        R_full = jnp.zeros(self.N_full)
        R_full = R_full.at[self.predicted_ind].set(R_pred)
        return self.kf_full.inverse(R_full)[self.predicted_ind]

    def dot_solve(self, R_pred):
        r"""Calculates :math:`\vec{r}^T \mathbf{K}_{\mathrm{cond}}^{-1} \vec{r}` for an array ``R_pred`` at the prediction locations."""
        K_inv_R = self.inverse(R_pred)
        return (R_pred * K_inv_R).sum()
    
    def matrix_inv_sqrt(self, R, transpose = 0):
        r"""Multiplies by a (non-square) inverse matrix square root of the conditional covariance matrix: ``R`` at the
        prediction locations is padded with zeros and multiplied by :math:`\mathbf{L}_{\mathrm{full}}^{-1}`. With
        ``transpose = 1`` it maps an array of the full shape back to the prediction locations.

        """
        if transpose:
            # require R to be of size (N_train + N_pred,) but returns size (N_pred,)
            return self.kf_full.matrix_inv_sqrt(R, transpose = 1)[self.predicted_ind]
        else:
            # Takes R as size (N_pred,) but returns size (N_train + N_pred,)
            R_full = jnp.zeros(self.N_full)
            R_full = R_full.at[self.predicted_ind].set(R)
            return self.kf_full.matrix_inv_sqrt(R_full)

    def _Ks_matmul(self, R_pred):
        r"""Multiplies an array at the prediction locations by :math:`\mathbf{K}_*`, giving an array at the observed locations."""
    
        R_full = jnp.zeros(self.N_full)
        R_full = R_full.at[self.predicted_ind].set(R_pred)
        return self.kf_full.matmul(self.X_full, self.X_full, R_full)[self.observed_ind]

    def _Ks_transp_matmul(self, R_obs):
        r"""Multiplies an array at the observed locations by :math:`\mathbf{K}_*^T`, giving an array at the prediction locations."""

        R_full = jnp.zeros(self.N_full)
        R_full = R_full.at[self.observed_ind].set(R_obs)
        return self.kf_full.matmul(self.X_full, self.X_full, R_full)[self.predicted_ind]

    def matrix_sqrt(self, R, transpose = 0):
        r"""Multiplies by a (non-square) matrix square root of the conditional covariance matrix. With
        ``transpose = 0``, an array ``R`` with the full shape (``sqrt_input_shape``) is mapped to
        :math:`\vec{y}_* - \mathbf{K}_*^T\mathbf{K}^{-1}\vec{y}`, where :math:`\vec{y} = \mathbf{L}_{\mathrm{full}}\mathbf{R}`
        is split into its observed and predicted parts. For ``R`` drawn from a standard normal distribution this is
        a sample from the conditional distribution. ``transpose = 1`` applies the transpose.

        """
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
        r"""Multiplies ``R_pred``, an array at the prediction locations, by the conditional covariance matrix without
        building it.

        Args:
            X1, X2: Not used, the stored full set of locations is used.
            R_pred (JAXArray): The array to multiply.
            full (bool, optional): Must be ``True``.
            **kwargs: Passed to ``kf_full.matmul``, e.g. ``wn``.

        Returns:
            JAXArray: The product at the prediction locations.

        """
        if not full:
            raise Exception("matmul for Conditional covariance matrix only defined for full = True\
                            i.e. multiplying by the full predictive covariance matrix")

        R_full = jnp.zeros(self.N_full)
        R_full = R_full.at[self.predicted_ind].set(R_pred)

        Ks_R_pred = self._Ks_matmul(R_pred)
        K_inv_Ks_R_pred = self.kf_train.inverse(Ks_R_pred)
        R_full = R_full.at[self.observed_ind].set(-K_inv_Ks_R_pred)

        return self.kf_full.matmul(self.X_full, self.X_full, R_full, **kwargs)[self.predicted_ind]