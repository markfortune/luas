import jax.numpy as jnp
from luas.kernels.covtype import CovType

__all__ = ["ConditionalKernel"]


class ConditionalKernel(CovType):
    r"""Covariance of a GP conditioned on observed (training) data, evaluated at held-out locations.

    Returned by :meth:`luas.GP.predict` so that the held-out data can be treated as a normal
    :class:`luas.GP` object (e.g. for ``gp.plot``, ``gp.sample`` or ``gp.logL``).

    Let the observed locations be indexed by ``o`` and the predicted/held-out locations by ``*``,
    with the joint covariance matrix over all locations ``X_total``

    .. math::

        \mathbf{K}_\mathrm{total} = \begin{pmatrix} \mathbf{K} & \mathbf{K}_* \\
                                     \mathbf{K}_*^T & \mathbf{K}_{**} \end{pmatrix}.

    This object represents the predictive (conditional) covariance

    .. math::

        \mathbf{\Sigma} = \mathbf{K}_{**} - \mathbf{K}_*^T \mathbf{K}^{-1} \mathbf{K}_*,

    without ever building it. Only a decomposition of ``K_total`` (the luas-optimised kernel over
    ``X_total``) and of the training covariance ``K`` are needed, using these exact identities:

    * Inverse: the ``(*, *)`` block of ``K_total^{-1}`` is ``Σ^{-1}`` (inverse of a Schur complement), so
      ``Σ^{-1} r = [K_total^{-1} (0, r)]_*``.
    * Log determinant: ``log|Σ| = log|K_total| - log|K|``.
    * Matrix product: ``Σ r = [K_total (-u, r)]_*`` with ``u = K^{-1} [K_total (0, r)]_o``.
    * Square root (Matheron's rule): if ``y = L_total z`` with ``z ~ N(0, I)`` over ``X_total``,
      then ``y_* - K_*^T K^{-1} y_o ~ N(0, Σ)``.

    Note that the matrix (inverse) square roots here are *not* square ``N_pred x N_pred`` matrices:
    ``matrix_sqrt`` takes a vector of shape ``N_total`` and returns one of the predicted shape,
    ``matrix_inv_sqrt`` does the reverse. Simply extracting the ``(*, *)`` block of a square root of
    ``K_total`` is only valid for a block-Cholesky factor with all observed points ordered first,
    which is not the case for the eigendecomposition-based roots used by most luas kernels
    or when observed and predicted points are interleaved after sorting.

    Args:
        kf_total (CovType): Kernel object (e.g. :class:`LuasKernel`) for the joint covariance
            over all locations ``X_total``.
        kf_train (CovType): Kernel object for the observed locations which has already been
            decomposed (i.e. ``kf_train.decompose(X_obs)`` has been called). Must correspond to the
            ``(o, o)`` block of ``kf_total``.
        N_total (tuple): Shape of the data array over all locations ``X_total``.
        observed_ind (tuple): Index (from ``jnp.ix_``) of the observed locations within ``X_total``.
        predicted_ind (tuple): Index (from ``jnp.ix_``) of the predicted locations within ``X_total``.
            Must not overlap with ``observed_ind``.
        logdet_K_train (Scalar): Log determinant of the training covariance matrix ``K``.
    """

    def __init__(self, kf_total, kf_train = None, N_total = None,
                 observed_ind = None, predicted_ind = None, logdet_K_train = None):

        self.kf_total = kf_total
        self.kf_train = kf_train
        self.N_total = N_total
        self.observed_ind = observed_ind
        self.predicted_ind = predicted_ind
        self.logdet_K_train = logdet_K_train
        self.opt_name = f"GP conditional object with {self.kf_total.opt_name}"

        # Shape of the random normal vector matrix_sqrt expects (used by GP.sample)
        self.sqrt_input_shape = tuple(N_total)

    # ---- helpers for moving between the predicted grid and the full X_total grid ----

    def _embed_pred(self, R):
        """Place an array of the predicted shape into a zero array of shape ``N_total``."""
        return jnp.zeros(self.N_total, dtype = R.dtype).at[self.predicted_ind].set(R)

    def _embed_obs(self, R):
        """Place an array of the observed shape into a zero array of shape ``N_total``."""
        return jnp.zeros(self.N_total, dtype = R.dtype).at[self.observed_ind].set(R)

    def _K_inv_Ks(self, R_total):
        """Computes ``K^{-1} [K_total R_total]_o`` i.e. ``K^{-1} K_* r`` if ``R_total = (0, r)``."""
        K_R = self.kf_total.matmul(self.X_total, self.X_total, R_total, full = True, wn = True)
        return self.kf_train.inverse(K_R[self.observed_ind])

    # ---- CovType interface ----

    def evaluate(self, X1, X2, **kwargs):
        raise NotImplementedError("Building the GP predictive covariance matrix not currently implemented.")

    def decompose(self, X_total, **kwargs):
        """Decomposes ``K_total`` and computes ``log|Σ| = log|K_total| - log|K|``."""

        self.X_total = X_total
        self.kf_total, stored_values_total = self.kf_total.decompose(X_total)
        self.logdet = stored_values_total["logdet"] - self.logdet_K_train

        stored_values_total["logdet"] = self.logdet
        return self, stored_values_total

    def inverse(self, R, **kwargs):
        """Computes ``Σ^{-1} R`` exactly as the ``(*, *)`` block of ``K_total^{-1}`` applied to ``R``."""
        return self.kf_total.inverse(self._embed_pred(R))[self.predicted_ind]

    def dot_solve(self, R):
        """Computes ``R^T Σ^{-1} R`` used in the log likelihood."""
        return (R * self.inverse(R)).sum()

    def matrix_inv_sqrt(self, R, transpose = 0, **kwargs):
        """Applies a (non-square) inverse square root ``W = L_total^{-1} P_*^T`` with ``W^T W = Σ^{-1}``.

        ``transpose = 0`` maps the predicted shape to shape ``N_total``,
        ``transpose = 1`` maps shape ``N_total`` back to the predicted shape.
        """
        if transpose:
            return self.kf_total.matrix_inv_sqrt(R, transpose = 1)[self.predicted_ind]
        else:
            return self.kf_total.matrix_inv_sqrt(self._embed_pred(R), transpose = 0)

    def matrix_sqrt(self, R, transpose = 0, **kwargs):
        """Applies a (non-square) square root ``A`` of ``Σ`` with ``A A^T = Σ`` (Matheron's rule).

        ``transpose = 0`` takes a vector of shape ``N_total`` (e.g. ``z ~ N(0, I)``) and returns the
        predicted shape: ``A z = y_* - K_*^T K^{-1} y_o`` where ``y = L_total z``.
        ``transpose = 1`` maps the predicted shape to shape ``N_total``.
        """
        if transpose:
            R_total = self._embed_pred(R)
            u = self._K_inv_Ks(R_total)
            return self.kf_total.matrix_sqrt(R_total - self._embed_obs(u), transpose = 1)
        else:
            y = self.kf_total.matrix_sqrt(R, transpose = 0)
            u = self.kf_train.inverse(y[self.observed_ind])
            K_s_u = self.kf_total.matmul(self.X_total, self.X_total, self._embed_obs(u), full = True, wn = True)
            return y[self.predicted_ind] - K_s_u[self.predicted_ind]

    def matmul(self, x1, x2, R, full = True, wn = True, **kwargs):
        r"""Computes ``Σ R`` where for ``wn = False`` the white noise is removed from the prediction locations.

        ``Σ R = [K_total (-u, R)]_*`` with ``u = K^{-1} K_* R``. The cross-covariance ``K_*`` used for
        ``u`` always includes white noise as ``K`` is the covariance of the noisy training data.
        """
        if not full:
            raise NotImplementedError("Only full = True is implemented for ConditionalKernel.matmul")

        self.X_total = x1
        R_total = self._embed_pred(R)
        u = self._K_inv_Ks(R_total)
        Sigma_R = self.kf_total.matmul(x1, x2, R_total - self._embed_obs(u), full = True, wn = wn)
        return Sigma_R[self.predicted_ind]
