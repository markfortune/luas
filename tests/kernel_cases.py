"""Kernels and data shared across the test modules.

ALGEBRA_COVTYPES: one instance of each 1D CovType (not all invertible), used to
    test + and * between every pair.
PD_COVTYPES: invertible 1D CovTypes, used to test decompositions and solves.
KERNEL_FNS_1D: 1D GP kernel functions from the same tutorial.
KERNEL_FNS_2D: 2D kernel functions (from Walking_through_luas_optimisations.ipynb) which between
    them exercise every 2D optimisation the kernel selector can choose.
"""
import jax
import jax.numpy as jnp
import numpy as np

import luas
from luas.kernels import covtype as ct, quasisep as q, lowrank as lr, base
from luas.kernels.lowrank import LowRank

K = luas.kernels

# --------------------------------------------------------------------------- 1D

N = 12
x = jnp.linspace(0., 3., N)
_rng = np.random.default_rng(0)
vec = jnp.array(_rng.uniform(0.5, 1.5, N))
alpha = jnp.array(_rng.normal(size=N))
_B = jnp.array(_rng.normal(size=(N, 2)))


def algebra_covtypes():
    return {
        "General": base.SquaredExp(scale=0.7, sigma=1.1),
        "Identity": ct.Identity(),
        "ScaledIdentity": ct.ScaledIdentity(diag=0.3, wn_diag=0.1),
        "Diagonal": ct.Diagonal(diag=vec, wn_diag=0.1),
        "GeneralQuasisep": q.Matern32(scale=0.8, sigma=0.9),
        "Exp": q.Exp(scale=0.6, sigma=1.2),
        "GQPN": ct.GeneralQuasisep(q.Matern52(0.5).tinygp_kf) + ct.ScaledIdentity(diag=0.2),
        "GQPN_noise": q.Banded(diag=vec, off_diags=0.1 * vec[:, None]) + q.Matern32(0.8),
        "Outer_vec": ct.Outer(alpha),
        "Outer_scalar": ct.Outer(0.7),
        "OPSI": ct.OuterPlusScaledIdentity(alpha, diag=0.4),
        "LowRank": LowRank(lambda hp, x1, x2, **kw: _B @ _B.T, rank=2),
    }


def pd_covtypes():
    return {
        "General": base.SquaredExp(scale=0.7, sigma=1.1) + ct.ScaledIdentity(diag=0.1),
        "Identity": ct.Identity(),
        "ScaledIdentity": ct.ScaledIdentity(diag=0.3, wn_diag=0.1),
        "Diagonal": ct.Diagonal(diag=vec, wn_diag=0.1),
        "GeneralQuasisep": q.Matern52(0.5),
        "GQPN": q.Matern32(scale=0.8, sigma=0.9) + ct.ScaledIdentity(diag=0.2),
        "Exp+Diagonal": q.Exp(scale=0.6, sigma=1.2) + ct.Diagonal(diag=vec),
        "GQPN_noise": q.Banded(diag=vec + 1, off_diags=0.1 * vec[:, None]) + q.Matern32(0.8),
        "OPSI": ct.OuterPlusScaledIdentity(alpha, diag=0.4),
    }


ALGEBRA_NAMES = list(algebra_covtypes())
PD_NAMES = list(pd_covtypes())

# --------------------------------------------------------------------------- 2D
# Kernel functions and hyperparameter values follow
# docs/source/tutorials/Walking_through_luas_optimisations.ipynb, on a smaller grid.

N_l, N_t = 12, 20     # N_l must be at least 10 (kf_multilowrank uses light curve index 9)
x_l = np.linspace(4., 5., N_l)          # numpy, as in the tutorial
x_t = jnp.linspace(-0.15, 0.15, N_t)
X = (x_l, x_t)

h_l_vec = 200e-6 * (x_l - x_l.mean())**2

P = {
    "log_sigma": np.log(100e-6) * np.ones(N_l),
    "log_h_readout": np.log(30e-6),
    "log_l_readout": np.log(0.3),
    "log_h": np.log(80e-6),
    "log_l_l": np.log(0.05),
    "log_l_t": np.log(0.04),
    "log_h_l": np.log(h_l_vec),
    "log_h_l_vec": np.log(h_l_vec),
    "log_h_const": np.log(80e-6),
    "log_l_t1": np.log(0.04),
    "log_h10": np.log(100e-6),
    "log_l_t2": np.log(0.02),
    "log_h_ramp": np.log(200e-6),
    "log_l_l_ramp": np.log(0.4),
    "log_l_t_ramp": np.log(0.03),
}

# Fixed residuals with roughly the same amplitude as the noise
Y = jnp.array(np.random.default_rng(1).normal(size=(N_l, N_t))) * 1e-4


def kf_readout(p, X):
    s_l = K.Noise(jnp.exp(p["log_sigma"]))
    s_l += jnp.exp(2*p["log_h_readout"]) * K.Exp(jnp.exp(p["log_l_readout"]))
    s_t = K.KroneckerDelta(1.)
    return (s_l, s_t)


def kf_readout_quasisep(p, X):
    s_l = q.Noise(jnp.exp(p["log_sigma"]))
    s_l += jnp.exp(2*p["log_h_readout"]) * q.Exp(jnp.exp(p["log_l_readout"]))
    s_t = q.KroneckerDelta(1.)
    return (s_l, s_t)


def kf_with_timecorr(p, X):
    s_l = K.Noise(jnp.exp(p["log_sigma"]))
    s_l += jnp.exp(2*p["log_h_readout"]) * K.Exp(jnp.exp(p["log_l_readout"]))
    s_t = K.KroneckerDelta()
    k_l = K.Constant(jnp.exp(p["log_h"]))
    k_t = K.Matern32(jnp.exp(p["log_l_t"]))
    return (s_l, s_t), (k_l, k_t)


def kf_timecorr_fast(p, X):
    s_l = q.Noise(jnp.exp(p["log_sigma"]))
    s_l += jnp.exp(2*p["log_h_readout"]) * q.Exp(jnp.exp(p["log_l_readout"]))
    s_t = q.KroneckerDelta()
    k_l = lr.Constant(jnp.exp(p["log_h"]))
    k_t = q.Matern32(jnp.exp(p["log_l_t"]))
    return (s_l, s_t), (k_l, k_t)


def kf_full2d_slow(p, X):
    s_l = K.Noise(jnp.exp(p["log_sigma"]))
    s_l += jnp.exp(2*p["log_h_readout"]) * K.Exp(jnp.exp(p["log_l_readout"]))
    s_t = K.KroneckerDelta()
    k_l = jnp.exp(2*p["log_h"]) * K.Matern52(jnp.exp(p["log_l_l"]))
    k_t = K.Matern32(jnp.exp(p["log_l_t"]))
    return (s_l, s_t), (k_l, k_t)


def kf_full2d_fast(p, X):
    s_l = q.Noise(jnp.exp(p["log_sigma"]))
    s_l += jnp.exp(2*p["log_h_readout"]) * q.Exp(jnp.exp(p["log_l_readout"]))
    s_t = q.KroneckerDelta()
    k_l = jnp.exp(2*p["log_h"]) * K.Matern52(jnp.exp(p["log_l_l"]))
    k_t = q.Matern32(jnp.exp(p["log_l_t"]))
    return (s_l, s_t), (k_l, k_t)


def kf_scaled2Dkernel(p, X):
    s_l = q.Noise(jnp.exp(p["log_sigma"]))
    s_l += jnp.exp(2*p["log_h_readout"]) * q.Exp(jnp.exp(p["log_l_readout"]))
    s_t = q.KroneckerDelta()
    k_l = lr.Amplitude(jnp.exp(p["log_h_l"])) * K.Matern52(jnp.exp(p["log_l_l"]))
    k_t = q.Matern32(jnp.exp(p["log_l_t"]))
    return (s_l, s_t), (k_l, k_t)


def kf_multilowrank(p, X):
    s_l = q.Noise(jnp.exp(p["log_sigma"]))
    s_l += jnp.exp(2*p["log_h_readout"]) * q.Exp(jnp.exp(p["log_l_readout"]))
    s_t = q.KroneckerDelta()
    alpha_l1 = lr.Constant(jnp.exp(p["log_h_const"]))
    k_t1 = q.Matern32(jnp.exp(p["log_l_t1"]))
    basis_vec = jnp.zeros(X[0].shape[-1])
    basis_vec = jnp.exp(p["log_h10"]) * basis_vec.at[9].set(1.)
    alpha_l2 = lr.Amplitude(basis_vec)
    k_t2 = q.SquaredExpApprox(jnp.exp(p["log_l_t2"]))
    return (s_l, s_t), (alpha_l1, k_t1), (alpha_l2, k_t2)


def kf_multilowranktime(p, X):
    s_l = q.Noise(jnp.exp(p["log_sigma"]))
    s_l += jnp.exp(2*p["log_h_readout"]) * q.Exp(jnp.exp(p["log_l_readout"]))
    s_t = q.KroneckerDelta()
    k_l1 = jnp.exp(2*p["log_h_ramp"]) * q.Matern32(jnp.exp(p["log_l_l_ramp"]))
    settle_vec = jnp.exp(-(X[1] - X[1][0])/jnp.exp(p["log_l_t_ramp"]))
    beta_t1 = lr.Amplitude(settle_vec)
    return (s_l, s_t), (k_l1, beta_t1)


def kf_luaspluslowrank(p, X):
    s_l = q.Noise(jnp.exp(p["log_sigma"]))
    s_l += jnp.exp(2*p["log_h_readout"]) * q.Exp(jnp.exp(p["log_l_readout"]))
    s_t = q.KroneckerDelta()
    k_l = jnp.exp(2*p["log_h"]) * K.Matern52(jnp.exp(p["log_l_l"]))
    k_t = q.Matern32(jnp.exp(p["log_l_t"]))
    k_l1 = jnp.exp(2*p["log_h_ramp"]) * q.Matern32(jnp.exp(p["log_l_l_ramp"]))
    settle_vec = jnp.exp(-(X[1] - X[1][0])/jnp.exp(p["log_l_t_ramp"]))
    beta_t1 = lr.Amplitude(settle_vec)
    return (s_l, s_t), (k_l, k_t), (k_l1, beta_t1)


def kf_doublelowrankterms(p, X):
    s_l = q.Noise(jnp.exp(p["log_sigma"]))
    s_t = q.KroneckerDelta()
    alpha_l1 = lr.Constant(jnp.exp(p["log_h_const"]))
    k_t1 = q.Matern32(jnp.exp(p["log_l_t1"]))
    e_1 = jnp.zeros(X[0].shape[-1])
    e_1 = jnp.exp(p["log_h10"]) * e_1.at[9].set(1.)
    alpha_l2 = lr.Amplitude(e_1)
    k_t2 = q.SquaredExpApprox(jnp.exp(p["log_l_t2"]))
    k_l1 = jnp.exp(2*p["log_h_ramp"]) * q.Matern32(jnp.exp(p["log_l_l_ramp"]))
    settle_vec = jnp.exp(-(X[1] - X[1][0])/jnp.exp(p["log_l_t_ramp"]))
    beta_t1 = lr.Amplitude(settle_vec)
    return (s_l, s_t), (alpha_l1, k_t1), (alpha_l2, k_t2), (k_l1, beta_t1)


def kf_extendedluas(p, X):
    s_l = q.Noise(jnp.exp(p["log_sigma"]))
    s_l += jnp.exp(2*p["log_h_readout"]) * q.Exp(jnp.exp(p["log_l_readout"]))
    s_t = q.KroneckerDelta()
    k_l = lr.Amplitude(jnp.exp(p["log_h_l_vec"])) * K.Matern52(jnp.exp(p["log_l_l"]))
    k_t = q.Matern32(jnp.exp(p["log_l_t"]))
    alpha_l1 = lr.Constant(jnp.exp(p["log_h_const"]))
    k_t1 = q.Matern32(jnp.exp(p["log_l_t1"]))
    e_1 = jnp.zeros(X[0].shape[-1])
    e_1 = jnp.exp(p["log_h10"]) * e_1.at[9].set(1.)
    alpha_l2 = lr.Amplitude(e_1)
    k_t2 = K.SquaredExp(jnp.exp(p["log_l_t2"]))
    k_l1 = jnp.exp(2*p["log_h_ramp"]) * q.Matern32(jnp.exp(p["log_l_l_ramp"]))
    settle_vec = jnp.exp(-(X[1] - X[1][0])/jnp.exp(p["log_l_t_ramp"]))
    beta_t1 = lr.Amplitude(settle_vec)
    return (s_l, s_t), (k_l, k_t), (alpha_l1, k_t1), (alpha_l2, k_t2), (k_l1, beta_t1)


# name: (kernel function, optimisation the kernel selector is expected to choose)
KERNEL_FNS_2D = {
    "readout": (kf_readout, "SeparableKernel"),
    "readout_quasisep": (kf_readout_quasisep, "SeparableKernel"),
    "with_timecorr": (kf_with_timecorr, "LuasLasrachKernel/MultiLowRankKernel"),
    "timecorr_fast": (kf_timecorr_fast, "LuasLasrachKernel/MultiLowRankKernel"),
    "full2d_slow": (kf_full2d_slow, "LuasKernel"),
    "full2d_fast": (kf_full2d_fast, "LuasLasrachKernel"),
    "scaled2Dkernel": (kf_scaled2Dkernel, "LuasLasrachKernel"),
    "multilowrank": (kf_multilowrank, "MultiLowRankKernel"),
    "multilowranktime": (kf_multilowranktime, "LuasLasrachKernel/MultiLowRankKernel"),
    "luaspluslowrank": (kf_luaspluslowrank, "LuasPlusMultiLowRankKernel"),
    "doublelowrankterms": (kf_doublelowrankterms, "DoubleMultiLowRankKernel"),
    "extendedluas": (kf_extendedluas, "ExtendedLuasKernel"),
}

# Kernel functions with a single 2D term, for which LuasKernel and
# LuasLasrachKernel (with fast_dim) should both be valid choices
SINGLE_2D_TERM = ["with_timecorr", "timecorr_fast", "full2d_slow", "full2d_fast",
                  "scaled2Dkernel", "multilowranktime"]

# --------------------------------------------------------------------------- 1D GP
# The 1D squared exponential example from the same tutorial

x_1D = jnp.linspace(-1., 1., 64)
P_1D = {
    "log_sigma": np.log(20e-6),
    "log_h": np.log(80e-6),
    "log_l_t": np.log(0.04),
}
Y_1D = jnp.array(np.random.default_rng(3).normal(size=x_1D.size)) * 1e-4


def kf_sqexp(p, x):
    k = K.Noise(jnp.exp(p["log_sigma"]))
    k += jnp.exp(2*p["log_h"]) * K.SquaredExp(jnp.exp(p["log_l_t"]))
    return k


def kf_quasi_sqexp(p, x):
    k = q.Noise(jnp.exp(p["log_sigma"]))
    k += jnp.exp(2*p["log_h"]) * q.SquaredExpApprox(jnp.exp(p["log_l_t"]))
    return k


KERNEL_FNS_1D = {"sqexp": kf_sqexp, "quasi_sqexp": kf_quasi_sqexp}


def kron_terms(out):
    """Normalises a kernel function's output to a list of (k_dim0, k_dim1) tuples."""
    return [out] if isinstance(out[0], ct.CovType) else list(out)


def dense_cov_2d(kf, p):
    """Dense covariance matrix built independently of any luas optimisation."""
    return sum(jnp.kron(a(x_l, x_l), b(x_t, x_t)) for a, b in kron_terms(kf(p, X)))


def dense_logL(K_dense, R):
    """Gaussian log likelihood via a dense Cholesky decomposition."""
    r = R.ravel()
    L = jnp.linalg.cholesky(K_dense)
    z = jax.scipy.linalg.solve_triangular(L, r, lower=True)
    return -0.5 * z @ z - jnp.log(jnp.diag(L)).sum() - 0.5 * r.size * jnp.log(2 * jnp.pi)


def dense_logL_2d(kf, p, R=Y):
    return dense_logL(dense_cov_2d(kf, p), R)
