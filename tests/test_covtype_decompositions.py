"""Decompositions and solves of invertible 1D CovTypes, checked against dense numpy.

Each CovType in ``kernel_cases.pd_covtypes`` is decomposed and its fast methods
(logdet, matrix_sqrt, matrix_inv_sqrt, inverse, dot_solve, logL, matmul,
eigendecomp) are compared with the same quantity computed from the dense matrix.
Right-hand sides are tested both as a vector (N,) and a matrix (N, 3).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kernel_cases import pd_covtypes, PD_NAMES, x, N, dense_logL

ATOL = 1e-8
_rng = np.random.default_rng(2)
R_MAT = jnp.array(_rng.normal(size=(N, 3)))
RHS = {"vector": R_MAT[:, 0], "matrix": R_MAT}


def assert_close(got, want, atol=ATOL):
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=atol, rtol=1e-7)


@pytest.fixture(params=PD_NAMES)
def case(request):
    """Returns (covtype, decomposed covtype, dense matrix)."""
    k = pd_covtypes()[request.param]
    K_dense = np.asarray(k.evaluate(x, x))
    k_dec, _ = k.decompose(x)
    return k, k_dec, K_dense


@pytest.fixture(params=list(RHS))
def R(request):
    return RHS[request.param]


def test_logdet(case):
    _, k_dec, K_dense = case
    assert_close(k_dec.logdet, np.linalg.slogdet(K_dense)[1])


def test_matrix_sqrt_inverts_matrix_inv_sqrt(case, R):
    _, k_dec, _ = case
    assert_close(k_dec.matrix_sqrt(k_dec.matrix_inv_sqrt(R)), R)


def test_inv_sqrt_and_transpose_solve(case, R):
    # L^-T L^-1 R = K^-1 R
    _, k_dec, K_dense = case
    got = k_dec.matrix_inv_sqrt(k_dec.matrix_inv_sqrt(R, transpose=0), transpose=1)
    assert_close(got, np.linalg.solve(K_dense, R))


def test_inverse(case, R):
    _, k_dec, K_dense = case
    assert_close(k_dec.inverse(R), np.linalg.solve(K_dense, R))


def test_dot_solve(case):
    _, k_dec, K_dense = case
    r = RHS["vector"]
    assert_close(k_dec.dot_solve(r), r @ np.linalg.solve(K_dense, r))


def test_logL(case):
    _, k_dec, K_dense = case
    r = RHS["vector"]
    assert_close(k_dec.logL(r), dense_logL(jnp.asarray(K_dense), r))


def test_logL_under_jit(case):
    k, _, K_dense = case
    r = RHS["vector"]

    @jax.jit
    def logL(r):
        k_dec, _ = k.decompose(x)
        return k_dec.logL(r)

    assert_close(logL(r), dense_logL(jnp.asarray(K_dense), r))


def test_matmul(case, R):
    k, _, K_dense = case
    assert_close(k.matmul(x, x, R), K_dense @ R)


def test_eigendecomp_reconstructs_matrix(case):
    k, _, K_dense = case
    lam, Q = k.eigendecomp(x)
    # Some CovTypes return a Householder transform object rather than an array
    Q = np.asarray(Q) if isinstance(Q, (jax.Array, np.ndarray)) else np.asarray(Q @ jnp.eye(N))
    assert_close(Q @ np.diag(np.asarray(lam)) @ Q.T, K_dense)
