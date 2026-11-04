"""2D GP log likelihoods for every optimisation, checked against a dense calculation.

The dense reference (``kernel_cases.dense_logL_2d``) builds the full covariance
matrix as a sum of Kronecker products of the evaluated 1D kernels and uses a
Cholesky decomposition, independently of any luas kernel class.

Gradient tests are marked ``slow``; skip them with ``pytest -m "not slow"``.
"""
import jax
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import luas
from kernel_cases import (KERNEL_FNS_2D, SINGLE_2D_TERM, P, X, Y, dense_logL_2d,
                          KERNEL_FNS_1D, P_1D, x_1D, Y_1D, dense_logL)

NAMES = list(KERNEL_FNS_2D)
ATOL = 1e-8
GRAD_RTOL = 1e-6   # gradient error relative to its largest component


def make_gp(name, jit=False, **kwargs):
    kf, _ = KERNEL_FNS_2D[name]
    return luas.GP(P, X, kf=kf, verbose=False, jit=jit, **kwargs)


def reference(name):
    kf, _ = KERNEL_FNS_2D[name]
    return dense_logL_2d(kf, P, Y)


@pytest.mark.parametrize("name", NAMES)
def test_kernel_selector_choice(name):
    # Catches changes to which optimisation is chosen automatically
    _, expected = KERNEL_FNS_2D[name]
    assert make_gp(name).opt_name == expected


@pytest.mark.parametrize("name", NAMES)
def test_logL_matches_dense(name):
    np.testing.assert_allclose(make_gp(name).logL(P, Y), reference(name), atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", NAMES)
def test_logL_jit_matches_dense(name):
    np.testing.assert_allclose(make_gp(name, jit=True).logL(P, Y), reference(name), atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", NAMES)
def test_general_kernel_matches_dense(name):
    gp = make_gp(name, use_kernel=luas.GeneralKernel)
    np.testing.assert_allclose(gp.logL(P, Y), reference(name), atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", SINGLE_2D_TERM)
def test_luas_kernel_alternative(name):
    gp = make_gp(name, use_kernel=luas.LuasKernel)
    np.testing.assert_allclose(gp.logL(P, Y), reference(name), atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", SINGLE_2D_TERM)
def test_luas_lasrach_kernel_alternative(name):
    gp = make_gp(name, use_kernel=luas.LuasLasrachKernel, fast_dim=1)
    np.testing.assert_allclose(gp.logL(P, Y), reference(name), atol=ATOL, rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_gradient_matches_dense(name):
    kf, _ = KERNEL_FNS_2D[name]
    gp = make_gp(name, jit=True)
    got = ravel_pytree(jax.grad(gp.logL)(P, Y))[0]
    want = ravel_pytree(jax.grad(lambda p: dense_logL_2d(kf, p, Y))(P))[0]
    # Compare relative to the largest gradient component
    rel_err = float(np.abs(got - want).max() / np.abs(want).max())
    assert rel_err < GRAD_RTOL, f"max relative gradient error {rel_err:.2e}"


# ----------------------------------------------------------------------------- 1D GPs

def dense_logL_1d(kf, p):
    return dense_logL(kf(p, x_1D)(x_1D, x_1D), Y_1D)


@pytest.mark.parametrize("jit", [False, True], ids=["nojit", "jit"])
@pytest.mark.parametrize("name", list(KERNEL_FNS_1D))
def test_1d_logL_matches_dense(name, jit):
    kf = KERNEL_FNS_1D[name]
    gp = luas.GP(P_1D, x_1D, kf=kf, verbose=False, jit=jit)
    np.testing.assert_allclose(gp.logL(P_1D, Y_1D), dense_logL_1d(kf, P_1D), atol=ATOL, rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(KERNEL_FNS_1D))
def test_1d_gradient_matches_dense(name):
    kf = KERNEL_FNS_1D[name]
    gp = luas.GP(P_1D, x_1D, kf=kf, verbose=False, jit=True)
    got = ravel_pytree(jax.grad(gp.logL)(P_1D, Y_1D))[0]
    want = ravel_pytree(jax.grad(lambda p: dense_logL_1d(kf, p))(P_1D))[0]
    rel_err = float(np.abs(got - want).max() / np.abs(want).max())
    assert rel_err < GRAD_RTOL, f"max relative gradient error {rel_err:.2e}"
