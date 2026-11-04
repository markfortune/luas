"""Regression tests: compare against values saved from a known-good version of luas.

These catch any change in results, including ones the dense comparisons cannot
(for example, an intentional change of convention that was not meant to happen).

To save new reference values from the current code (do this from a version you
trust, then commit the .npz file):
    pytest tests/test_regression.py --update-reference
"""
from pathlib import Path

import jax
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import luas
from kernel_cases import KERNEL_FNS_2D, P, X, Y

REF_FILE = Path(__file__).parent / "reference" / "gp_reference_values.npz"
NAMES = list(KERNEL_FNS_2D)
_updated = {}


@pytest.fixture(scope="module")
def reference(update_reference):
    if update_reference:
        yield None
        REF_FILE.parent.mkdir(exist_ok=True)
        values = dict(np.load(REF_FILE)) if REF_FILE.exists() else {}
        # Drop entries for kernel cases that no longer exist in kernel_cases.py
        values = {k: v for k, v in values.items() if k.split("/", 1)[1] in NAMES}
        values.update(_updated)
        np.savez(REF_FILE, **values)
    elif not REF_FILE.exists():
        pytest.skip(f"No reference file, create one with: pytest {Path(__file__).name} --update-reference")
    else:
        yield dict(np.load(REF_FILE))


def _compare(key, value, reference, rtol):
    value = np.asarray(value)
    if reference is None:
        _updated[key] = value
        pytest.skip(f"saved new reference value for {key}")
    if key not in reference:
        pytest.skip(f"{key} not in reference file, rerun with --update-reference")
    np.testing.assert_allclose(value, reference[key], rtol=rtol, atol=0)


def _gp(name):
    kf, _ = KERNEL_FNS_2D[name]
    return luas.GP(P, X, kf=kf, verbose=False, jit=True)


@pytest.mark.parametrize("name", NAMES)
def test_logL_regression(name, reference):
    _compare(f"logL/{name}", _gp(name).logL(P, Y), reference, rtol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_gradient_regression(name, reference):
    gp = _gp(name)
    grad = ravel_pytree(jax.grad(gp.logL)(P, Y))[0]
    _compare(f"grad/{name}", grad, reference, rtol=1e-8)
