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


# Tolerances allow for round-off differences between machines (e.g. the reference made on one CPU,
# CI on another), while still catching real changes in results.
LOGL_RTOL = 1e-9     # relative to |logL|
GRAD_RTOL = 1e-6     # relative to the largest gradient component, so tiny components don't fail on round-off


def _compare(key, value, reference, rtol, relative_to_max=False):
    value = np.asarray(value)
    if reference is None:
        _updated[key] = value
        pytest.skip(f"saved new reference value for {key}")
    if key not in reference:
        pytest.skip(f"{key} not in reference file, rerun with --update-reference")
    ref = reference[key]
    if relative_to_max:
        err = np.abs(value - ref).max() / max(np.abs(ref).max(), 1e-300)
        assert err < rtol, f"{key}: max error {err:.2e} relative to the largest component (tolerance {rtol:.0e})"
    else:
        np.testing.assert_allclose(value, ref, rtol=rtol, atol=0)


def _gp(name):
    kf, _ = KERNEL_FNS_2D[name]
    return luas.GP(P, X, kf=kf, verbose=False, jit=True)


@pytest.mark.parametrize("name", NAMES)
def test_logL_regression(name, reference):
    _compare(f"logL/{name}", _gp(name).logL(P, Y), reference, rtol=LOGL_RTOL)


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_gradient_regression(name, reference):
    gp = _gp(name)
    grad = ravel_pytree(jax.grad(gp.logL)(P, Y))[0]
    _compare(f"grad/{name}", grad, reference, rtol=GRAD_RTOL, relative_to_max=True)
