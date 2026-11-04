"""Addition and multiplication between CovType objects.

For every pair of 1D CovTypes (in both orders), and for scalars on either side:
  - if the operation returns a result, it must be a CovType whose dense matrix
    equals the dense sum / elementwise product of the operands;
  - if the operation is unsupported it must raise cleanly. These are reported as
    skips (see them with ``pytest -rs``) so the unsupported combinations stay visible.

Can also be run as a script to print a summary table:
    python tests/test_covtype_algebra.py
"""
import itertools

import numpy as np
import pytest

from luas.kernels import covtype as ct
from kernel_cases import algebra_covtypes, ALGEBRA_NAMES, x

TOL = 1e-8
SCALAR = 2.0
OPS = {
    "+": (lambda a, b: a + b, lambda A, B: A + B),
    "*": (lambda a, b: a * b, lambda A, B: A * B),
}


def dense(k):
    return k if np.isscalar(k) else np.asarray(k.evaluate(x, x))


def check(a, b, sym):
    """Returns (status, detail), status one of OK, UNSUPPORTED, WRONG, BADRESULT, EVALERR."""
    f, f_dense = OPS[sym]
    try:
        result = f(a, b)
    except Exception as e:
        return "UNSUPPORTED", f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
    if not isinstance(result, ct.CovType):
        return "BADRESULT", f"returned {type(result).__name__}, not a CovType"
    try:
        got = np.asarray(result.evaluate(x, x))
    except Exception as e:
        return "EVALERR", f"{type(result).__name__}.evaluate failed: {type(e).__name__}: {e}"
    err = np.abs(got - f_dense(dense(a), dense(b))).max()
    if err > TOL:
        return "WRONG", f"{type(result).__name__} max abs error {err:.2e}"
    return "OK", type(result).__name__


def _assert(status, detail):
    if status == "UNSUPPORTED":
        pytest.skip(detail)
    assert status == "OK", detail


@pytest.mark.parametrize("sym", list(OPS))
@pytest.mark.parametrize("name_b", ALGEBRA_NAMES)
@pytest.mark.parametrize("name_a", ALGEBRA_NAMES)
def test_covtype_pair(name_a, name_b, sym):
    ks = algebra_covtypes()
    _assert(*check(ks[name_a], ks[name_b], sym))


@pytest.mark.parametrize("scalar_first", [False, True], ids=["k_op_c", "c_op_k"])
@pytest.mark.parametrize("sym", list(OPS))
@pytest.mark.parametrize("name", ALGEBRA_NAMES)
def test_covtype_scalar(name, sym, scalar_first):
    k = algebra_covtypes()[name]
    a, b = (SCALAR, k) if scalar_first else (k, SCALAR)
    _assert(*check(a, b, sym))


if __name__ == "__main__":
    import collections
    import conftest  # noqa: F401  (enables 64-bit precision)

    ks = algebra_covtypes()
    rows = []
    for (na, a), (nb, b) in itertools.product(ks.items(), repeat=2):
        for sym in OPS:
            rows.append((na, sym, nb, *check(a, b, sym)))
    for na, a in ks.items():
        for sym in OPS:
            rows.append((na, sym, "scalar", *check(a, SCALAR, sym)))
            rows.append(("scalar", sym, na, *check(SCALAR, a, sym)))
    print("SUMMARY", dict(collections.Counter(r[3] for r in rows)))
    for na, sym, nb, status, detail in rows:
        if status != "OK":
            print(f"{status:11s} {na:15s} {sym} {nb:15s} | {detail}")
