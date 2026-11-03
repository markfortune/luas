"""Checks addition and multiplication between CovType objects.

For every pair of CovTypes (in both orders) and for scalars on either side:
  - if the operation returns a result, it must be a CovType whose dense
    matrix matches the dense sum/elementwise product of the operands;
  - if the operation is unsupported, it must raise cleanly (reported as a
    skip under pytest, so unsupported combinations stay visible).

Run with pytest:      pytest dev_tests/test_covtype_ops.py -q -rs
Or as a script:       python dev_tests/test_covtype_ops.py
"""
import itertools
import warnings

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from luas.kernels import covtype as ct, quasisep as qs, base
from luas.kernels.lowrank import LowRank

warnings.filterwarnings("ignore")

N = 12
TOL = 1e-8
x = jnp.linspace(0., 3., N)
_rng = np.random.default_rng(0)
vec = jnp.array(_rng.uniform(0.5, 1.5, N))
alpha = jnp.array(_rng.normal(size=N))
B = jnp.array(_rng.normal(size=(N, 2)))


def make_covtypes():
    return {
        "General": base.SquaredExp(scale=0.7, sigma=1.1),
        "Identity": ct.Identity(),
        "ScaledIdentity": ct.ScaledIdentity(diag=0.3, wn_diag=0.1),
        "Diagonal": ct.Diagonal(diag=vec, wn_diag=0.1),
        "GeneralQuasisep": qs.Matern32(scale=0.8, sigma=0.9),
        "Exp": qs.Exp(scale=0.6, sigma=1.2),
        "GQPN": ct.GeneralQuasisep(qs.Matern52(0.5).tinygp_kf) + ct.ScaledIdentity(diag=0.2),
        "GQPN_noise": qs.Banded(diag=vec, off_diags=0.1 * vec[:, None]) + qs.Matern32(0.8),
        "Outer_vec": ct.Outer(alpha),
        "Outer_scalar": ct.Outer(0.7),
        "OPSI": ct.OuterPlusScaledIdentity(alpha, diag=0.4),
        "LowRank": LowRank(lambda hp, x1, x2, **kw: B @ B.T, rank=2),
    }


OPS = {
    "+": (lambda a, b: a + b, lambda A, Bm: A + Bm),
    "*": (lambda a, b: a * b, lambda A, Bm: A * Bm),
}
SCALAR = 2.0
NAMES = list(make_covtypes())


def dense(k):
    return k if np.isscalar(k) else np.asarray(k.evaluate(x, x))


def check(a, b, sym):
    """Returns (status, detail). status is OK, UNSUPPORTED, WRONG, BADRESULT or EVALERR."""
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
@pytest.mark.parametrize("name_b", NAMES)
@pytest.mark.parametrize("name_a", NAMES)
def test_covtype_pair(name_a, name_b, sym):
    ks = make_covtypes()
    _assert(*check(ks[name_a], ks[name_b], sym))


@pytest.mark.parametrize("scalar_first", [False, True], ids=["k_op_c", "c_op_k"])
@pytest.mark.parametrize("sym", list(OPS))
@pytest.mark.parametrize("name", NAMES)
def test_covtype_scalar(name, sym, scalar_first):
    k = make_covtypes()[name]
    a, b = (SCALAR, k) if scalar_first else (k, SCALAR)
    _assert(*check(a, b, sym))


if __name__ == "__main__":
    import collections
    ks = make_covtypes()
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
