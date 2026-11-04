"""Stress tests for 2D (and 3D) GP kernels under reasonable user behaviour.

Goes beyond the tutorial examples to find kernel definitions, inputs and usage
patterns that break luas. Every case is checked against a dense log likelihood
built from the evaluated 1D kernels, so a failure means one of:

  WRONG  luas returned a value that disagrees with the dense calculation (worst:
         it fails silently)
  ERROR  luas raised an exception for something a user could reasonably do

Unsorted inputs to quasisep kernels are a known limitation: a wrong result there
passes (reported as WARNED) as long as luas warned that the inputs are not sorted,
and fails only if no warning was given. Cases in EXPECTED_ERRORS are misuse
that should raise luas's own clear error (reported as EXPECTED).

Cases are grouped by test id prefix:
  sweep-<condition>-<kernel>  input/parameter conditions applied to each tutorial kernel
  compose-<name>              alternative ways of building kernels
  usage-<name>                jit / vmap / grad / repeated-call patterns

All tests are marked ``slow`` and ``stress``:
    pytest tests/test_stress_2d.py                  # run everything (several minutes)
    pytest tests/test_stress_2d.py -k unsorted      # one condition
    python tests/test_stress_2d.py                  # summary table instead of pytest output
"""
import warnings
from copy import deepcopy
from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import luas
from luas.kernels import covtype as ct, quasisep as q, lowrank as lr
import kernel_cases as kc

K = luas.kernels
pytestmark = [pytest.mark.slow, pytest.mark.stress]

RTOL = 1e-8          # relative tolerance on logL
SORT_WARNING = "not sorted"   # text of luas's unsorted-input warning

# Misuse that luas should reject with its own clear message (case name: text the error must contain).
# These pass when that error is raised, and fail if luas raises something else or runs without error.
EXPECTED_ERRORS = {
    "lowrank_term_before_noise": "Outer product matrix is not invertible",
    "returns_list": "must return a CovType object, or a tuple",
    "two_full_rank_terms": "maximum of two dense kernels",
}
GRAD_RTOL = 1e-7     # tolerance on gradients, relative to the largest component

# --------------------------------------------------------------------------- dense reference


def kron_terms(out):
    if isinstance(out, list):
        out = tuple(out)
    return [out] if isinstance(out[0], ct.CovType) else list(out)


def dense_cov(kf, p, X):
    mats = []
    for term in kron_terms(kf(p, X)):
        mats.append(reduce(jnp.kron, [k(x, x) for k, x in zip(term, X)]))
    return sum(mats)


def dense_logL(kf, p, X, Y):
    return kc.dense_logL(dense_cov(kf, p, X), jnp.asarray(Y, dtype=jnp.float64))


# --------------------------------------------------------------------------- data builders


def make_X(N_l=12, N_t=20, x_l=None, x_t=None):
    x_l = np.linspace(4., 5., N_l) if x_l is None else x_l
    x_t = jnp.linspace(-0.15, 0.15, N_t) if x_t is None else x_t
    return (x_l, x_t)


def make_p(X):
    """Tutorial hyperparameters sized for the given inputs."""
    x_l = np.asarray(X[0], dtype=float)
    p = deepcopy(kc.P)
    N_l = x_l.shape[-1]
    p["log_sigma"] = np.log(100e-6) * np.ones(N_l)
    h = 200e-6 * (x_l - x_l.mean())**2 + 1e-6      # +1e-6 keeps the log finite when N_l is odd
    p["log_h_l"] = np.log(h)
    p["log_h_l_vec"] = np.log(h)
    return p


def make_Y(X, seed=1):
    shape = tuple(np.shape(x)[-1] for x in X)
    return jnp.array(np.random.default_rng(seed).normal(size=shape)) * 1e-4


SWEEP_KERNELS = ["readout_quasisep", "timecorr_fast", "full2d_slow", "full2d_fast", "scaled2Dkernel",
                 "multilowrank", "multilowranktime", "luaspluslowrank", "doublelowrankterms", "extendedluas"]
NEEDS_10_LC = {"multilowrank", "doublelowrankterms", "extendedluas"}   # use light curve index 9

_rng = np.random.default_rng(7)


def _cond(**kw):
    return kw


# Each condition returns X (and optionally modifies p / Y via p_fn / y_fn)
CONDITIONS = {
    "baseline": _cond(),
    "more_lc_than_times": _cond(X=make_X(N_l=24, N_t=8)),
    "square_grid": _cond(X=make_X(N_l=15, N_t=15)),
    "odd_sizes": _cond(X=make_X(N_l=11, N_t=13)),
    "three_times": _cond(X=make_X(N_l=12, N_t=3)),
    "two_lc": _cond(X=make_X(N_l=2, N_t=20)),
    "single_lc": _cond(X=make_X(N_l=1, N_t=20)),
    "nonuniform_times": _cond(X=make_X(x_t=jnp.sort(jnp.array(_rng.uniform(-0.15, 0.15, 20))))),
    "unsorted_times": _cond(X=make_X(x_t=jnp.array(_rng.permutation(np.linspace(-0.15, 0.15, 20))))),
    "descending_times": _cond(X=make_X(x_t=jnp.linspace(0.15, -0.15, 20))),
    "duplicate_times": _cond(X=make_X(x_t=jnp.repeat(jnp.linspace(-0.15, 0.15, 10), 2))),
    "bjd_times": _cond(X=make_X(x_t=2460000.5 + jnp.linspace(-0.15, 0.15, 20))),
    "unsorted_wavelengths": _cond(X=make_X(x_l=_rng.permutation(np.linspace(4., 5., 12)))),
    "integer_inputs": _cond(X=(np.arange(12), np.arange(20)),
                            p_fn=lambda p: {**p, "log_l_l": np.log(2.), "log_l_t": np.log(3.), "log_l_readout": np.log(5.),
                                            "log_l_t1": np.log(3.), "log_l_t2": np.log(2.), "log_l_l_ramp": np.log(5.),
                                            "log_l_t_ramp": np.log(4.)}),
    "float32_inputs": _cond(X=(np.linspace(4., 5., 12, dtype=np.float32), np.linspace(-0.15, 0.15, 20, dtype=np.float32)),
                            y_fn=lambda Y: np.asarray(Y, dtype=np.float32)),
    "jnp_scalar_hyperparams": _cond(p_fn=lambda p: {k: jnp.asarray(v) for k, v in p.items()}),
    "short_lengthscales": _cond(p_fn=lambda p: {k: (v + np.log(0.05) if k.startswith("log_l") else v) for k, v in p.items()}),
    "long_lengthscales": _cond(p_fn=lambda p: {k: (v + np.log(100.) if k.startswith("log_l") else v) for k, v in p.items()}),
    "tiny_correlated_amplitudes": _cond(p_fn=lambda p: {k: (v - 20. if k.startswith("log_h") and k != "log_h_readout" else v)
                                                        for k, v in p.items()}),
    "large_correlated_amplitudes": _cond(p_fn=lambda p: {k: (v + 6. if k.startswith("log_h") else v) for k, v in p.items()}),
    "noise_dynamic_range": _cond(p_fn=lambda p: {**p, "log_sigma": np.linspace(np.log(1e-6), np.log(1e-2), len(p["log_sigma"]))}),
}


def sweep_case(cond_name, kernel_name):
    cond = CONDITIONS[cond_name]
    X = cond.get("X", make_X())
    p = make_p(X)
    if "p_fn" in cond:
        p = cond["p_fn"](p)
    Y = make_Y(X)
    if "y_fn" in cond:
        Y = cond["y_fn"](Y)
    return dict(kf=kc.KERNEL_FNS_2D[kernel_name][0], X=X, p=p, Y=Y)


SWEEP_IDS = [(c, k) for c in CONDITIONS for k in SWEEP_KERNELS
             if not (k in NEEDS_10_LC and np.shape(CONDITIONS[c].get("X", make_X())[0])[-1] < 10)]

# --------------------------------------------------------------------------- composition cases


def _noise_l(p):
    s_l = q.Noise(jnp.exp(p["log_sigma"]))
    return s_l + jnp.exp(2*p["log_h_readout"]) * q.Exp(jnp.exp(p["log_l_readout"]))


def _matern_t(p, key="log_l_t"):
    return q.Matern32(jnp.exp(p[key]))


def compose_cases():
    X = make_X()
    p = make_p(X)
    N_l, N_t = 12, 20
    alpha = jnp.asarray(np.random.default_rng(5).normal(size=N_l)) * 80e-6
    beta = jnp.exp(-(X[1] - X[1][0]) / 0.03)
    sig_t = jnp.linspace(0.5, 2., N_t)
    k_l = lambda p: jnp.exp(2*p["log_h"]) * K.Matern52(jnp.exp(p["log_l_l"]))
    cases = {
        # Low-rank amplitudes
        "amplitude_with_zeros": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),
                                              (lr.Amplitude(alpha.at[::3].set(0.)) * K.Matern52(jnp.exp(p["log_l_l"])), _matern_t(p))),
        "amplitude_negative_entries": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),
                                                    (lr.Amplitude(alpha), _matern_t(p))),
        "lowrank_amplitude_exactly_zero": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),
                                                        (lr.Constant(0.), _matern_t(p))),
        "lowrank_one_hot_first_lc": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),
                                                  (lr.Amplitude(jnp.zeros(N_l).at[0].set(1e-4)), _matern_t(p))),
        "lowrank_both_dims": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),
                                           (lr.Amplitude(alpha), lr.Amplitude(beta * 1e-4))),
        "duplicate_lowrank_terms": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),
                                                 (lr.Constant(80e-6), _matern_t(p)), (lr.Constant(80e-6), _matern_t(p))),
        "parallel_lowrank_vectors": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),
                                                  (lr.Amplitude(alpha), _matern_t(p)),
                                                  (lr.Amplitude(2 * alpha), q.Matern52(0.02))),
        "five_lowrank_terms": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),) + tuple(
            (lr.Amplitude(jnp.zeros(N_l).at[i].set(1e-4)), q.Matern32(0.02 + 0.01 * i)) for i in range(5)),
        # Ordering and dimension layout
        "noise_term_not_first": lambda p, X: ((k_l(p), _matern_t(p)), (_noise_l(p), q.KroneckerDelta())),
        "lowrank_term_before_noise": lambda p, X: ((lr.Constant(80e-6), _matern_t(p)), (_noise_l(p), q.KroneckerDelta())),
        "noise_in_time_dim": lambda p, X: ((q.KroneckerDelta(1.), q.Noise(100e-6 * sig_t)), (lr.Constant(80e-6), _matern_t(p))),
        "kroneckerdelta_vector_time": lambda p, X: ((_noise_l(p), q.KroneckerDelta(sig_t)), (lr.Constant(80e-6), _matern_t(p))),
        "base_noise_instead_of_kd": lambda p, X: ((_noise_l(p), K.Noise(1.)), (lr.Constant(80e-6), _matern_t(p))),
        "scalar_noise": lambda p, X: ((q.Noise(100e-6), q.KroneckerDelta()), (k_l(p), _matern_t(p))),
        "noise_both_dims": lambda p, X: ((q.Noise(jnp.exp(p["log_sigma"])), q.Noise(sig_t)), (k_l(p), _matern_t(p))),
        "returns_list": lambda p, X: [(_noise_l(p), q.KroneckerDelta()), (k_l(p), _matern_t(p))],
        # Building kernels within one dimension
        "sum_quasisep_time": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),
                                           (k_l(p), q.Matern32(0.04) + 0.5 * q.Matern52(0.02))),
        "sum_dense_and_quasisep_time": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),
                                                     (k_l(p), q.Matern32(0.04) + K.SquaredExp(0.02))),
        "product_quasisep_time": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),
                                               (k_l(p), q.Matern32(0.1) * q.Cosine(0.05))),
        "scalar_on_right": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),
                                         (K.Matern52(jnp.exp(p["log_l_l"])) * jnp.exp(2*p["log_h"]), _matern_t(p) * 1.0)),
        "constant_times_kernel_l": lambda p, X: ((_noise_l(p), q.KroneckerDelta()),
                                                 (lr.Constant(80e-6) * K.Matern52(jnp.exp(p["log_l_l"])), _matern_t(p))),
        "mixed_base_quasisep_noise": lambda p, X: ((K.Noise(jnp.exp(p["log_sigma"])) + 1e-9 * q.Exp(0.3), q.KroneckerDelta()),
                                                   (k_l(p), _matern_t(p))),
        "two_full_rank_terms": lambda p, X: ((_noise_l(p), q.KroneckerDelta()), (k_l(p), _matern_t(p)),
                                             (1e-9 * K.Matern32(0.2), q.Matern52(0.02))),
        "full_rank_plus_lowrank_l": lambda p, X: ((_noise_l(p), q.KroneckerDelta()), (k_l(p), _matern_t(p)),
                                                  (lr.Constant(80e-6), q.Matern52(0.02))),
    }
    return {name: dict(kf=kf, X=X, p=p, Y=make_Y(X)) for name, kf in cases.items()}


def extra_layout_cases():
    """Cases that change X itself rather than the kernel."""
    out = {}
    # Dimensions swapped: time first, wavelength second
    X = make_X()
    p = make_p(X)
    kf = kc.kf_timecorr_fast
    Xs = (X[1], X[0])
    out["dims_swapped"] = dict(kf=lambda p, X: tuple(t[::-1] for t in kf(p, (X[1], X[0]))), X=Xs, p=p, Y=make_Y(Xs))
    # 3D separable and 3D with a second term
    X3 = (np.linspace(0, 1, 4), np.linspace(4., 5., 5), jnp.linspace(-0.15, 0.15, 6))
    p3 = {"s": 100e-6}
    out["3d_separable"] = dict(kf=lambda p, X: (q.Noise(p["s"] * jnp.ones(4)), q.Exp(0.5) + q.Noise(0.3), q.KroneckerDelta()),
                               X=X3, p=p3, Y=make_Y(X3))
    out["3d_two_terms"] = dict(kf=lambda p, X: ((q.Noise(p["s"] * jnp.ones(4)), q.KroneckerDelta(), q.KroneckerDelta()),
                                                (lr.Constant(80e-6), K.Matern52(0.3), q.Matern32(0.04))),
                               X=X3, p=p3, Y=make_Y(X3))
    return out


COMPOSE = {**compose_cases(), **extra_layout_cases()}

# --------------------------------------------------------------------------- runner


def check_logL(case, jit=False, **gp_kwargs):
    kf, X, p, Y = case["kf"], case["X"], case["p"], case["Y"]
    ref = float(dense_logL(kf, p, X, Y))
    if not np.isfinite(ref):
        return "REFFAIL", f"dense reference not finite ({ref})"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gp = luas.GP(p, X, kf=kf, verbose=False, jit=jit, **gp_kwargs)
        val = float(gp.logL(p, Y))
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {str(e).strip().splitlines()[0][:160]}"
    err = abs(val - ref)
    if not np.isfinite(val) or err > RTOL * max(1., abs(ref)):
        if any(SORT_WARNING in str(w.message) for w in caught):
            # Known limitation: quasisep kernels need sorted inputs. Acceptable as long as luas warned.
            return "WARNED", f"{gp.opt_name}: luas {val:.6g} vs dense {ref:.6g}, unsorted-input warning given"
        return "WRONG", f"{gp.opt_name}: luas {val:.10g} vs dense {ref:.10g} (diff {val - ref:.2e})"
    return "OK", f"{gp.opt_name} (diff {val - ref:.1e})"


def check_grad(case):
    kf, X, p, Y = case["kf"], case["X"], case["p"], case["Y"]
    want = ravel_pytree(jax.grad(lambda pp: dense_logL(kf, pp, X, Y))(p))[0]
    try:
        gp = luas.GP(p, X, kf=kf, verbose=False, jit=True)
        got = ravel_pytree(jax.grad(gp.logL)(p, Y))[0]
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {str(e).strip().splitlines()[0][:160]}"
    rel = float(jnp.abs(got - want).max() / jnp.abs(want).max())
    if not np.isfinite(rel) or rel > GRAD_RTOL:
        return "WRONG", f"{gp.opt_name}: max relative gradient error {rel:.2e}"
    return "OK", f"{gp.opt_name} (grad rel err {rel:.1e})"


def check_repeat(case):
    """jit-compiled logL called with p1, p2, then p1 again must match dense each time."""
    kf, X, p, Y = case["kf"], case["X"], case["p"], case["Y"]
    p2 = {k: (v + 0.3 if k.startswith("log_l") else v) for k, v in p.items()}
    try:
        gp = luas.GP(p, X, kf=kf, verbose=False, jit=True)
        vals = [float(gp.logL(pp, Y)) for pp in (p, p2, p)]
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {str(e).strip().splitlines()[0][:160]}"
    refs = [float(dense_logL(kf, pp, X, Y)) for pp in (p, p2, p)]
    errs = [abs(a - b) for a, b in zip(vals, refs)]
    if max(errs) > RTOL * max(1., max(map(abs, refs))):
        return "WRONG", f"diffs for p1, p2, p1: {[f'{e:.1e}' for e in errs]}"
    return "OK", gp.opt_name


def check_vmap(case):
    """vmap of logL over a batch of three parameter sets."""
    kf, X, p, Y = case["kf"], case["X"], case["p"], case["Y"]
    shifts = jnp.array([0., 0.2, -0.2])
    batch = {k: (jnp.asarray(v)[None] + shifts.reshape((3,) + (1,) * np.ndim(v)) if k.startswith("log_l")
                 else jnp.broadcast_to(jnp.asarray(v), (3,) + np.shape(v))) for k, v in p.items()}
    try:
        gp = luas.GP(p, X, kf=kf, verbose=False, jit=True)
        vals = np.asarray(jax.vmap(lambda pp: gp.logL(pp, Y))(batch))
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {str(e).strip().splitlines()[0][:160]}"
    refs = np.array([float(dense_logL(kf, jax.tree.map(lambda a: a[i], batch), X, Y)) for i in range(3)])
    err = np.abs(vals - refs).max()
    if not np.isfinite(err) or err > RTOL * max(1., np.abs(refs).max()):
        return "WRONG", f"max diff {err:.2e}"
    return "OK", gp.opt_name


USAGE_KERNELS = ["timecorr_fast", "full2d_fast", "scaled2Dkernel", "multilowrank", "multilowranktime",
                 "luaspluslowrank", "doublelowrankterms", "extendedluas"]
USAGE = {
    **{f"jit_repeated_params-{k}": (check_repeat, k) for k in USAGE_KERNELS},
    **{f"vmap_params-{k}": (check_vmap, k) for k in USAGE_KERNELS},
    **{f"grad_nonuniform_times-{k}": (check_grad, ("nonuniform_times", k)) for k in USAGE_KERNELS},
    **{f"grad_more_lc_than_times-{k}": (check_grad, ("more_lc_than_times", k)) for k in USAGE_KERNELS},
}


def _usage_case(target):
    if isinstance(target, tuple):
        return sweep_case(*target)
    return sweep_case("baseline", target)


def _assert(status, detail):
    if status == "REFFAIL":
        pytest.skip(detail)
    assert status in ("OK", "WARNED", "EXPECTED"), f"{status}: {detail}"


# --------------------------------------------------------------------------- tests


@pytest.mark.parametrize("cond,kernel", SWEEP_IDS, ids=[f"sweep-{c}-{k}" for c, k in SWEEP_IDS])
def test_sweep(cond, kernel):
    _assert(*check_logL(sweep_case(cond, kernel)))


def check_compose(name):
    status, detail = check_logL(COMPOSE[name])
    if name not in EXPECTED_ERRORS:
        return status, detail
    if status == "ERROR" and EXPECTED_ERRORS[name] in detail:
        return "EXPECTED", f"raised the intended error: {detail}"
    return "WRONG", f"expected an error containing {EXPECTED_ERRORS[name]!r}, got {status}: {detail}"


@pytest.mark.parametrize("name", list(COMPOSE), ids=[f"compose-{n}" for n in COMPOSE])
def test_compose(name):
    _assert(*check_compose(name))


@pytest.mark.parametrize("name", list(USAGE), ids=[f"usage-{n}" for n in USAGE])
def test_usage(name):
    fn, target = USAGE[name]
    _assert(*fn(_usage_case(target)))


# --------------------------------------------------------------------------- report


def run_all(select=None):
    import conftest  # noqa: F401  (enables 64-bit precision)
    rows = []
    jobs = ([(f"sweep-{c}-{k}", lambda c=c, k=k: check_logL(sweep_case(c, k))) for c, k in SWEEP_IDS]
            + [(f"compose-{n}", lambda n=n: check_compose(n)) for n in COMPOSE]
            + [(f"usage-{n}", lambda n=n: USAGE[n][0](_usage_case(USAGE[n][1]))) for n in USAGE])
    for name, job in jobs:
        if select and select not in name:
            continue
        status, detail = job()
        rows.append((name, status, detail))
        print(f"{status:8s} {name:55s} {detail}", flush=True)
    return rows


if __name__ == "__main__":
    import sys
    import collections
    from pathlib import Path
    print(f"Testing luas from: {Path(luas.__file__).resolve().parent}")
    if Path(luas.__file__).resolve().parent != Path(__file__).resolve().parent.parent / "src" / "luas":
        print("WARNING: not this repo's src/luas; run `pip install -e .` from the repo root to test your working copy.")
    rows = run_all(sys.argv[1] if len(sys.argv) > 1 else None)
    print("\nSUMMARY", dict(collections.Counter(r[1] for r in rows)))
