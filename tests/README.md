# luas tests

Most tests compare a fast luas calculation against the same quantity computed from
the dense covariance matrix with plain numpy/JAX.

| File | What it checks |
| --- | --- |
| `test_covtype_algebra.py` | `+` and `*` between every pair of 1D CovTypes (both orders) and with scalars |
| `test_covtype_decompositions.py` | logdet, sqrt/inverse sqrt, solves, logL, matmul and eigendecomposition of invertible 1D CovTypes |
| `test_gp_logL.py` | 2D GP log likelihood for every optimisation: automatic choice, `GeneralKernel`, alternative kernels, `jit` and gradients |
| `test_regression.py` | logL and gradients against saved values in `reference/` |
| `test_stress_2d.py` | Unusual but reasonable usage: input layouts, parameter extremes, alternative kernel constructions, jit/vmap/grad patterns (`slow` and `stress`) |

Shared kernels and data live in `kernel_cases.py` (following
`docs/source/tutorials/Walking_through_luas_optimisations.ipynb`); 64-bit precision is enabled in `conftest.py`.

## Running

```bash
pytest                      # everything
pytest -m "not slow"        # quick run: skip gradient and stress tests
pytest -m "not stress"     # everything except the stress tests
pytest tests/test_stress_2d.py              # stress tests only (several minutes)
python tests/test_stress_2d.py [filter]     # stress tests as a summary table
pytest -rs                  # also list skipped tests (unsupported +/* combinations)
pytest tests/test_gp_logL.py -k extendedluas   # one file / one case
python tests/test_covtype_algebra.py           # summary table of supported +/* combinations
```

## Before opening a pull request

1. Run `pytest` in your usual environment.
2. If anything fails, check whether it also fails on `main`
   (`git worktree add ../luas-main main`, then run the same tests there).
3. Only if a change in results is intended: `pytest tests/test_regression.py --update-reference`
   and commit the updated `reference/gp_reference_values.npz` with the change.

GitHub Actions (`.github/workflows/tests.yml`) runs everything except the stress tests on every pull request.
The stress tests document known limitations as failures, so run them by hand when working on the kernels.
