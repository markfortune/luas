"""Shared pytest configuration for the luas test suite.

64-bit precision is enabled here, before any test module imports luas, so every
test compares fast and dense calculations in double precision.
"""
import os

# Same flag recommended in the tutorials; speeds up tinygp's small while loops on CPU
os.environ.setdefault("XLA_FLAGS", "--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=1000000")

import jax

jax.config.update("jax_enable_x64", True)

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-reference",
        action="store_true",
        default=False,
        help="Regenerate tests/reference/*.npz from the current code instead of comparing against it.",
    )


@pytest.fixture(scope="session")
def update_reference(request):
    return request.config.getoption("--update-reference")


def pytest_report_header(config):
    """Shows which copy of luas is being tested, and warns if it isn't this repo's src/."""
    from pathlib import Path
    import luas

    luas_dir = Path(luas.__file__).resolve().parent
    repo_src = (Path(__file__).resolve().parent.parent / "src" / "luas")
    lines = [f"luas imported from: {luas_dir}"]
    if luas_dir != repo_src:
        lines.append("WARNING: this is not this repo's src/luas, so results may not reflect your current code. "
                     "Run `pip install -e .` from the repo root to test the working copy.")
    return lines

