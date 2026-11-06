"""Compares the luas.exoplanet light curves (computed with jaxoplanet) against batman.

batman (https://github.com/lkreidberg/batman) is an independent, widely used transit code, so
agreement here checks the luas wrappers: parameter conversions (b -> inclination, Kipping limb
darkening, depth -> radius ratio), baselines and eclipse normalisation.

batman is only needed for these tests, not for using luas. Its normal build compiles against an old
numpy and then fails to import under numpy 2, so install it against your installed numpy:

    pip install --no-build-isolation batman-package

The tests are skipped if batman or jaxoplanet isn't installed, unless the environment variable
LUAS_REQUIRE_EXOPLANET_TESTS is set (CI sets it), in which case a missing package is an error.
"""
import os
import warnings

import numpy as np
import pytest

if os.environ.get("LUAS_REQUIRE_EXOPLANET_TESTS"):
    import batman          # fail loudly in CI rather than skipping
    import jaxoplanet  # noqa: F401
else:
    batman = pytest.importorskip("batman")
    pytest.importorskip("jaxoplanet")

import jax.numpy as jnp

from luas import exoplanet as ex
from luas.exoplanet.transit_fns import eclipse_light_curve, ld_from_kipping

warnings.filterwarnings("ignore", module="batman")

# luas and batman agree to about 1e-8 (jaxoplanet 0.1 and 0.2)
ATOL = 1e-6   # 1 ppm, well below the precision of current transit observations

P, A, T0 = 3., 8., 0.
t = np.linspace(-0.15, 0.15, 2001)


def batman_params(rho, b, u=(), limb_dark="quadratic"):
    """batman parameters for a planet on a circular orbit with impact parameter b."""
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = T0, P, rho, A
    p.inc = np.degrees(np.arccos(b / A))     # b = a cos(i) for a circular orbit
    p.ecc, p.w, p.limb_dark, p.u = 0., 90., limb_dark, list(u)
    return p


def batman_transit(rho, b, u, times=t, **kw):
    p = batman_params(rho, b, u, **kw)
    return batman.TransitModel(p, times).light_curve(p)


BASE = {"T0": T0, "P": P, "a": A, "Foot": 1., "Tgrad": 0.}
# (rho, b, u1, u2): central, typical, grazing-ish and grazing transits
GEOMETRIES = [(0.1, 0.5, 0.3, 0.2), (0.15, 0.0, 0.5, 0.1), (0.05, 0.9, 0.1, 0.4), (0.1, 1.02, 0.3, 0.2)]
GEOM_IDS = [f"rho{r}-b{b}" for r, b, _, _ in GEOMETRIES]


@pytest.mark.parametrize("rho,b,u1,u2", GEOMETRIES, ids=GEOM_IDS)
def test_transit_light_curve(rho, b, u1, u2):
    p = {**BASE, "rho": rho, "b": b, "u1": u1, "u2": u2}
    got = np.asarray(ex.transit_light_curve(p, jnp.asarray(t)))
    np.testing.assert_allclose(got, batman_transit(rho, b, (u1, u2)), atol=ATOL, rtol=0)


@pytest.mark.parametrize("rho,b,u1,u2", GEOMETRIES, ids=GEOM_IDS)
def test_transit_light_curve_no_ld(rho, b, u1, u2):
    # luas returns the flux change (0 out of transit), batman the flux (1 out of transit)
    p = {**BASE, "rho": rho, "b": b}
    got = np.asarray(ex.transit_light_curve_no_ld(p, jnp.asarray(t)))
    np.testing.assert_allclose(got, batman_transit(rho, b, (), limb_dark="uniform") - 1, atol=ATOL, rtol=0)


@pytest.mark.parametrize("rho,b,u1,u2", GEOMETRIES, ids=GEOM_IDS)
def test_eclipse_light_curve(rho, b, u1, u2):
    # normalised eclipse shape: -1 at full eclipse, 0 outside
    p = {**BASE, "rho": rho, "b": b}
    got = np.asarray(eclipse_light_curve(p, jnp.asarray(t)))
    want = (batman_transit(rho, b, (), limb_dark="uniform") - 1) / rho**2
    np.testing.assert_allclose(got, want, atol=ATOL / rho**2, rtol=0)


def test_transit_2D():
    # per-wavelength depth, Kipping limb darkening and linear baseline
    p = {**BASE, "b": 0.5, "d": jnp.array([0.008, 0.01, 0.012]), "q1": jnp.array([0.2, 0.36, 0.5]),
         "q2": jnp.array([0.3, 0.416, 0.2]), "Foot": jnp.array([1., 0.99, 1.01]), "Tgrad": jnp.array([0., 1e-4, -2e-4])}
    got = np.asarray(ex.transit_2D(p, (jnp.arange(3), jnp.asarray(t))))
    u1, u2 = ld_from_kipping(p["q1"], p["q2"])
    for i in range(3):
        want = batman_transit(float(jnp.sqrt(p["d"][i])), 0.5, (float(u1[i]), float(u2[i])))
        want = want * (float(p["Foot"][i]) + 24 * float(p["Tgrad"][i]) * (t - T0))
        np.testing.assert_allclose(got[i], want, atol=ATOL, rtol=0, err_msg=f"wavelength {i}")


def test_eclipse_2D():
    # luas: 1 out of eclipse, 1 - fp_fs in full eclipse, centred on T0 for a circular orbit
    fp = jnp.array([300e-6, 500e-6])
    p = {**BASE, "rho": 0.1, "b": 0.5, "fp_fs": fp, "Foot": jnp.ones(2), "Tgrad": jnp.zeros(2)}
    got = np.asarray(ex.eclipse_2D(p, (jnp.arange(2), jnp.asarray(t))))
    for i in range(2):
        pb = batman_params(0.1, 0.5, limb_dark="uniform")
        pb.fp, pb.t_secondary = float(fp[i]), T0
        want = batman.TransitModel(pb, t, transittype="secondary").light_curve(pb) - pb.fp
        np.testing.assert_allclose(got[i], want, atol=ATOL, rtol=0, err_msg=f"wavelength {i}")
