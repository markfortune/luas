import jax
import jax.numpy as jnp
from jaxoplanet.orbits import KeplerianOrbit
from jaxoplanet.light_curves import QuadLightCurve
from typing import Tuple
from ..luas_types import PyTree, JAXArray

jax.config.update("jax_enable_x64", True)


def ld_to_kipping(c1: JAXArray, c2: JAXArray) -> Tuple[JAXArray, JAXArray]:
    c1_c2_sum = c1+c2
    q1 = c1_c2_sum**2
    q2 = c1/(2*c1_c2_sum)
    return q1, q2


def ld_from_kipping(q1: JAXArray, q2: JAXArray) -> Tuple[JAXArray, JAXArray]:
    q1_sqrt = jnp.sqrt(q1)
    c1 = 2*q1_sqrt*q2
    c2 = q1_sqrt - c1
    return c1, c2


def transit_light_curve(mfp: PyTree, x_t: JAXArray) -> JAXArray:
    light_curve = QuadLightCurve.init(u1=mfp["c1"], u2=mfp["c2"])
    orbit = KeplerianOrbit.init(
        time_transit=mfp["T0"],
        period=mfp["P"],
        semimajor=mfp["a"],
        impact_param=mfp["b"],
        radius=mfp["rho"],
    )
    
    flux = (mfp["Foot"] + 24*mfp["Tgrad"]*(x_t-mfp["T0"]))*(1+light_curve.light_curve(orbit, x_t)[0])
    
    return flux

transit_light_curve_vmap = jax.vmap(transit_light_curve, in_axes=({"T0":None, "P":None, "a":None, "rho":0, "b":None, "c1":0, "c2":0, "Foot":0, "Tgrad":0}, None), out_axes = 0)


def transit_2D(p: PyTree, x_l: JAXArray, x_t: JAXArray) -> JAXArray:
    transit_params = ["T0", "P", "a", "b", "Foot", "Tgrad"]

    mfp = {k:p[k] for k in transit_params}
    mfp["rho"] = jnp.sqrt(p["d"])
    mfp["c1"], mfp["c2"] = ld_from_kipping(p["u1"], p["u2"])
    
    return transit_light_curve_vmap(mfp, x_t)
