import jax
import jax.numpy as jnp
from jaxoplanet.orbits import KeplerianOrbit
from jaxoplanet.light_curves import QuadLightCurve
from copy import deepcopy
from typing import Tuple
from ..luas_types import PyTree, JAXArray

jax.config.update("jax_enable_x64", True)

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
    transit_params = ["T0", "P", "a", "rho", "b", "c1", "c2", "Foot", "Tgrad"]
    
    mfp = {k:p[k] for k in transit_params}
    
    return transit_light_curve_vmap(mfp, x_t)
transit_2D.mfp = ["T0", "P", "a", "rho", "b", "c1", "c2", "Foot", "Tgrad"]


def transit_2D_multi_x_t(p: PyTree, x_l: JAXArray, x_t: JAXArray) -> JAXArray:
    transit_params = ["T0", "P", "a", "rho", "b", "c1", "c2", "Foot", "Tgrad"]
    
    mfp = {k:p[k] for k in transit_params}
    
    return transit_light_curve_vmap(mfp, x_t[0, :])
transit_2D.mfp = ["T0", "P", "a", "rho", "b", "c1", "c2", "Foot", "Tgrad"]



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


def transit_param_transform(
    p_vary: PyTree,
    p_fixed: PyTree, 
    log_params: list[str]
) -> PyTree:
    
    # Copy to avoid transformation affecting stored values
    p = deepcopy(p_fixed)
    
    # Update fixed values with values being varied
    p.update(p_vary)
    
    p["rho"] = jnp.sqrt(p["rho_2"])
    p["c1"], p["c2"] = ld_from_kipping(p["c1"], p["c2"])

    # Example of a polynomial parameterisation of a parameter being transformed to the polynomial
    # p["h_CM"] = jnp.polyval(p["h_CM"], x = jnp.linspace(-1, 1, self.N_l))

    # Transform log parameters
    for name in log_params:
        p[name] = jnp.power(10, p[name])

    return p
