import jax
import jax.numpy as jnp
from jaxoplanet.orbits import KeplerianOrbit
from jaxoplanet.light_curves import QuadLightCurve

jax.config.update("jax_enable_x64", True)

def transit_light_curve(mf_params, x_t):
    light_curve = QuadLightCurve.init(u1=mf_params["c1"], u2=mf_params["c2"])
    orbit = KeplerianOrbit.init(
        time_transit=mf_params["T0"],
        period=mf_params["P"],
        semimajor=mf_params["a"],
        impact_param=mf_params["b"],
        radius=mf_params["rho_2"],
    )
    
    flux = (mf_params["Foot"] + 24*mf_params["Tgrad"]*(x_t-mf_params["T0"]))*(1+light_curve.light_curve(orbit, x_t)[0])
    
    return flux

transit_light_curve_vmap = jax.jit(jax.vmap(transit_light_curve, in_axes=({"T0":None, "P":None, "a":None, "rho_2":0, "b":None, "c1":0, "c2":0, "Foot":0, "Tgrad":0}, None), out_axes = 0))

def transit_2D(p, x_l, x_t):
    transit_params = ["T0", "P", "a", "rho_2", "b", "c1", "c2", "Foot", "Tgrad"]
    
    mfp = {k:p[k] for k in transit_params}
    
    return transit_light_curve_vmap(mfp, x_t)
transit_2D.mfp = ["T0", "P", "a", "rho_2", "b", "c1", "c2", "Foot", "Tgrad"]

