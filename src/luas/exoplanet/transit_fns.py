import jax
import jax.numpy as jnp
from jaxoplanet.orbits import KeplerianOrbit
from jaxoplanet.light_curves import QuadLightCurve
from typing import Tuple
from ..luas_types import PyTree, JAXArray

jax.config.update("jax_enable_x64", True)

__all__ = [
    "ld_to_kipping",
   "ld_from_kipping",
   "transit_light_curve",
   "transit_2D",
]

def ld_to_kipping(c1: JAXArray, c2: JAXArray) -> Tuple[JAXArray, JAXArray]:
    """Converts quadratic limb darkening parameters to the Kipping (2013) triangular parameterisation.
    
    Args:
        c1 (JAXArray): First quadratic limb darkening coefficient(s).
        c2 (JAXArray): Second quadratic limb darkening coefficient(s).
        
    Returns:
        JAXArray: The first limb darkening coefficient in the Kipping (2013) parameterisation.
        JAXArray: The second limb darkening coefficient in the Kipping (2013) parameterisation.
        
    """
    
    c1_c2_sum = c1+c2
    q1 = c1_c2_sum**2
    q2 = c1/(2*c1_c2_sum)
    return q1, q2


def ld_from_kipping(q1: JAXArray, q2: JAXArray) -> Tuple[JAXArray, JAXArray]:
    """Converts limb darkening parameters from the Kipping (2013) triangular parameterisation to the standard
    limb darkening coefficients.
    
    Args:
        q1 (JAXArray): The first limb darkening coefficient in the Kipping (2013) parameterisation.
        q2 (JAXArray): The second limb darkening coefficient in the Kipping (2013) parameterisation.
        
    Returns:
        JAXArray: First quadratic limb darkening coefficient(s).
        JAXArray: Second quadratic limb darkening coefficient(s).
        
    """
    
    q1_sqrt = jnp.sqrt(q1)
    c1 = 2*q1_sqrt*q2
    c2 = q1_sqrt - c1
    return c1, c2


def transit_light_curve(par: PyTree, x_t: JAXArray) -> JAXArray:
    """Uses the package jaxoplanet (https://github.com/exoplanet-dev/jaxoplanet) to calculate transit light curves using JAX
    assuming quadratic limb darkening.
    
    This particular function will only compute a single transit light curve but jax's vmap function can be used to calculate
    the transit light curve of multiple wavelength bands at once.
    
    Args:
        par (PyTree): The transit parameters stored in a PyTree/dictionary with values for:
            T0: Central transit time
            P: Period
            a: Semi-major axis to stellar radius ratio (a/Rs)
            rho: Planet-to-star radius ratio (Rp/Rs)
            b: Impact parameter
            c1: First quadratic limb-darkening parameter
            c2: Second quadratic limb-darkening parameter
            Foot: Baseline flux out-of-transit
            Tgrad: Linear trend in baseline flux
        x_t (JAXArray): Array of times to calculate the light curve at.
            
    Returns:
        JAXArray: Array of flux values for each time input.
        
    """
    
    light_curve = QuadLightCurve.init(u1=par["c1"], u2=par["c2"])
    orbit = KeplerianOrbit.init(
        time_transit=par["T0"],
        period=par["P"],
        semimajor=par["a"],
        impact_param=par["b"],
        radius=par["rho"],
    )
    
    flux = (par["Foot"] + 24*par["Tgrad"]*(x_t-par["T0"]))*(1+light_curve.light_curve(orbit, x_t)[0])
    
    return flux


"""vmap of transit_light_curve function which assumes separate values of rho, c1, c2, Foot, Tgrad for each wavelength
but uses shared values of T0, P, a, b for all wavelengths and assumes the same time points for all wavelengths
"""
transit_light_curve_vmap = jax.vmap(transit_light_curve,
                                    in_axes=({"T0":None, "P":None, "a":None, "b":None,       # Parameters assumed to be shared between all wavelengths
                                              "rho":0, "c1":0, "c2":0, "Foot":0, "Tgrad":0}, # Parameters assumed to be separate for each wavelength
                                             None,  # Array of timestamps also assumed to be the same for each wavelength
                                            ), 
                                    out_axes = 0) # Will output extra flux values for each light curve as additional rows


def transit_2D(p: PyTree, x_l: JAXArray, x_t: JAXArray) -> JAXArray:
    """Uses vmap of transit_light_curve function to generate a 2D JAXArray of transit light curves for multiple wavelengths simultaneously.
    Input limb darkening parameters are assumed to follow the Kipping (2013) parameterisation and are converted to standard limb darkening coefficients.
    Also assumed that the transit depth d = rho^2 is being input which is then converted to radius ratio values
    
    Args:
        par (PyTree): The transit parameters stored in a PyTree/dictionary with values for:
            T0: Central transit time
            P: Period
            a: Semi-major axis to stellar radius ratio (a/Rs)
            d: Transit depth (Rp/Rs)^2
            b: Impact parameter
            u1: First quadratic limb-darkening parameter in Kipping (2013) parameterisation
            u2: Second quadratic limb-darkening parameter in Kipping (2013) parameterisation
            Foot: Baseline flux out-of-transit
            Tgrad: Linear trend in baseline flux
        x_t (JAXArray): Array of times to calculate the light curve at.
            
    Returns:
        JAXArray: Array of flux values for each time input.
        
    """
    
    # vmap requires that we only input the parameters which have been explicitly defined how they vectorise
    transit_params = ["T0", "P", "a", "b", "Foot", "Tgrad"]
    mfp = {k:p[k] for k in transit_params}
    
    # Calculate the radius ratio rho from the transit depth d
    mfp["rho"] = jnp.sqrt(p["d"])
    
    # Calculate limb darkening coefficients from the Kipping (2013) parameterisation.
    mfp["c1"], mfp["c2"] = ld_from_kipping(p["u1"], p["u2"])
    
    # Use the vmap of transit_light_curve to calculate a 2D array of shape (M, N) of flux values
    # For M wavelengths and N time points.
    return transit_light_curve_vmap(mfp, x_t)
