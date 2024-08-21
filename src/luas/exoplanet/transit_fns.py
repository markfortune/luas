import jax
import jax.numpy as jnp
import jaxoplanet
from typing import Tuple
from ..luas_types import PyTree, JAXArray
from astropy.constants import M_sun, R_sun, G
import astropy.units as u
jax.config.update("jax_enable_x64", True)

__all__ = [
    "ld_to_kipping",
    "ld_from_kipping",
    "transit_light_curve",
    "transit_2D",
]

def ld_to_kipping(u1: JAXArray, u2: JAXArray) -> Tuple[JAXArray, JAXArray]:
    r"""Converts quadratic limb darkening parameters to the `Kipping (2013) <https://arxiv.org/abs/1308.0009>`_
    parameterisation from the standard quadratic limb darkening coefficients:
    
    .. math::
        I(r) = 1 − u_1(1 − \mu) − u_2(1 − \mu)^2
        
    Values are converted using:
    
    .. math::
        q_1 = (u_1 + u_2)^2
        
    .. math::
        q_2 = \frac{u_1}{2(u_1 + u_2)}
    
    Args:
        u1 (JAXArray): First quadratic limb darkening coefficient(s).
        u2 (JAXArray): Second quadratic limb darkening coefficient(s).
        
    Returns:
        (JAXArray, JAXArray): The first and second quadratic limb darkening coefficient(s)
        in the Kipping (2013) parameterisation.
        
    """
    
    u1_u2_sum = u1+u2
    q1 = u1_u2_sum**2
    q2 = u1/(2*u1_u2_sum)
    return q1, q2


def ld_from_kipping(q1: JAXArray, q2: JAXArray) -> Tuple[JAXArray, JAXArray]:
    r"""Converts limb darkening parameters from the `Kipping (2013) <https://arxiv.org/abs/1308.0009>`_
    parameterisation to the standard quadratic limb darkening coefficients:
    
    .. math::
        I(r) = 1 − u_1(1 − \mu) − u_2(1 − \mu)^2
        
    Values are converted using:
    
    .. math::
        u_1 = 2 \sqrt{q_1} q_2
        
    .. math::
        u_2 = \sqrt{q_1} (1 - 2 q_2)
    
    Args:
        q1 (JAXArray): The first limb darkening coefficient(s).
        q2 (JAXArray): The second limb darkening coefficient(s).
        
    Returns:
        (JAXArray, JAXArray): The first and second quadratic limb darkening coefficient(s)
        in the standard parameterisation.
        
    """
    
    q1_sqrt = jnp.sqrt(q1)
    u1 = 2*q1_sqrt*q2
    u2 = q1_sqrt - u1
    return u1, u2


# Calculates solar density in kg/m^3 to convert between fitting for stellar density or a/R*
solar_density = ((M_sun/R_sun**3)/(u.kg/u.m**3)).si
def transit_light_curve(par, t):
    """Uses the package `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_ to calculate
    transit light curves using JAX assuming quadratic limb darkening and a simple circular orbit.
    
    Note:
        If using this function then make sure to cite jaxoplanet separately as it is an independent
        package from luas.
    
    This particular function will only compute a single transit light curve but JAX's vmap function
    can be used to calculate the transit light curve of multiple wavelength bands at once.
    
    It assumes that the central transit time `T0`, period `P`, semi-major axis to stellar ratio a/R* = `a`
    and impact parameter `b` are size-1 arrays as this makes it easier to implement with vmap
    and PyMC. Feel free to vary this however, for example it can easily be modified to fit for T0 separately
    for each light curve.
    
    .. code-block:: python

        >>> from luas.exoplanet import transit_light_curve
        >>> import jax.numpy as jnp
        >>> par = {
        >>> ... "T0":0.*jnp.ones(1),     # Central transit time (days)
        >>> ... "P":3.4*jnp.ones(1),     # Period (days)
        >>> ... "a":8.2*jnp.ones(1),     # Semi-major axis to stellar ratio (aka a/R*)
        >>> ... "rho":0.1,               # Radius ratio (aka Rp/R* or rho)
        >>> ... "b":0.5*jnp.ones(1),     # Impact parameter
        >>> ... # Uses standard quadratic limb darkening parameterisation:
        >>> ... # I(r) = 1 − u1(1 − mu) − u2(1 − mu)^2
        >>> ... "u1":0.5,                # First quadratic limb darkening coefficient
        >>> ... "u2":0.1,                # Second quadratic limb darkening coefficient
        >>> ... "Foot":1.,               # Baseline flux out of transit
        >>> ... "Tgrad":0.               # Gradient in baseline flux (hrs^-1)
        >>> }
        >>> t = jnp.linspace(-0.1, 0.1, 100)
        >>> flux = transit_light_curve(par, t)
    
    Args:
        par (PyTree): The transit parameters stored in a PyTree/dictionary (see example above).
        t (JAXArray): Array of times to calculate the light curve at.
            
    Returns:
        JAXArray: Array of flux values for each time input.
        
    """

    # Calculates stellar density in kg/m^3 using par["a"] = a/R*
    # Can modify this function to explicitly fit for the stellar density if desired
    rho_s = 3*jnp.pi*par["a"][0]**3/(G.value*(par["P"][0]*86400)**2)

    # Creates an object describing the star
    # This code actually sets the stellar radius as 1 solar radius
    # It gives the density relative to solar density
    # This actually gives a different mass for the central star but this does not affect the transit model
    # It effectively just creates an analogous system scaled in distance by a factor R_sun/R*
    # This avoids having to input a value for R* which is irrelevant for transit calculations
    # This does not affect a/R*, Rp/R* or b as they are all dimensionless quantities
    central = jaxoplanet.orbits.keplerian.Central(density=rho_s/solar_density,radius=1.)

    # Define the planetary body
    body = jaxoplanet.orbits.keplerian.Body(
        period=par["P"][0],
        time_transit=par["T0"][0],
        radius=par["rho"],
        impact_param=par["b"][0],
        eccentricity=0., # Can optionally include eccenticity here
        omega_peri = 0.,
    )

    # Creates an orbit object with both the central `star` object and the `body` planet object
    orbit = jaxoplanet.orbits.keplerian.OrbitalBody(central = central, body = body)

    # Define light curve function `lc` using the quadratic limb darkening coefficients u1 and u2
    lc = jaxoplanet.light_curves.limb_dark.light_curve(orbit, [par["u1"], par["u2"]])

    # Calculates the transit light curve flux dip with a baseline of one (default from jaxoplanet is zero)
    flux = 1 + lc(t)

    # Scales the transit model with a linear baseline
    baseline = par["Foot"] + 24*par["Tgrad"]*(t - par["T0"][0])
    
    return baseline*flux



"""vmap of transit_light_curve function which assumes separate values of rho, c1, c2, Foot, Tgrad for each wavelength
but uses shared values of T0, P, a, b for all wavelengths and assumes the same time points for all wavelengths
"""
transit_light_curve_vmap = jax.vmap(transit_light_curve,
                                    in_axes=({
                                        # Parameters to be shared between all wavelengths
                                        "T0":None, "P":None, "a":None, "b":None, 
                                        
                                        # Parameters to be separate for each wavelength
                                        "rho":0, "u1":0, "u2":0, "Foot":0, "Tgrad":0}, 
                                        
                                        # Array of timestamps to be the same for each wavelength
                                        None, 
                                        ), 
                                    # Will output extra flux values for each light curve as additional rows
                                    out_axes = 0 
                                   ) 


def transit_2D(p: PyTree, x_l: JAXArray, x_t: JAXArray) -> JAXArray:
    r"""Uses ``jax.vmap`` on the ``transit_light_curve`` function to generate a 2D ``JAXArray`` of
    transit light curves for multiple wavelengths simultaneously.
    
    This is just meant to be a simple example for generating multiple simultaneous light curves
    in wavelength, it should be easy to modify for different limb darkening parameterisations, etc.
    See the package `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_ to see the range of
    currently implemented light curve models.
    
    Note:
        Unlike ``transit_light_curve``, input limb darkening parameters are assumed to follow
        the `Kipping (2013) <https://arxiv.org/abs/1308.0009>`_ parameterisation and are converted
        to standard limb darkening coefficients. Also assumed that the transit depth d = rho^2 is
        being input which is then converted to radius ratio values for ``transit_light_curve``.
    
    .. code-block:: python

        >>> from luas.exoplanet import transit_2D
        >>> import jax.numpy as jnp
        >>> N_l = 16 # Number of wavelength channels
        >>> par = {
        >>> ... "T0":0.*jnp.ones(1),        # Central transit time (days)
        >>> ... "P":3.4*jnp.ones(1),        # Period (days)
        >>> ... "a":8.2*jnp.ones(1),        # Semi-major axis to stellar ratio (aka a/R*)
        >>> ... "d":0.01*jnp.ones(N_l),     # Transit depth (aka (Rp/R*)^2 or rho^2)
        >>> ... "b":0.5*jnp.ones(1),        # Impact parameter
        >>> ... # Kipping (2013) limb darkening parameterisation is used
        >>> ... "q1":0.36*jnp.ones(N_l),    # First quadratic limb darkening coefficient for each wv
        >>> ... "q2":0.416*jnp.ones(N_l),   # Second quadratic limb darkening coefficient for each wv
        >>> ... "Foot":1.*jnp.ones(N_l),    # Baseline flux out of transit for each wv
        >>> ... "Tgrad":0.*jnp.ones(N_l),   # Gradient in baseline flux for each wv (hrs^-1)
        >>> }
        >>> x_l = jnp.linspace(4000, 7000, N_l)
        >>> x_t = jnp.linspace(-0.1, 0.1, 100)
        >>> flux = transit_2D(par, x_l, x_t)
    
    Args:
        par (PyTree): The transit parameters stored in a PyTree/dictionary (see example above).
        x_l (JAXArray): Array of wavelengths, not used but included for compatibility with :class:`luas.GP`.
        x_t (JAXArray): Array of times to calculate the light curve at.
            
    Returns:
        JAXArray: 2D array of flux values in a wavelength by time grid of shape ``(N_l, N_t)``.
        
    """
    
    # vmap requires that we only input the parameters which have been explicitly defined how they vectorise
    transit_params = ["T0", "P", "a", "b", "Foot", "Tgrad"]
    mfp = {k:p[k] for k in transit_params}
    
    # Calculate the radius ratio rho from the transit depth d
    mfp["rho"] = jnp.sqrt(p["d"])
    
    # Calculate limb darkening coefficients from the Kipping (2013) parameterisation.
    mfp["u1"], mfp["u2"] = ld_from_kipping(p["q1"], p["q2"])
    
    # Use the vmap of transit_light_curve to calculate a 2D array of shape (M, N) of flux values
    # For M wavelengths and N time points.
    return transit_light_curve_vmap(mfp, x_t)

