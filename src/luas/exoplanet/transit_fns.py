import jax
import jax.numpy as jnp
import jaxoplanet
from typing import Tuple
from ..luas_types import PyTree, JAXArray
from astropy.constants import M_sun, R_sun, G
import astropy.units as u

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

        I(\mu) = 1 - u_1(1 - \mu) - u_2(1 - \mu)^2

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

        I(\mu) = 1 - u_1(1 - \mu) - u_2(1 - \mu)^2

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
def transit_light_curve(par, t, **kwargs):
    r"""Uses the package `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_ to calculate
    transit light curves using JAX assuming quadratic limb darkening and a simple circular orbit.

    Note:
        If using this function then make sure to cite `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_
        separately as it is an independent package from luas.

    This particular function will only compute a single transit light curve but JAX's vmap function
    can be used to calculate the transit light curve of multiple wavelength bands at once (see ``transit_2D``).

    The central transit time ``T0``, period ``P``, semi-major axis to stellar radius ratio a/R* = ``a`` and
    impact parameter ``b`` should be scalars.

    The transit is multiplied by a linear baseline, ``Foot + Tgrad * 24 * (t - T0)``, so with ``t`` in days
    ``Tgrad`` is the gradient in flux per hour.

    .. code-block:: python

        >>> from luas.exoplanet import transit_light_curve
        >>> import jax.numpy as jnp
        >>> par = {
        ...     "T0":0.,                 # Central transit time (days)
        ...     "P":3.4,                 # Period (days)
        ...     "a":8.2,                 # Semi-major axis to stellar ratio (aka a/R*)
        ...     "rho":0.1,               # Radius ratio (aka Rp/R* or rho)
        ...     "b":0.5,                 # Impact parameter
        ...     # Uses standard quadratic limb darkening parameterisation:
        ...     # I(mu) = 1 - u1(1 - mu) - u2(1 - mu)^2
        ...     "u1":0.5,                # First quadratic limb darkening coefficient
        ...     "u2":0.1,                # Second quadratic limb darkening coefficient
        ...     "Foot":1.,               # Baseline flux out of transit
        ...     "Tgrad":0.               # Gradient in baseline flux (hrs^-1)
        ... }
        >>> t = jnp.linspace(-0.1, 0.1, 100)
        >>> flux = transit_light_curve(par, t)

    Args:
        par (PyTree): The transit parameters stored in a PyTree/dictionary (see example above).
        t (JAXArray): Array of times (in days) to calculate the light curve at.
        **kwargs: Passed to ``jaxoplanet.light_curves.limb_dark.light_curve``.

    Returns:
        JAXArray: Array of flux values for each time input, including the baseline.

    """

    # Calculates stellar density in kg/m^3 using par["a"] = a/R*
    # Can modify this function to explicitly fit for the stellar density if desired
    rho_s = 3*jnp.pi*par["a"]**3/(G.value*(par["P"]*86400)**2)

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
        period=par["P"],
        time_transit=par["T0"],
        radius=par["rho"],
        impact_param=par["b"],
        eccentricity=0., # Can optionally include eccenticity here
        omega_peri = 0.,
    )

    # Creates an orbit object with both the central `star` object and the `body` planet object
    orbit = jaxoplanet.orbits.keplerian.OrbitalBody(central = central, body = body)

    # Define light curve function `lc` using the quadratic limb darkening coefficients u1 and u2
    lc = jaxoplanet.light_curves.limb_dark.light_curve(orbit, [par["u1"], par["u2"]], **kwargs)

    # Calculates the transit light curve flux dip with a baseline of one (default from jaxoplanet is zero)
    flux = 1 + lc(t)

    # Scales the transit model with a linear baseline
    baseline = par["Foot"] + 24*par["Tgrad"]*(t - par["T0"])
    
    return baseline*flux




def transit_2D(p: PyTree, X: tuple, **kwargs) -> JAXArray:
    r"""Uses ``jax.vmap`` on the ``transit_light_curve`` function to generate a 2D ``JAXArray`` of
    transit light curves for multiple wavelengths simultaneously.

    This is just meant to be a simple example for generating multiple simultaneous light curves
    in wavelength, it should be easy to modify for different limb darkening parameterisations, etc.
    See the package `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_ to see the range of
    currently implemented light curve models.

    The parameters ``T0``, ``P``, ``a`` and ``b`` are shared between all wavelengths, while ``d``, ``q1``,
    ``q2``, ``Foot`` and ``Tgrad`` have one value per wavelength. The same time points are used for every
    wavelength.

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
        ...     "T0":0.,                    # Central transit time (days)
        ...     "P":3.4,                    # Period (days)
        ...     "a":8.2,                    # Semi-major axis to stellar ratio (aka a/R*)
        ...     "d":0.01*jnp.ones(N_l),     # Transit depth (aka (Rp/R*)^2 or rho^2)
        ...     "b":0.5,                    # Impact parameter
        ...     # Kipping (2013) limb darkening parameterisation is used
        ...     "q1":0.36*jnp.ones(N_l),    # First quadratic limb darkening coefficient for each wv
        ...     "q2":0.416*jnp.ones(N_l),   # Second quadratic limb darkening coefficient for each wv
        ...     "Foot":1.*jnp.ones(N_l),    # Baseline flux out of transit for each wv
        ...     "Tgrad":0.*jnp.ones(N_l),   # Gradient in baseline flux for each wv (hrs^-1)
        ... }
        >>> x_l = jnp.linspace(4000, 7000, N_l)
        >>> x_t = jnp.linspace(-0.1, 0.1, 100)
        >>> X = (x_l, x_t)
        >>> flux = transit_2D(par, X)

    Args:
        p (PyTree): The transit parameters stored in a PyTree/dictionary (see example above).
        X (tuple): A tuple containing the wavelength and time arrays. Only the time array ``X[1]`` is used,
            the number of wavelengths comes from the length of the per-wavelength parameters.
        **kwargs: Passed to ``jaxoplanet.light_curves.limb_dark.light_curve``.

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

    transit_fn_w_kwargs = lambda p, x_t: transit_light_curve(p, x_t, **kwargs)

    """vmap of transit_light_curve function which assumes separate values of rho, c1, c2, Foot, Tgrad for each wavelength
    but uses shared values of T0, P, a, b for all wavelengths and assumes the same time points for all wavelengths
    """
    transit_light_curve_vmap = jax.vmap(transit_fn_w_kwargs,
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

    
    # Use the vmap of transit_light_curve to calculate a 2D array of shape (M, N) of flux values
    # For M wavelengths and N time points.
    return transit_light_curve_vmap(mfp, X[1])



def transit_light_curve_no_ld(par, t, **kwargs):
    r"""Uses the package `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_ to calculate the
    transit of a uniform (not limb darkened) stellar disc for a simple circular orbit, without any baseline.
    Used by ``eclipse_light_curve``.

    Note:
        If using this function then make sure to cite `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_
        separately as it is an independent package from luas.

    Args:
        par (PyTree): The parameters ``T0``, ``P``, ``a``, ``rho`` and ``b`` stored in a PyTree/dictionary (see
            ``transit_light_curve``).
        t (JAXArray): Array of times (in days) to calculate the light curve at.
        **kwargs: Passed to ``jaxoplanet.light_curves.limb_dark.light_curve``.

    Returns:
        JAXArray: The change in flux relative to out of transit at each time, which is zero out of transit and
        :math:`-\rho^2` when the planet is fully in front of the star.

    """

    # Calculates stellar density in kg/m^3 using par["a"] = a/R*
    # Can modify this function to explicitly fit for the stellar density if desired
    rho_s = 3*jnp.pi*par["a"]**3/(G.value*(par["P"]*86400)**2)

    central = jaxoplanet.orbits.keplerian.Central(density=rho_s/solar_density,radius=1.)

    # Define the planetary body
    body = jaxoplanet.orbits.keplerian.Body(
        period=par["P"],
        time_transit=par["T0"],
        radius=par["rho"],
        impact_param=par["b"],
        eccentricity=0., # Can optionally include eccenticity here
        omega_peri = 0.,
    )

    # Creates an orbit object with both the central `star` object and the `body` planet object
    orbit = jaxoplanet.orbits.keplerian.OrbitalBody(central = central, body = body)

    # Define light curve function `lc` using the quadratic limb darkening coefficients u1 and u2
    lc = jaxoplanet.light_curves.limb_dark.light_curve(orbit, [0., 0.], **kwargs)
    
    return lc(t)


def eclipse_light_curve(par, t, **kwargs):
    r"""Uses the package `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_ to calculate the shape of a
    secondary eclipse for a uniformly bright planet on a circular orbit, normalised to an eclipse depth of one.

    The fraction of the planet hidden behind the star during eclipse is the uniform disc transit light curve
    (see ``transit_light_curve_no_ld``) divided by :math:`\rho^2`, so ``T0`` should be set to the time of
    mid-eclipse. Light travel time across the orbit is not included.

    Note:
        If using this function then make sure to cite `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_
        separately as it is an independent package from luas.

    Args:
        par (PyTree): The parameters ``T0`` (time of mid-eclipse), ``P``, ``a``, ``rho`` and ``b`` stored in a
            PyTree/dictionary (see ``transit_light_curve``).
        t (JAXArray): Array of times (in days) to calculate the light curve at.
        **kwargs: Passed to ``jaxoplanet.light_curves.limb_dark.light_curve``.

    Returns:
        JAXArray: The change in flux at each time for an eclipse depth of one, which is zero out of eclipse
        and -1 when the planet is fully eclipsed.

    """
    return transit_light_curve_no_ld(par, t, **kwargs)/par["rho"]**2


def eclipse_2D(mfp, X, **kwargs):
    r"""Uses the package `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_ (through
    ``eclipse_light_curve``) to generate a 2D ``JAXArray`` of secondary eclipse light curves for multiple
    wavelengths simultaneously.

    Note:
        If using this function then make sure to cite `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_
        separately as it is an independent package from luas.

    A single eclipse shape is calculated with ``eclipse_light_curve`` and scaled by the planet-to-star flux
    ratio at each wavelength, then multiplied by a linear baseline for each wavelength:

    .. math::

        F(\lambda, t) = \left[F_\mathrm{oot}(\lambda) + T_\mathrm{grad}(\lambda)\,(t - \bar{t})\right]
        \left[1 + \frac{F_p}{F_s}(\lambda)\,E(t)\right]

    where :math:`E(t)` is the normalised eclipse shape and :math:`\bar{t}` is the mean of the times. The flux
    is therefore one out of eclipse (before the baseline) and :math:`1 - F_p/F_s` during full eclipse.

    Note:
        Unlike ``transit_2D``, the baseline gradient ``Tgrad`` is per unit of the input times (e.g. per day),
        and the baseline is centred on the mean time rather than ``T0``.

    .. code-block:: python

        >>> from luas.exoplanet import eclipse_2D
        >>> import jax.numpy as jnp
        >>> N_l = 16 # Number of wavelength channels
        >>> par = {
        ...     "T0":0.,                        # Time of mid-eclipse (days)
        ...     "P":3.4,                        # Period (days)
        ...     "a":8.2,                        # Semi-major axis to stellar ratio (aka a/R*)
        ...     "rho":0.1,                      # Radius ratio (aka Rp/R* or rho), sets the eclipse shape
        ...     "b":0.5,                        # Impact parameter
        ...     "fp_fs":1e-3*jnp.ones(N_l),     # Planet-to-star flux ratio for each wv
        ...     "Foot":1.*jnp.ones(N_l),        # Baseline flux for each wv
        ...     "Tgrad":0.*jnp.ones(N_l),       # Gradient in baseline flux for each wv (per day)
        ... }
        >>> X = (jnp.linspace(4000, 7000, N_l), jnp.linspace(-0.1, 0.1, 100))
        >>> flux = eclipse_2D(par, X)

    Args:
        mfp (PyTree): The eclipse parameters stored in a PyTree/dictionary (see example above).
        X (tuple): A tuple containing the wavelength and time arrays. Only the time array ``X[1]`` is used.
        **kwargs: Passed to ``jaxoplanet.light_curves.limb_dark.light_curve``.

    Returns:
        JAXArray: 2D array of flux values in a wavelength by time grid of shape ``(N_l, N_t)``.

    """
    x_t = X[1]
    t_baseline = x_t - x_t.mean()

    # only generate one transit light curve and rescale for all wavelengths with jnp.outer
    normalised_transit = eclipse_light_curve(mfp, x_t, **kwargs)
    eclipse_lc = 1. + jnp.outer(mfp["fp_fs"], normalised_transit)

    # baseline functions can also be outer products
    baseline = jnp.outer(mfp["Tgrad"], t_baseline) + jnp.outer(mfp["Foot"], jnp.ones(x_t.size))

    return baseline * eclipse_lc

