import pytest
import jax.numpy as jnp
import jax
import numpy as np
from copy import deepcopy

from .kernel_functions import (
    evaluate_kernel,
    squared_exponential_kernel,
    matern32_kernel,
    rational_quadratic_kernel
)
from .GeneralKernel import GeneralKernel
from .LuasKernel import LuasKernel
from .GPClass import GP

@pytest.fixture
def rtol():
    return 1e-10

@pytest.fixture
def atol():
    return 1e-8

@pytest.fixture
def N_l():
    return 16

@pytest.fixture
def N_t():
    return 100


def mf_1D(par, x_t):
    """
    Simple transit model for cicular orbit and no limb darkening.

    par is a parameter vector:
    par = [T0,P,aRs,rho,b,foot,Tgrad]
    where:
    T0 - central transit time
    P - orbital period
    aRs - semi-major axis in units of stellar radius (system scale) or a/R_star
    rho - planet-to-star radius ratio - the parameter of interest for trasmission spectroscopy
    b - impact parameter
    x_t is time (in same units as T0 and P)

    """

    #calculate phase angle
    theta = (2*jnp.pi/par["P"]) * (x_t - par["T0"])

    #normalised separation z
    z = jnp.sqrt((par["a"]*jnp.sin(theta))**2 + (par["b"]*jnp.cos(theta))**2)

    #calculate flux
    f = jnp.ones(x_t.size)-(par["rho"]**2)*(z<=1) #fully in transit
    
    #return flux
    return f

mf_2D = jax.vmap(mf_1D, in_axes=({"T0":None, "P":None, "a":None, "rho":0, "b":None}, None), out_axes = 0)
mf_2D.mfp = ["T0", "P", "a", "rho", "b"]


def mf(par, x_l, x_t):
    mfp = {k:par[k] for k in mf_2D.mfp}
    return mf_2D(mfp, x_t)


def Kl(hp, x_l1, x_l2, wn = True):
    
    Kl = evaluate_kernel(rbf_kernel, x_l1, x_l2, hp["l_l_CM"])
    
    h_mat = jnp.diag(hp["h_CM"] * jnp.ones_like(x_l1))
    Kl = h_mat @ Kl @ h_mat
    
    Kl += jnp.diag(hp["h_WSS"]**2)
    
    return Kl
Kl.hp = ["h_CM", "h_WSS", "l_l_CM"]
Kl.diag = False


def Kt(hp, x_t1, x_t2, wn = True):
    
    Kt = evaluate_kernel(matern32_kernel, x_t1, x_t2, hp["l_t"])
    
    return Kt
Kt.hp = ["l_t"]
Kt.diag = False


def Sl(hp, x_l1, x_l2, wn = True):
    
    Sl = hp["h_HFS"]**2 * evaluate_kernel(ration_quad_kernel, x_l1, x_l2, hp["l_l_HFS"])
    
    if wn:
        Sl += jnp.diag(jnp.square(hp["sigma"]) * jnp.ones_like(x_l1))
    
    return Sl
Sl.hp = ["h_HFS", "l_l_HFS", "sigma"]
Sl.diag = False


def St(hp, x_t1, x_t2, wn = True):
    return jnp.diag(hp["wn_t"])
St.hp = ["wn_t"]
St.diag = True


@pytest.fixture
def kf_luas():
    return LuasKernel(Kl = Kl, Kt = Kt, Sl = Sl, St = St)

@pytest.fixture
def kf_general():
    return GeneralKernel(Kl_fns = [Kl, Sl], Kt_fns = [Kt, St])


@pytest.fixture
def par_GP(N_l, N_t):
    return {"T0":-0.0021*jnp.ones(1), "P":3.4059095*jnp.ones(1), "a":8.19*jnp.ones(1), "b":0.761*jnp.ones(1), "rho":0.12546*jnp.ones(N_l),
                  "h_CM":1.5e-3*jnp.ones(1), "l_t":0.011*jnp.ones(1), "l_l_CM":2201.*jnp.ones(1), "sigma":1.2e-3*jnp.ones(N_l),
                  "h_WSS":2e-4*jnp.ones(N_l), "h_HFS":3e-4*jnp.ones(1), "l_l_HFS":1000.*jnp.ones(1), "wn_t":1.1*jnp.ones(N_t)}

@pytest.fixture
def par_sim(N_l, N_t):
    return {"T0":-0.00205*jnp.ones(1), "P":3.4059091*jnp.ones(1), "a":8.186*jnp.ones(1), "b":0.762*jnp.ones(1), "rho":0.12563*jnp.ones(N_l),
                  "h_CM":1.25e-3*jnp.ones(1), "l_t":0.015*jnp.ones(1), "l_l_CM":1530.*jnp.ones(1), "sigma":1.23e-3*jnp.ones(N_l),
                  "h_WSS":1.5e-4*jnp.ones(N_l), "h_HFS":3.4e-4*jnp.ones(1), "l_l_HFS":909.*jnp.ones(1), "wn_t":1.05*jnp.ones(N_t)}


@pytest.fixture
def x_l(N_l):
    return jnp.linspace(4000., 7000., N_l)


@pytest.fixture
def x_t(N_t):
    return jnp.linspace(-0.15, 0.15, N_t)


@pytest.fixture
def M(par_sim, x_l, x_t):
    return mf(par_sim, x_l, x_t)

# @pytest.fixture
# def random():
#     return np.random.default_rng(42)

@pytest.fixture
def Y(M, x_l, x_t, par_sim, kf_luas):
    np.random.seed(42)
    R = kf_luas.generate_noise(par_sim, x_l, x_t)
    return M*(1+R)


def gp(par, x_l, x_t, Y, kf):
    
    N_l = x_l.size
    N_t = x_t.size

    # Specify parameters to vary for inference
    mfp_to_vary = ["T0", "a", "b", "rho"]         # Mean function parameters
    hp_to_vary = ["h_CM", "l_l_CM", "l_t", "h_WSS", "h_HFS", "l_l_HFS", "sigma"] # Hyperparameters

    # Specify parameters to use log priors on
    log_params = ["h_CM", "l_t", "l_l_CM", "sigma", "h_WSS", "h_HFS", "l_l_HFS"]
    
    # Specify Gaussian priors on some parameters
    prior_values = {"a":8.3452, "b":0.787354}
    prior_std = {"a":0.1, "b":0.018}

    prior_dict = {"a":[prior_values["a"], prior_std["a"]], "b":[prior_values["b"], prior_std["b"]]}
    
    initial_params = deepcopy(par)
    for name in log_params:
        initial_params[name] = jnp.log10(par[name])
    
    return GP(initial_params, mfp_to_vary, hp_to_vary, x_l, x_t, Y, kf,
               mf = mf, log_params = log_params, gaussian_prior_dict = prior_dict)


def test_logL_comparison(par_GP, x_l, x_t, Y, kf_luas, kf_general, rtol, atol):
    
    gp_luas = gp(par_GP, x_l, x_t, Y, kf_luas)
    gp_general = gp(par_GP, x_l, x_t, Y, kf_general)
    
    logL1 = gp_luas.logL(gp_luas.p)
    logL2 = gp_general.logL(gp_luas.p)
    np.testing.assert_allclose(logL1, logL2, rtol = rtol, atol = atol)
    
    grad_logL1 = gp_luas.grad_logL(gp_luas.p_arr)
    grad_logL2 = gp_general.grad_logL(gp_luas.p_arr)
    np.testing.assert_allclose(grad_logL1, grad_logL2, rtol = rtol, atol = atol)
    
    logL1, grad_logL1 = gp_luas.value_and_grad_logL(gp_luas.p_arr)
    logL2, grad_logL2 = gp_general.value_and_grad_logL(gp_luas.p_arr)
    np.testing.assert_allclose(logL1, logL2, rtol = rtol, atol = atol)
    np.testing.assert_allclose(grad_logL1, grad_logL2, rtol = rtol, atol = atol)
    
    gp_mean_luas, sigma_diag_luas, M_luas = gp_luas.predict(gp_luas.p)
    gp_mean_chol, sigma_diag_chol, M_chol = gp_general.predict(gp_luas.p)
    np.testing.assert_allclose(gp_mean_luas, gp_mean_chol, rtol = rtol, atol = atol)
    np.testing.assert_allclose(sigma_diag_luas, sigma_diag_chol, rtol = rtol, atol = atol)
    np.testing.assert_allclose(M_luas, M_chol, rtol = rtol, atol = atol)
    
    hessian_logL1 = gp_luas.hessian_logL(gp_luas.p_arr, large = False)
    hessian_logL2 = gp_general.hessian_logL(gp_luas.p_arr, large = True)
    np.testing.assert_allclose(hessian_logL1, hessian_logL2, rtol = rtol, atol = atol)
    
    large_hessian_logL1 = gp_luas.hessian_logL(gp_luas.p_arr, large = True)
    np.testing.assert_allclose(large_hessian_logL1, hessian_logL2, rtol = rtol, atol = atol)
    