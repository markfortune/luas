import jax.numpy as jnp
import numpy as np
import scipy.linalg
from scipy.special import gammaln


def state_space_matrices_from_coeffs(h_coeffs):
    r"""Builds the state space matrices of a linear stochastic differential equation from the coefficients of its
    characteristic polynomial, :math:`s^m + h_{m-1}s^{m-1} + \dots + h_0`.

    Args:
        h_coeffs (JAXArray): The coefficients :math:`(h_0, h_1, \dots, h_{m-1})`.

    Returns:
        (JAXArray, JAXArray): The feedback matrix :math:`\mathbf{F}` in companion form, and the stationary
        covariance :math:`\mathbf{P}_\infty` (from the continuous Lyapunov equation, using ``scipy``) normalised
        so its first element is one.

    """
    order = len(h_coeffs)

    F_ref = jnp.diag(jnp.ones(order-1), k = 1)
    F_ref = F_ref.at[-1, :].set(-h_coeffs)

    LqL_T = jnp.zeros((order, order)).at[-1, -1].set(1.)

    P_inf_ref = jnp.array(scipy.linalg.solve_continuous_lyapunov(F_ref, -LqL_T))
    P_inf_ref /= P_inf_ref[0, 0]

    return F_ref, P_inf_ref


def modal_decomp_SDE(h_coeffs):
    r"""Calculates the modal decomposition of a linear stochastic differential equation, which writes it as a sum
    of independent modes: one for each real root of the characteristic polynomial and one for each complex
    conjugate pair.

    Args:
        h_coeffs (JAXArray): The coefficients :math:`(h_0, h_1, \dots, h_{m-1})` of the characteristic polynomial.

    Returns:
        tuple: ``(T_modal, T_modal_inv, all_real_parts, imag_parts, F_ref, P_inf_ref)``: the (real) transformation
        to the modal basis and its inverse, the real parts of the roots (real roots first), the imaginary parts
        of the complex roots, and the feedback matrix and stationary covariance from
        :func:`state_space_matrices_from_coeffs`.

    """

    order = len(h_coeffs)

    # Calculate transfer function matrix and stationary covariance of reference SDE with length scale of 1
    F_ref, P_inf_ref = state_space_matrices_from_coeffs(h_coeffs)
    
    # Use eigendecomposition to get modal decomposition of SDE
    lam, eigenvecs = jnp.linalg.eig(F_ref.T)

    # Need to get conjugate pairs of roots
    # First sort by real part, then sort complex roots by magnitude of imaginary part to get conjugate pairs together
    sort_idx = jnp.argsort(jnp.real(lam))
    lam = lam[sort_idx]
    eigenvecs = eigenvecs[:, sort_idx]

    tol = 1e-10

    # First get real roots and their eigenvectors
    real_roots_ind = jnp.abs(jnp.imag(lam)) < tol
    real_roots = jnp.real(lam[real_roots_ind])
    real_eigenvecs = jnp.real(eigenvecs[:, real_roots_ind])

    # Get remaining roots and sort by magnitude of imaginary part to get conjugate pairs together
    complex_roots = lam[~real_roots_ind]
    complex_eigenvecs = eigenvecs[:, ~real_roots_ind]

    complex_roots_imag = jnp.imag(complex_roots)
    complex_sort_ind = jnp.argsort(jnp.abs(complex_roots_imag))
    complex_roots_sorted = complex_roots[complex_sort_ind]
    complex_eigenvecs_sorted = complex_eigenvecs[:, complex_sort_ind]

    # Store real and imaginary parts of first of each conjugate pair
    complex_sort_ind_firsts = jnp.arange(0, complex_roots_sorted.size, 2)
    complex_roots_real_sorted = jnp.real(complex_roots_sorted[complex_sort_ind_firsts])
    complex_roots_imag_sorted = jnp.imag(complex_roots_sorted[complex_sort_ind_firsts])

    complex_eigenvecs_real = jnp.real(complex_eigenvecs_sorted[:, complex_sort_ind_firsts])
    complex_eigenvecs_imag = jnp.imag(complex_eigenvecs_sorted[:, complex_sort_ind_firsts])

    all_complex_eigenvecs = jnp.concatenate([complex_eigenvecs_real, complex_eigenvecs_imag], axis = 1)
    
    # Form the interlacing shuffle to get real and imaginary parts of each conjugate pair together
    resort_ind = jnp.arange(all_complex_eigenvecs.shape[1])
    resort_ind = resort_ind.reshape((all_complex_eigenvecs.shape[1]//2, 2), order = "F").ravel("C")

    # Need to group conjugate pairs together, first stack each block then interlace real and imaginary parts
    all_complex_eigenvecs = all_complex_eigenvecs[:, resort_ind]

    # T_modal now has real roots first, then each conjugate pair's real and imaginary parts interlaced
    T_modal = jnp.concatenate([real_eigenvecs, all_complex_eigenvecs], axis = 1)


    T_modal_inv = jnp.linalg.inv(T_modal)

    # These values are repeated for each conjugate pair, so take real roots followed by the first of each conjugate pair
    all_real_parts = jnp.concatenate([real_roots, complex_roots_real_sorted])
    imag_parts = complex_roots_imag_sorted.copy()

    return T_modal, T_modal_inv, all_real_parts, imag_parts, F_ref, P_inf_ref


def get_stable_roots(power_spec_coeffs, order):
    r"""Returns the stable roots (negative real part) of a power spectrum polynomial. If there aren't exactly
    ``order`` of them, the ``order`` roots with the smallest real parts are used.

    Args:
        power_spec_coeffs (JAXArray): Coefficients of the polynomial, highest power first.
        order (int): The number of roots to return.

    Returns:
        JAXArray: The stable roots.

    """
    roots = jnp.roots(power_spec_coeffs, strip_zeros=False)
    
    tol = 1e-10
    stable = roots[jnp.real(roots) < -tol]
    if stable.size != order:
        idx = jnp.argsort(np.real(roots))
        stable = roots[idx[:order]]

    return stable

def stoc_diff_coeffs_from_roots(stable):
    r"""Returns the coefficients :math:`(h_0, h_1, \dots, h_{m-1})` of the monic polynomial with the given roots,
    i.e. the characteristic polynomial of the corresponding stochastic differential equation.

    Args:
        stable (JAXArray): The roots.

    Returns:
        JAXArray: The real coefficients, lowest power first, without the leading one.

    """
    monic_desc = jnp.poly(stable)  # [1, a_{p-1}, ..., a0]
    # monic_desc = np.real_if_close(monic_desc, tol=1e4).astype(np.float64)
    monic_desc = jnp.real(monic_desc)
    return monic_desc[1:][::-1]


def get_modal_decomp(ps_coeffs):
    r"""Calculates the modal decomposition of the stochastic differential equation whose power spectrum has the
    given denominator polynomial, by taking its stable roots (see :func:`get_stable_roots` and
    :func:`modal_decomp_SDE`).

    Args:
        ps_coeffs (JAXArray): Coefficients of the power spectrum denominator, highest power first.

    Returns:
        tuple: The outputs of :func:`modal_decomp_SDE`.

    """
    order = len(ps_coeffs)//2
    stable = get_stable_roots(ps_coeffs, order)
    stoc_diff_coeffs = stoc_diff_coeffs_from_roots(stable)

    T_modal, T_modal_inv, all_real_parts, imag_parts, F_ref, P_inf_ref = modal_decomp_SDE(stoc_diff_coeffs)

    return T_modal, T_modal_inv, all_real_parts, imag_parts, F_ref, P_inf_ref


def calc_sq_exp_denom_ps_coeffs(order):
    r"""Calculates the polynomial used to approximate the squared exponential power spectrum: the Taylor
    expansion of :math:`e^{\omega^2/2}` up to order ``order`` in :math:`\omega^2`, written in terms of
    :math:`s` with :math:`\omega^2 = -s^2` and scaled by :math:`2^{\mathrm{order}}\,\mathrm{order}!`. Used to
    precompute the modal decompositions of ``SquaredExpApprox``.

    Args:
        order (int): The order of the approximation.

    Returns:
        JAXArray: The ``2 * order + 1`` coefficients, highest power first.

    """
    m = jnp.arange(order, -1, -1)

    log_coeffs = gammaln(order + 1) - gammaln(m + 1) + (order - m) * jnp.log(2)
    all_coeffs = jnp.zeros(2*order + 1)
    all_coeffs = all_coeffs.at[::2].set(jnp.power(-1., m) * jnp.exp(log_coeffs))
    
    return all_coeffs


def create_modal_decomps(ps_coeff_fn, orders, filename):
    """
    Example usage: create_modal_decomps(calc_sq_exp_denom_ps_coeffs, np.arange(2, 20, 2), "sq_exp_modal_decomps.npz")
    """
    
    modal_decomps = {}

    for order in orders:
        ps_coeffs = ps_coeff_fn(order)
        T_modal, T_modal_inv, all_real_parts, imag_parts, F_ref, P_inf_ref = get_modal_decomp(ps_coeffs)

        modal_decomps[order] = {"T_modal":T_modal,
                                "T_modal_inv":T_modal_inv,
                                "all_real_parts":all_real_parts,
                                "imag_parts":imag_parts,
                                "F_ref":F_ref,
                                "P_inf_ref":P_inf_ref,
        }

    #Save — flatten to "{order}__{name}" keys
    flat = {}
    for order, decomp in modal_decomps.items():
        for name, arr in decomp.items():
            flat[f"{order}__{name}"] = arr

    np.savez_compressed(filename, **flat)


def h_coeffs_matern(double_nu):
    """
    double_nu is the numerator of nu = double_nu/2, where nu is the smoothness parameter of the Matern kernel
    Only supports odd positive integer double_nu, (i.e. 1, 3, 5, ...) corresponding to half-integer nu (1/2, 3/2, 5/2, ...)
    """

    p = (double_nu - 1)//2

    lam = 1 # jnp.sqrt(double_nu)
    i = jnp.arange(p+1, 0, -1)

    log_coeffs = gammaln(p + 2) - gammaln(i + 1) - gammaln(p + 2 - i) + i * jnp.log(lam)

    return jnp.exp(log_coeffs)


def create_matern_state_space_matrices(double_nus, filename):
    """
    Precomputes the state space matrices for the SDE representation of Matern kernels with different smoothness parameters, and saves them to a file. 
    Useful as these calculations don't need to be repeated for different length scales and avoids non-JIT friendly Lyaponuv calculations
    Matrices are calculated assuming a reference length scale of 1.
    Example usage: create_matern_state_space_matrices(np.arange(1, 23, 2), "matern_SSMs_stable.npz")
    """
    
    state_space_matrices = {}

    for double_nu in double_nus:

        h_coeffs = h_coeffs_matern(double_nu)
        F_ref, P_inf_ref = state_space_matrices_from_coeffs(h_coeffs)

        state_space_matrices[double_nu] = {
                                "F_ref":F_ref,
                                "P_inf_ref":P_inf_ref,
        }

    #Save — flatten to "{order}__{name}" keys
    flat = {}
    for double_nu, decomp in state_space_matrices.items():
        for name, arr in decomp.items():
            flat[f"{double_nu}__{name}"] = arr

    np.savez_compressed(filename, **flat)
