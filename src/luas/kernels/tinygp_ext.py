from __future__ import annotations
import tinygp
from tinygp.kernels.quasisep import Quasisep, _prod_helper
from tinygp.solvers.quasisep.core import DiagQSM, StrictLowerTriQSM, StrictUpperTriQSM, SymmQSM, SquareQSM
from tinygp.noise import Noise
from tinygp.solvers.quasisep.block import Block
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from jax import lax
from jax.scipy.special import gammaln
import equinox as eqx
import numpy as np
from luas.luas_types import JAXArray, is_scalar
from luas.kernels.modal_decomp_calc import stoc_diff_coeffs_from_roots, modal_decomp_SDE
import os.path

# Module-level: load once at import, not inside JIT. Uses precomputed SSM matrices which are not JIT-friendly to calculate
_module_dir = os.path.dirname(__file__)
sq_exp_decomp_data = np.load(os.path.join(_module_dir, 'data', 'sq_exp_modal_decomps.npz'))
matern_general_decomp_data = np.load(os.path.join(_module_dir, 'data', 'matern_SSMs_stable.npz'))


def _get_sq_exp_decomp_arrays(order: int):
    """Called at init time with static order — pure Python, not traced."""
    prefix = f"{order}__"
    return {k[len(prefix):]: jnp.asarray(sq_exp_decomp_data[k]) 
            for k in sq_exp_decomp_data if k.startswith(prefix)}

def _get_matern_general_decomp_arrays(order: int):
    """Called at init time with static order — pure Python, not traced."""
    prefix = f"{order}__"
    return {k[len(prefix):]: jnp.asarray(matern_general_decomp_data[k]) 
            for k in matern_general_decomp_data if k.startswith(prefix)}


class Sum(tinygp.kernels.quasisep.Sum):

    kernel1: Quasisep
    kernel2: Quasisep
    use_block: bool = eqx.field(static=True, default=True)

    
class Product(tinygp.kernels.quasisep.Product):
    """A helper to represent the product of two quasiseparable kernels"""

    kernel1: Quasisep
    kernel2: Quasisep


class Scale(tinygp.kernels.quasisep.Wrapper):
    """The product of a scalar and a quasiseparable kernel"""
    scale: JAXArray | float

    def stationary_covariance(self) -> JAXArray:
        return self.scale * self.kernel.stationary_covariance()
    
    def observation_model(self, X: JAXArray) -> JAXArray:
        return self.kernel.observation_model(X)

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.transition_matrix(
            X1, X2
        )


@tinygp.helpers.dataclass
class HandleIdx(tinygp.kernels.quasisep.Wrapper):
    def coord_to_sortable(self, X):
        return X[0]

@tinygp.helpers.dataclass
class DiagonalCorrelation(tinygp.kernels.quasisep.Wrapper):
    u: JAXArray

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        X_stack = jnp.stack(X)
        return (X_stack * self.u).sum()


@tinygp.helpers.dataclass
class ScaledKernel(tinygp.kernels.quasisep.Wrapper):
    """A base class for wrapping kernels with some custom implementations"""
    amplitudes: JAXArray

    def __init__(self, kernel, amplitudes):
        self.kernel = kernel
        self.amplitudes = amplitudes

    def observation_model(self, X):
        _, idx = X
        # If kernel already a ScaledKernel object then amplitudes nest here
        return self.amplitudes[idx] * self.kernel.observation_model(X)
    
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.transition_matrix(
            X1, X2
        )

class SumDiffVec(Sum):
    """A helper to represent the sum of two quasiseparable kernels"""

    kernel1: Quasisep
    kernel2: Quasisep
    def observation_model(self, X: JAXArray) -> JAXArray:
        return jnp.concatenate(
            (
                self.kernel1.observation_model(X[0]),
                self.kernel2.observation_model(X[1]),
            )
        )

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return Block(
            self.kernel1.transition_matrix(X1[0], X2[0]),
            self.kernel2.transition_matrix(X1[1], X2[1]),
        )

class ProductDiffVec(Product):
    """A helper to represent the product of two quasiseparable kernels"""

    kernel1: Quasisep
    kernel2: Quasisep
    idx1: int = eqx.field(default_factory=lambda: 0)
    idx2: int = eqx.field(default_factory=lambda: 1)
    sort_idx: int = eqx.field(default_factory=lambda: 0)

    def __init__(self, kernel1, kernel2, idx1 = 0, idx2 = 1, sort_idx = 0):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.idx1 = idx1
        self.idx2 = idx2
        self.sort_idx = sort_idx

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """We assume that both kernels use the same coordinates"""
        return self.kernel1.coord_to_sortable(X[self.sort_idx])


    def observation_model(self, X: JAXArray) -> JAXArray:
        return _prod_helper(
            self.kernel1.observation_model(X[self.idx1]),
            self.kernel2.observation_model(X[self.idx2]),
        )

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return _prod_helper(
            self.kernel1.transition_matrix(X1[self.idx1], X2[self.idx1]),
            self.kernel2.transition_matrix(X1[self.idx2], X2[self.idx2]),
        )

@tinygp.helpers.dataclass
class Multiband(tinygp.kernels.quasisep.Wrapper):
    """A base class for wrapping kernels with some custom implementations"""
    band_amplitudes: JAXArray

    def observation_model(self, X):
        x_vec, idx = X[0], X[1]

        return self.band_amplitudes[idx % self.band_amplitudes.size] * self.kernel.observation_model((x_vec, idx // self.band_amplitudes.size))
    
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.transition_matrix(
            X1, X2
        )
    
@tinygp.helpers.dataclass
class SpecialMultiband(tinygp.kernels.quasisep.Wrapper):
    """A base class for wrapping kernels with some custom implementations"""
    band_amplitudes: JAXArray

    def observation_model(self, X):
        x_vec, idx = X

        return self.band_amplitudes[idx] * self.kernel.observation_model((x_vec, idx // self.band_amplitudes.size))
    
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.transition_matrix(
            X1, X2
        )


class LowRankProduct(Noise):
    P: JAXArray
    Q: JAXArray

    def __check_init__(self) -> None:
        if jnp.ndim(self.P) != 2:
            raise ValueError(
                "Shape must be (n_data, n_vec)"
            )
        if jnp.ndim(self.Q) != 2:
            raise ValueError(
                "Shape must be (n_data, n_vec)"
            )
        if self.P.shape != self.Q.shape:
            raise ValueError(
                "Shape of P and Q must be the same"
            )

    def diagonal(self) -> JAXArray:
        return (self.P * self.Q).sum(1)
    
    def _add(self, other):
        P_comb = jnp.concatenate([self.P, other.P], axis = 1)
        Q_comb = jnp.concatenate([self.Q, other.Q], axis = 1)
        return LowRankProduct(P_comb, Q_comb)
    
    def __add__(self, other: JAXArray) -> JAXArray:
        return self._add(other)

    def __radd__(self, other: JAXArray) -> JAXArray:
        return self._add(other)

    def __mul__(self, other):
        if is_scalar(other):
            return self.scale(other)
        else:
            raise Exception("Not implemented!")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other: JAXArray) -> JAXArray:
        H_T_other = self.Q.T @ other
        return self.P @ H_T_other

    def scale(self, c):
        return LowRankProduct(self.P * c, self.Q)
    
    def to_qsm(self):

        diag_term = DiagQSM(d=self.diagonal())
        lower_term = StrictLowerTriQSM(p=self.P, q=self.Q, a=jnp.kron(jnp.ones((self.P.shape[0], 1, 1)), jnp.eye(self.P.shape[1])))
        
        return SymmQSM(diag=diag_term, lower=lower_term)


@tinygp.helpers.dataclass
class Constant(Quasisep):
    const: JAXArray | float
    
    def design_matrix(self) -> JAXArray:
            return jnp.zeros((1, 1))
        
    def stationary_covariance(self) -> JAXArray:
            return self.const*jnp.ones((1, 1))
        
    def observation_model(self, X: JAXArray) -> JAXArray:
            del X
            return jnp.array([1.])
      
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
            return jnp.ones((1, 1))


@tinygp.helpers.dataclass
class Amplitude(Quasisep):
    alpha: JAXArray
    const: JAXArray | float = 1.

    def coord_to_sortable(self, X):
        return X[0]

    def design_matrix(self) -> JAXArray:
        return jnp.zeros((1, 1))
    
    def stationary_covariance(self) -> JAXArray:
        return self.const*jnp.ones((1, 1))
    
    def observation_model(self, X: JAXArray) -> JAXArray:
        return self.alpha[X[1]] * jnp.ones(1)
    
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return jnp.ones((1, 1))


@tinygp.helpers.dataclass
class ConstantBlocks(Quasisep):
    endpoints: JAXArray | float
    sigma: float = 1.
    
    def design_matrix(self) -> JAXArray:
        return jnp.zeros((1, 1))
        
    def stationary_covariance(self) -> JAXArray:
        return self.sigma*jnp.ones((1, 1))
        
    def observation_model(self, X: JAXArray) -> JAXArray:
        del X
        return jnp.array([1.])
      
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        same_cal = jax.lax.cond(jnp.all((X2 - self.endpoints) * (X1 - self.endpoints) >= 0.), lambda _: 1., lambda _: 0., 1.)
        return same_cal * jnp.ones((1, 1))



@tinygp.helpers.dataclass
class MaternHalfInt(Quasisep):
    double_nu: int = eqx.field(static=True)
    scale: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    _F: JAXArray
    _P_inf: JAXArray

    def __init__(self, double_nu, scale, sigma = 1):
        assert double_nu % 2 == 1, "Only odd positive integer double_nu supported for MaternNuHalf"
        assert double_nu >= 1, "Only odd positive integer double_nu supported for MaternNuHalf"
        assert double_nu <= 21, "Matern general only supported up to double_nu = 21 (due to numerical stability issues)"

        self.double_nu = double_nu

        # State space matrices calculated with a reference length scale of sqrt(double_nu) for numerical stability
        ref_scale = jnp.sqrt(double_nu)
        self.scale = scale/ref_scale
        self.sigma = sigma

        SSM_arrays = _get_matern_general_decomp_arrays(self.double_nu)
        self._F = SSM_arrays["F_ref"]
        self._P_inf = SSM_arrays["P_inf_ref"]
     
    def stationary_covariance(self):

        return self._P_inf

    def design_matrix(self):
        powers = jnp.arange((self.double_nu+1)//2)
        scale_diag = self.scale**powers

        F_ref = self._F

        return (1/self.scale) * jnp.diag(scale_diag) @ F_ref @ jnp.diag(1/scale_diag)
        
    def transition_matrix(self, X1, X2):

        tau = (X2 - X1)/self.scale
        
        return jsp.linalg.expm(self._F.T * tau)
    
    def observation_model(self, X):
        
        H = jnp.zeros((self.double_nu+1)//2)
        H = H.at[0].set(self.sigma)

        return H


@tinygp.helpers.dataclass
class SquaredExpApprox(Quasisep):
    scale: JAXArray | float
    order: int = eqx.field(static=True)
    _P_inf: JAXArray
    T: JAXArray
    T_inv: JAXArray
    total_kernel: tinygp.kernels.quasisep.Quasisep
    sigma: JAXArray | float = 1.
    _F: JAXArray
    use_block: bool = True
    
    def __init__(self, scale, order, sigma = 1, use_block = True):
        self.scale = scale
        self.order = order
        self.sigma = sigma
        self.use_block = use_block

        assert order % 2 == 0, "Only even orders supported for squared exponential approximation"
        assert order <= 14, "Modal decompositions only supported up to order 14"

        modal_decomp = _get_sq_exp_decomp_arrays(order)
        self.T = modal_decomp["T_modal"]
        self.T_inv = modal_decomp["T_modal_inv"]
        self._F = modal_decomp["F_ref"]
        self._P_inf = modal_decomp["P_inf_ref"]
        real_parts = modal_decomp["all_real_parts"]
        imag_parts = modal_decomp["imag_parts"]

        self.total_kernel = self._build_kernel(real_parts, imag_parts)

    def _build_kernel(self, real_parts, imag_parts):

        n_real = real_parts.size - imag_parts.size
        start = True
        for i in range(real_parts.size):
            kernel_term = tinygp.kernels.quasisep.Exp(-1/real_parts[i])
            if i >= n_real:
                kernel_term *= tinygp.kernels.quasisep.Cosine(2 * jnp.pi/imag_parts[i - n_real])

            if start:
                total_kernel = kernel_term
                start = False
            else:
                total_kernel = tinygp.kernels.quasisep.Sum(total_kernel, kernel_term, use_block = self.use_block)

        return total_kernel
    
    def stationary_covariance(self):

        P_inf_ref = self._P_inf

        return self.T.T @ P_inf_ref @ self.T

    def design_matrix(self):
        powers = jnp.arange(int(self.order), dtype=jnp.result_type(self.scale))
        scale_diag = self.scale**powers

        F_ref = self._F

        return (1/self.scale) * self.T @ jnp.diag(scale_diag) @ F_ref @ self.T_inv @ jnp.diag(1/scale_diag)
        
    def transition_matrix(self, X1, X2):

        A_ref = self.total_kernel.transition_matrix(X1 / self.scale, X2 / self.scale)
        
        return A_ref
    
    def observation_model(self, X):
        
        H = jnp.zeros(self.order)
        H = H.at[0].set(self.sigma)
        H = H @ self.T_inv.T

        return H


@tinygp.helpers.dataclass
class ExpSineSquaredApprox(Quasisep):
    """
    Credit: smolgp (Rubenzahl et al. 2026), original derivation from Solin & Särkkä (2014)
    """
    # Note quasisep will call this with a default order of 5 which can be inaccurate
    period: JAXArray | float
    gamma: JAXArray | float
    sigma: JAXArray | float = 1.
    order: int | None = None
    kernel: Quasisep
    periodic_scale: float
    use_block: bool

    def __init__(self, period, gamma, order = None, sigma = 1, use_block = True):

        self.period = period
        self.gamma = gamma
        self.sigma = sigma
        self.use_block = use_block

        self.periodic_scale = jnp.sqrt(2/self.gamma)
        recommended_min_order = self._order_fn(self.periodic_scale)
        
        # order gives the number of cosine terms which will be added. Quasisep rank will be J = 1 + 2 * order
        if order is None:
            order = recommended_min_order
            # if self.periodic_scale < 1/6:
            #     warnings.warn(
            #         "ExpSineSquared kernel with scale < 0.25 (gamma > 16) may require a high order approximation; "
            #         "it may be worthwhile to change units to a more compatible scale (recommended) "
            #         "or specify the 'order' parameter explicitly."
            #     )
        # elif order < self.periodic_scale:
        #     warnings.warn(
        #             f"""Chosen order of approximation {order} lower than recommended minimum of {recommended_min_order} for this
        #               value of gamma ({gamma}). Note that for large values of gamma the approximation order must be large
        #               to accurately replicate the kernel function."""
        #         )
        
        self.order = order
        self.kernel = self._build_kernel()
    
    def _build_kernel(self):

        q0 = self._Ij(0, self.periodic_scale) / jnp.exp(1/self.periodic_scale**2)
        kernel = self.sigma**2 * q0 * Constant(1.)
        
        for j in range(1, self.order+1):
            q_j2 = 2 * self._Ij(j, self.periodic_scale) / jnp.exp(1/self.periodic_scale**2)
            new_term = self.sigma**2 * q_j2 * tinygp.kernels.quasisep.Cosine(self.period/j)
            kernel = tinygp.kernels.quasisep.Sum(kernel, new_term, use_block = self.use_block)

        return kernel

    
    def _order_fn(self, scale):
        # This function roughly 

        max_order = jax.lax.cond(scale < 1/6, lambda _: 16, lambda _: (jnp.floor(4. * scale**-0.8)).astype("int"), scale)
        order = jax.lax.cond(scale > 1., lambda _: 4, lambda _: max_order, scale)
        return order-1
    
    
    def _Ij(self, j, scale, terms=50) -> JAXArray:
        """
        The modified Bessel function of the first kind, order j, at scale.
        Approximated via a truncated Taylor series expansion.
        """
        i = jnp.arange(terms)
        log_terms = -gammaln(i + 1) - gammaln(i + j + 1) - (j + 2 * i) * jnp.log(2*scale**2)
        return jnp.sum(jnp.exp(log_terms))


    def stationary_covariance(self):

        return self.kernel.stationary_covariance()

    def design_matrix(self):
        
        return self.kernel.design_matrix()
        
    def transition_matrix(self, X1, X2):
    
        return self.kernel.transition_matrix( X1, X2)
    
    def observation_model(self, X):

        return self.kernel.observation_model(X)



@tinygp.helpers.dataclass
class QuasiperiodicApprox(Quasisep):
    """
    Credit: smolgp (Rubenzahl et al. 2026), original derivation from Solin & Särkkä (2014)
    """

    period: JAXArray | float
    gamma: JAXArray | float
    decay_kernel: Quasisep
    sigma: JAXArray | float = 1.
    order: int | None = None
    kernel: Quasisep
    periodic_scale: float

    def __init__(self, period, gamma, decay_kernel, sigma = 1, order = None):

        self.period = period
        self.gamma = gamma
        self.sigma = sigma
        self.decay_kernel = decay_kernel

        self.periodic_scale = jnp.sqrt(2/self.gamma)
        
        # order gives the number of cosine terms which will be added. Quasisep rank will be J = 1 + 2 * order
        if order is None:
            order = self._order_fn(self.periodic_scale)
        
        self.order = order
        self.kernel = self._build_kernel()
    
    def _build_kernel(self):

        q0 = self._Ij(0, self.periodic_scale) / jnp.exp(1/self.periodic_scale**2)
        kernel = self.sigma**2 * q0 * Constant(1.) * self.decay_kernel
        
        for j in range(1, self.order + 1):
            q_j2 = 2 * self._Ij(j, self.periodic_scale) / jnp.exp(1/self.periodic_scale**2)
            new_term = self.sigma**2 * q_j2 * tinygp.kernels.quasisep.Cosine(self.period/j) * self.decay_kernel
            kernel = tinygp.kernels.quasisep.Sum(kernel, new_term, use_block = True)

        return kernel

    
    def _order_fn(self, scale):

        max_order = jax.lax.cond(scale < 1/6, lambda _: 16, lambda _: (jnp.floor(4. * scale**-0.8)).astype("int"), scale)
        order = jax.lax.cond(scale > 1., lambda _: 4, lambda _: max_order, scale)
        return order - 1
    
    
    def _Ij(self, j, scale, terms=50) -> JAXArray:
        """
        The modified Bessel function of the first kind, order j, at scale.
        Approximated via a truncated Taylor series expansion.
        """
        i = jnp.arange(terms)
        log_terms = -gammaln(i + 1) - gammaln(i + j + 1) - (j + 2 * i) * jnp.log(2*scale**2)
        return jnp.sum(jnp.exp(log_terms))


    def stationary_covariance(self):

        return self.kernel.stationary_covariance()

    def design_matrix(self):
        
        return self.kernel.design_matrix()
        
    def transition_matrix(self, X1, X2):
    
        return self.kernel.transition_matrix(X1, X2)
    
    def observation_model(self, X):

        return self.kernel.observation_model(X)



@tinygp.helpers.dataclass
class GeneralCARMA(Quasisep):
    roots: JAXArray | float
    scale: JAXArray | float
    _P_inf: JAXArray
    T: JAXArray
    T_inv: JAXArray
    total_kernel: tinygp.kernels.quasisep.Quasisep
    sigma: JAXArray | float = 1.
    _F: JAXArray
    order: int
    
    def __init__(self, roots, scale, sigma = 1):
        self.roots = roots
        self.scale = scale
        self.sigma = sigma
        self.order = roots.size

        stoc_diff_coeffs = stoc_diff_coeffs_from_roots(roots)

        print("h_{m-1}, h_{m-2}, .., h_0:", stoc_diff_coeffs[::-1])
        self.T, self.T_inv, real_parts, imag_parts, self._F, self._P_inf = modal_decomp_SDE(stoc_diff_coeffs)
        
        self.total_kernel = self._build_kernel(real_parts, imag_parts)
        
    def _build_kernel(self, real_parts, imag_parts):

        n_real = real_parts.size - imag_parts.size
        start = True
        for i in range(real_parts.size):
            kernel_term = tinygp.kernels.quasisep.Exp(-1/real_parts[i])
            if i >= n_real:
                kernel_term *= tinygp.kernels.quasisep.Cosine(2 * jnp.pi/imag_parts[i - n_real])

            if start:
                total_kernel = kernel_term
                start = False
            else:
                total_kernel += kernel_term

        return total_kernel


    def to_symm_qsm(self, X):

        a = jax.vmap(self.total_kernel.transition_matrix)(
            jax.tree_util.tree_map(lambda y: jnp.append(y[0], y[:-1]), X/self.scale), X/self.scale
        )
        h = jax.vmap(self.observation_model)(X)

        # powers = jnp.arange(int(self.order), dtype=jnp.result_type(self.scale))
        # scale_diag = self.scale**powers

        q = h @ self.T_inv.T
        p = (h @ self._P_inf) @ self.T
        d = jnp.sum(p * q, axis=1)
        
        p = jax.vmap(lambda x, y: x @ y)(p, a)

        return SymmQSM(diag=DiagQSM(d=d), lower=StrictLowerTriQSM(p=p, q=q, a=a))
    
    
    def stationary_covariance(self):

        powers = jnp.arange(int(self.order), dtype=jnp.result_type(self.scale))
        scale_diag = self.scale**powers

        P_inf_ref = self._P_inf

        return jnp.diag(1/scale_diag) @ P_inf_ref @ jnp.diag(1/scale_diag)


    def design_matrix(self):
        powers = jnp.arange(int(self.order), dtype=jnp.result_type(self.scale))
        scale_diag = self.scale**powers

        F_ref = self._F

        return (1/self.scale) * jnp.diag(scale_diag) @ self.T @ F_ref @ self.T_inv @ jnp.diag(1/scale_diag)
        
    def transition_matrix(self, X1, X2):

        # F(scale)^T = D @ (F_ref^T / scale) @ D^{-1}, D=diag(scale^k)

        A_ref = self.total_kernel.transition_matrix(X1 / self.scale, X2 / self.scale)
        powers = jnp.arange(int(self.order), dtype=jnp.result_type(self.scale))
        d = self.scale**powers
    
        return jnp.diag(d) @ self.T @ A_ref @ self.T_inv @ jnp.diag(1/d) 
    
    def observation_model(self, X):
        
        H = jnp.zeros(self.order)
        H = H.at[0].set(self.sigma)

        return H

