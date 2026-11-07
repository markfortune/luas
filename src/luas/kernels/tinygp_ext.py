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
    r"""Sum of two ``tinygp`` quasiseparable kernels. ``luas``'s version of ``tinygp.kernels.quasisep.Sum``, which
    declares ``use_block`` as a static field: whether the transition matrices of the two kernels are combined as
    a ``tinygp`` ``Block`` matrix rather than a dense block diagonal matrix.

    """

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
        r"""Returns the wrapped kernel's stationary covariance multiplied by ``scale``, which scales the kernel by ``scale``."""
        return self.scale * self.kernel.stationary_covariance()
    
    def observation_model(self, X: JAXArray) -> JAXArray:
        r"""Returns the wrapped kernel's observation model unchanged."""
        return self.kernel.observation_model(X)

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r"""Returns the wrapped kernel's transition matrix unchanged."""
        return self.kernel.transition_matrix(
            X1, X2
        )


@tinygp.helpers.dataclass
class HandleIdx(tinygp.kernels.quasisep.Wrapper):
    r"""Wraps a ``tinygp`` quasiseparable kernel so that it takes the ``(x, index)`` coordinates used throughout
    ``luas``. The kernel itself only sees ``x``, and the points are sorted by ``x``.

    """
    def coord_to_sortable(self, X):
        r"""Returns ``x`` from the ``(x, index)`` coordinates, used to sort the points."""
        return X[0]

@tinygp.helpers.dataclass
class DiagonalCorrelation(tinygp.kernels.quasisep.Wrapper):
    r"""Wraps a ``tinygp`` quasiseparable kernel taking several coordinates, sorting the points by the projection
    of their coordinates onto the direction ``u``, e.g. for correlations along a diagonal of a 2D grid.

    Args:
        kernel: The ``tinygp`` kernel to wrap.
        u (JAXArray): The direction to project the coordinates onto.

    """
    u: JAXArray

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        r"""Returns the coordinates projected onto ``u``, :math:`\sum_i u_i X_i`, used to sort the points."""
        X_stack = jnp.stack(X)
        return (X_stack * self.u).sum()


@tinygp.helpers.dataclass
class ScaledKernel(tinygp.kernels.quasisep.Wrapper):
    """A base class for wrapping kernels with some custom implementations"""
    amplitudes: JAXArray

    def __init__(self, kernel, amplitudes):
        r"""Args:
            kernel: The ``tinygp`` kernel to wrap, taking ``(x, index)`` coordinates.
            amplitudes (JAXArray): The amplitude :math:`a_i` at each index, giving
                :math:`k'(x_i, x_j) = a_i a_j k(x_i, x_j)`.

        """
        self.kernel = kernel
        self.amplitudes = amplitudes

    def observation_model(self, X):
        r"""Returns the wrapped kernel's observation model multiplied by the amplitude at the index in ``X``."""
        _, idx = X
        # If kernel already a ScaledKernel object then amplitudes nest here
        return self.amplitudes[idx] * self.kernel.observation_model(X)
    
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r"""Returns the wrapped kernel's transition matrix unchanged."""
        return self.kernel.transition_matrix(
            X1, X2
        )

class SumDiffVec(Sum):
    """A helper to represent the sum of two quasiseparable kernels"""

    kernel1: Quasisep
    kernel2: Quasisep
    def observation_model(self, X: JAXArray) -> JAXArray:
        r"""Concatenates the observation models of ``kernel1`` at ``X[0]`` and ``kernel2`` at ``X[1]``."""
        return jnp.concatenate(
            (
                self.kernel1.observation_model(X[0]),
                self.kernel2.observation_model(X[1]),
            )
        )

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r"""Returns the transition matrices of the two kernels, at ``X[0]`` and ``X[1]`` respectively, as a ``tinygp`` ``Block`` matrix."""
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
        r"""Args:
            kernel1: The first ``tinygp`` quasiseparable kernel.
            kernel2: The second ``tinygp`` quasiseparable kernel.
            idx1 (int, optional): Which coordinate of the input ``kernel1`` uses. Defaults to 0.
            idx2 (int, optional): Which coordinate of the input ``kernel2`` uses. Defaults to 1.
            sort_idx (int, optional): Which coordinate is used to sort the points. Defaults to 0.

        """
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.idx1 = idx1
        self.idx2 = idx2
        self.sort_idx = sort_idx

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """We assume that both kernels use the same coordinates"""
        return self.kernel1.coord_to_sortable(X[self.sort_idx])


    def observation_model(self, X: JAXArray) -> JAXArray:
        r"""Returns the product of the two kernels' observation models, each at its own coordinate."""
        return _prod_helper(
            self.kernel1.observation_model(X[self.idx1]),
            self.kernel2.observation_model(X[self.idx2]),
        )

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r"""Returns the product of the two kernels' transition matrices, each at its own coordinate."""
        return _prod_helper(
            self.kernel1.transition_matrix(X1[self.idx1], X2[self.idx1]),
            self.kernel2.transition_matrix(X1[self.idx2], X2[self.idx2]),
        )

@tinygp.helpers.dataclass
class Multiband(tinygp.kernels.quasisep.Wrapper):
    """A base class for wrapping kernels with some custom implementations"""
    band_amplitudes: JAXArray

    def observation_model(self, X):
        r"""Returns the wrapped kernel's observation model multiplied by a band amplitude. The bands are interleaved:
        index :math:`i` belongs to band ``i % n_bands`` at position ``i // n_bands`` along the input. Used by
        ``MixingMatQuasisep`` for the method of Gordon et al. (2020).

        """
        x_vec, idx = X[0], X[1]

        return self.band_amplitudes[idx % self.band_amplitudes.size] * self.kernel.observation_model((x_vec, idx // self.band_amplitudes.size))
    
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r"""Returns the wrapped kernel's transition matrix unchanged."""
        return self.kernel.transition_matrix(
            X1, X2
        )
    
@tinygp.helpers.dataclass
class SpecialMultiband(tinygp.kernels.quasisep.Wrapper):
    """A base class for wrapping kernels with some custom implementations"""
    band_amplitudes: JAXArray

    def observation_model(self, X):
        r"""Returns the wrapped kernel's observation model multiplied by ``band_amplitudes`` at the index in ``X``, with
        the wrapped kernel evaluated at position ``index // band_amplitudes.size``.

        """
        x_vec, idx = X

        return self.band_amplitudes[idx] * self.kernel.observation_model((x_vec, idx // self.band_amplitudes.size))
    
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r"""Returns the wrapped kernel's transition matrix unchanged."""
        return self.kernel.transition_matrix(
            X1, X2
        )


class LowRankProduct(Noise):
    r"""A low-rank matrix :math:`\mathbf{P}\mathbf{Q}^T` as a ``tinygp`` noise model.

    Args:
        P (JAXArray): Matrix of shape ``(n_data, n_vec)``.
        Q (JAXArray): Matrix of shape ``(n_data, n_vec)``.

    """
    P: JAXArray
    Q: JAXArray

    def __check_init__(self) -> None:
        r"""Checks the shapes of ``P`` and ``Q``.

        Raises:
            ValueError: If ``P`` and ``Q`` are not 2D arrays of the same shape.

        """
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
        r"""Returns the diagonal of :math:`\mathbf{P}\mathbf{Q}^T`."""
        return (self.P * self.Q).sum(1)
    
    def _add(self, other):
        r"""Returns the sum with another ``LowRankProduct``, by joining the columns of their ``P`` and ``Q`` matrices."""
        P_comb = jnp.concatenate([self.P, other.P], axis = 1)
        Q_comb = jnp.concatenate([self.Q, other.Q], axis = 1)
        return LowRankProduct(P_comb, Q_comb)
    
    def __add__(self, other: JAXArray) -> JAXArray:
        r"""Returns the sum with another ``LowRankProduct``."""
        return self._add(other)

    def __radd__(self, other: JAXArray) -> JAXArray:
        r"""Returns the sum with another ``LowRankProduct``."""
        return self._add(other)

    def __mul__(self, other):
        r"""Multiplies by a scalar.

        Raises:
            Exception: For anything other than a scalar.

        """
        if is_scalar(other):
            return self.scale(other)
        else:
            raise Exception("Not implemented!")

    def __rmul__(self, other):
        r"""Multiplies by a scalar, see :meth:`__mul__`."""
        return self.__mul__(other)

    def __matmul__(self, other: JAXArray) -> JAXArray:
        r"""Multiplies ``other`` by :math:`\mathbf{P}\mathbf{Q}^T` without building it."""
        H_T_other = self.Q.T @ other
        return self.P @ H_T_other

    def scale(self, c):
        r"""Returns a ``LowRankProduct`` for :math:`c\,\mathbf{P}\mathbf{Q}^T`."""
        return LowRankProduct(self.P * c, self.Q)
    
    def to_qsm(self):
        r"""Returns the matrix as a ``tinygp`` ``SymmQSM``, with diagonal :math:`\mathrm{diag}(\mathbf{P}\mathbf{Q}^T)`
        and strictly lower triangle from :math:`\mathbf{P}\mathbf{Q}^T`. Assumes :math:`\mathbf{P}\mathbf{Q}^T` is symmetric.

        """

        diag_term = DiagQSM(d=self.diagonal())
        lower_term = StrictLowerTriQSM(p=self.P, q=self.Q, a=jnp.kron(jnp.ones((self.P.shape[0], 1, 1)), jnp.eye(self.P.shape[1])))
        
        return SymmQSM(diag=diag_term, lower=lower_term)


@tinygp.helpers.dataclass
class Constant(Quasisep):
    r"""A constant kernel, :math:`k(x, x') = c`, as a ``tinygp`` quasiseparable kernel.

    Args:
        const (JAXArray | float): The constant :math:`c`.

    """
    const: JAXArray | float
    
    def design_matrix(self) -> JAXArray:
            r"""Returns zero, the state doesn't evolve."""
            return jnp.zeros((1, 1))
        
    def stationary_covariance(self) -> JAXArray:
            r"""Returns the constant."""
            return self.const*jnp.ones((1, 1))
        
    def observation_model(self, X: JAXArray) -> JAXArray:
            r"""Returns one."""
            del X
            return jnp.array([1.])
      
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
            r"""Returns one, the state doesn't change."""
            return jnp.ones((1, 1))


@tinygp.helpers.dataclass
class Amplitude(Quasisep):
    r"""A rank-one kernel, :math:`k(x_i, x_j) = c\,\alpha_i\alpha_j`, as a ``tinygp`` quasiseparable kernel taking
    ``(x, index)`` coordinates.

    Args:
        alpha (JAXArray): The vector :math:`\vec{\alpha}`, with one value per index.
        const (JAXArray | float, optional): The constant :math:`c`. Defaults to 1.

    """
    alpha: JAXArray
    const: JAXArray | float = 1.

    def coord_to_sortable(self, X):
        r"""Returns ``x`` from the ``(x, index)`` coordinates, used to sort the points."""
        return X[0]

    def design_matrix(self) -> JAXArray:
        r"""Returns zero, the state doesn't evolve."""
        return jnp.zeros((1, 1))
    
    def stationary_covariance(self) -> JAXArray:
        r"""Returns the constant :math:`c`."""
        return self.const*jnp.ones((1, 1))
    
    def observation_model(self, X: JAXArray) -> JAXArray:
        r"""Returns :math:`\alpha_i` at the index :math:`i` in ``X``."""
        return self.alpha[X[1]] * jnp.ones(1)
    
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r"""Returns one, the state doesn't change."""
        return jnp.ones((1, 1))


@tinygp.helpers.dataclass
class ConstantBlocks(Quasisep):
    r"""A piecewise constant kernel: :math:`k(x, x') = \sigma` if :math:`x` and :math:`x'` are in the same block, and
    zero otherwise. Two points are in the same block if they lie on the same side of every endpoint.

    Args:
        endpoints (JAXArray | float): The boundaries between the blocks.
        sigma (float, optional): The value of the kernel within a block. Defaults to 1.

    """
    endpoints: JAXArray | float
    sigma: float = 1.
    
    def design_matrix(self) -> JAXArray:
        r"""Returns zero, the state doesn't evolve within a block."""
        return jnp.zeros((1, 1))
        
    def stationary_covariance(self) -> JAXArray:
        r"""Returns ``sigma``."""
        return self.sigma*jnp.ones((1, 1))
        
    def observation_model(self, X: JAXArray) -> JAXArray:
        r"""Returns one."""
        del X
        return jnp.array([1.])
      
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r"""Returns one if ``X1`` and ``X2`` are on the same side of every endpoint, otherwise zero."""
        same_cal = jax.lax.cond(jnp.all((X2 - self.endpoints) * (X1 - self.endpoints) >= 0.), lambda _: 1., lambda _: 0., 1.)
        return same_cal * jnp.ones((1, 1))



@tinygp.helpers.dataclass
class MaternHalfInt(Quasisep):
    r"""The half-integer Matérn kernel, :math:`\nu = ` ``double_nu / 2``, as a ``tinygp`` quasiseparable (state space)
    kernel.

    The state space matrices are precomputed (``data/matern_SSMs_stable.npz``) at a reference length scale of
    :math:`\sqrt{\texttt{double\_nu}}` for numerical stability, and rescaled to the length scale given.

    Args:
        double_nu (int): Twice the smoothness :math:`\nu`. Must be an odd integer between 1 and 21.
        scale (JAXArray | float): The length scale :math:`l`.
        sigma (JAXArray | float, optional): The amplitude. Defaults to 1.

    """
    double_nu: int = eqx.field(static=True)
    scale: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    _F: JAXArray
    _P_inf: JAXArray

    def __init__(self, double_nu, scale, sigma = 1):
        r"""Creates the object, see the class docstring for the arguments."""
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
        r"""Returns the stationary covariance :math:`\mathbf{P}_\infty` of the state space model."""

        return self._P_inf

    def design_matrix(self):
        r"""Returns the feedback (design) matrix :math:`\mathbf{F}` of the state space model. Rescaled from the reference length scale."""
        powers = jnp.arange((self.double_nu+1)//2)
        scale_diag = self.scale**powers

        F_ref = self._F

        return (1/self.scale) * jnp.diag(scale_diag) @ F_ref @ jnp.diag(1/scale_diag)
        
    def transition_matrix(self, X1, X2):
        r"""Returns the transition matrix :math:`\mathbf{A}` which evolves the state from ``X1`` to ``X2``. Calculated as the matrix exponential :math:`\exp(\mathbf{F}^T\tau)`."""

        tau = (X2 - X1)/self.scale
        
        return jsp.linalg.expm(self._F.T * tau)
    
    def observation_model(self, X):
        r"""Returns :math:`\vec{h} = (\sigma, 0, \dots, 0)`."""
        
        H = jnp.zeros((self.double_nu+1)//2)
        H = H.at[0].set(self.sigma)

        return H


@tinygp.helpers.dataclass
class SquaredExpApprox(Quasisep):
    r"""A quasiseparable (state space) approximation to the squared exponential kernel.

    The squared exponential power spectrum is approximated by a rational function from a Taylor expansion,
    which gives a stochastic differential equation of order ``order``. Its modal decomposition is a sum of
    exponential and exponential-times-cosine kernels. The modal decompositions are precomputed
    (``data/sq_exp_modal_decomps.npz``).

    Args:
        scale (JAXArray | float): The length scale :math:`l`.
        order (int): The order of the approximation. Must be even and at most 14.
        sigma (JAXArray | float, optional): The amplitude. Defaults to 1.
        use_block (bool, optional): Whether to combine the terms using ``tinygp``'s block representation.
            Defaults to ``True``.

    """
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
        r"""Creates the object, see the class docstring for the arguments."""
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
        r"""Builds the sum of kernels for each mode: an exponential kernel for each real root, multiplied by a cosine
        kernel for each complex pair.

        Args:
            real_parts (JAXArray): Real parts of the roots, the real roots first.
            imag_parts (JAXArray): Imaginary parts of the complex roots.

        Returns:
            The summed ``tinygp`` kernel.

        """

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
        r"""Returns the stationary covariance :math:`\mathbf{P}_\infty` of the state space model. In the modal basis."""

        P_inf_ref = self._P_inf

        return self.T.T @ P_inf_ref @ self.T

    def design_matrix(self):
        r"""Returns the feedback (design) matrix :math:`\mathbf{F}` of the state space model. In the modal basis and rescaled to the length scale."""
        powers = jnp.arange(int(self.order), dtype=jnp.result_type(self.scale))
        scale_diag = self.scale**powers

        F_ref = self._F

        return (1/self.scale) * self.T @ jnp.diag(scale_diag) @ F_ref @ self.T_inv @ jnp.diag(1/scale_diag)
        
    def transition_matrix(self, X1, X2):
        r"""Returns the transition matrix :math:`\mathbf{A}` which evolves the state from ``X1`` to ``X2``. Calculated from the summed modal kernels at :math:`x / l`."""

        A_ref = self.total_kernel.transition_matrix(X1 / self.scale, X2 / self.scale)
        
        return A_ref
    
    def observation_model(self, X):
        r"""Returns :math:`\vec{h} = (\sigma, 0, \dots, 0)` transformed to the modal basis."""
        
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
        r"""Args:
            period (JAXArray | float): The period :math:`P`.
            gamma (JAXArray | float): The parameter :math:`\gamma` of the exponential sine squared kernel.
            order (int, optional): The number of cosine terms. Defaults to a value chosen from ``gamma``
                (see :meth:`_order_fn`).
            sigma (JAXArray | float, optional): The amplitude. Defaults to 1.
            use_block (bool, optional): Whether to combine the terms using ``tinygp``'s block representation.
                Defaults to ``True``.

        """

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
        r"""Builds the approximation as a constant plus ``order`` cosine kernels with periods :math:`P/j`. With
        :math:`\ell = \sqrt{2/\gamma}`, the weights are :math:`q_0^2 = I_0(\ell^{-2})/e^{\ell^{-2}}` and
        :math:`q_j^2 = 2I_j(\ell^{-2})/e^{\ell^{-2}}`, where :math:`I_j` is the modified Bessel function of the first kind.

        Returns:
            The summed ``tinygp`` kernel.

        """

        q0 = self._Ij(0, self.periodic_scale) / jnp.exp(1/self.periodic_scale**2)
        kernel = self.sigma**2 * q0 * Constant(1.)
        
        for j in range(1, self.order+1):
            q_j2 = 2 * self._Ij(j, self.periodic_scale) / jnp.exp(1/self.periodic_scale**2)
            new_term = self.sigma**2 * q_j2 * tinygp.kernels.quasisep.Cosine(self.period/j)
            kernel = tinygp.kernels.quasisep.Sum(kernel, new_term, use_block = self.use_block)

        return kernel

    
    def _order_fn(self, scale):
        r"""Chooses the number of cosine terms from :math:`\ell = \sqrt{2/\gamma}`: 3 if :math:`\ell > 1`,
        otherwise :math:`\lfloor 4\ell^{-0.8} \rfloor - 1`, up to 15 for :math:`\ell < 1/6`.

        """
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
        r"""Returns the stationary covariance :math:`\mathbf{P}_\infty` of the state space model. From the summed cosine kernels."""

        return self.kernel.stationary_covariance()

    def design_matrix(self):
        r"""Returns the feedback (design) matrix :math:`\mathbf{F}` of the state space model. From the summed cosine kernels."""
        
        return self.kernel.design_matrix()
        
    def transition_matrix(self, X1, X2):
        r"""Returns the transition matrix :math:`\mathbf{A}` which evolves the state from ``X1`` to ``X2``. From the summed cosine kernels."""
    
        return self.kernel.transition_matrix( X1, X2)
    
    def observation_model(self, X):
        r"""Returns the observation model :math:`\vec{h}`, which maps the state at the input ``X`` to the output. From the summed cosine kernels."""

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
        r"""Args:
            period (JAXArray | float): The period :math:`P`.
            gamma (JAXArray | float): The parameter :math:`\gamma` of the exponential sine squared kernel.
            decay_kernel: The ``tinygp`` quasiseparable kernel multiplying every term.
            sigma (JAXArray | float, optional): The amplitude. Defaults to 1.
            order (int, optional): The number of cosine terms. Defaults to a value chosen from ``gamma``
                (see :meth:`_order_fn`).

        """

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
        r"""Builds the approximation as in :meth:`ExpSineSquaredApprox._build_kernel`, with every term multiplied by
        ``decay_kernel``.

        Returns:
            The summed ``tinygp`` kernel.

        """

        q0 = self._Ij(0, self.periodic_scale) / jnp.exp(1/self.periodic_scale**2)
        kernel = self.sigma**2 * q0 * Constant(1.) * self.decay_kernel
        
        for j in range(1, self.order + 1):
            q_j2 = 2 * self._Ij(j, self.periodic_scale) / jnp.exp(1/self.periodic_scale**2)
            new_term = self.sigma**2 * q_j2 * tinygp.kernels.quasisep.Cosine(self.period/j) * self.decay_kernel
            kernel = tinygp.kernels.quasisep.Sum(kernel, new_term, use_block = True)

        return kernel

    
    def _order_fn(self, scale):
        r"""Chooses the number of cosine terms from :math:`\ell = \sqrt{2/\gamma}`: 3 if :math:`\ell > 1`,
        otherwise :math:`\lfloor 4\ell^{-0.8} \rfloor - 1`, up to 15 for :math:`\ell < 1/6`.

        """

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
        r"""Returns the stationary covariance :math:`\mathbf{P}_\infty` of the state space model. From the summed kernels."""

        return self.kernel.stationary_covariance()

    def design_matrix(self):
        r"""Returns the feedback (design) matrix :math:`\mathbf{F}` of the state space model. From the summed kernels."""
        
        return self.kernel.design_matrix()
        
    def transition_matrix(self, X1, X2):
        r"""Returns the transition matrix :math:`\mathbf{A}` which evolves the state from ``X1`` to ``X2``. From the summed kernels."""
    
        return self.kernel.transition_matrix(X1, X2)
    
    def observation_model(self, X):
        r"""Returns the observation model :math:`\vec{h}`, which maps the state at the input ``X`` to the output. From the summed kernels."""

        return self.kernel.observation_model(X)



@tinygp.helpers.dataclass
class GeneralCARMA(Quasisep):
    r"""A general state space kernel defined by the roots of the characteristic polynomial of its stochastic
    differential equation (a continuous autoregressive model). The equation is put into modal form, a sum of
    exponential and exponential-times-cosine kernels. Prints the polynomial coefficients when created.

    Args:
        roots (JAXArray): The roots of the characteristic polynomial, which should have negative real parts.
        scale (JAXArray | float): The length scale the inputs are divided by.
        sigma (JAXArray | float, optional): The amplitude. Defaults to 1.

    """
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
        r"""Creates the object, see the class docstring for the arguments."""
        self.roots = roots
        self.scale = scale
        self.sigma = sigma
        self.order = roots.size

        stoc_diff_coeffs = stoc_diff_coeffs_from_roots(roots)

        print("h_{m-1}, h_{m-2}, .., h_0:", stoc_diff_coeffs[::-1])
        self.T, self.T_inv, real_parts, imag_parts, self._F, self._P_inf = modal_decomp_SDE(stoc_diff_coeffs)
        
        self.total_kernel = self._build_kernel(real_parts, imag_parts)
        
    def _build_kernel(self, real_parts, imag_parts):
        r"""Builds the sum of kernels for each mode: an exponential kernel for each real root, multiplied by a cosine
        kernel for each complex pair.

        Returns:
            The summed ``tinygp`` kernel.

        """

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
        r"""Builds the covariance matrix at the inputs ``X`` directly as a ``tinygp`` ``SymmQSM``.

        Args:
            X (JAXArray): Sorted input locations.

        Returns:
            SymmQSM: The symmetric quasiseparable covariance matrix.

        """

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
        r"""Returns the stationary covariance :math:`\mathbf{P}_\infty` of the state space model. Rescaled to the length scale."""

        powers = jnp.arange(int(self.order), dtype=jnp.result_type(self.scale))
        scale_diag = self.scale**powers

        P_inf_ref = self._P_inf

        return jnp.diag(1/scale_diag) @ P_inf_ref @ jnp.diag(1/scale_diag)


    def design_matrix(self):
        r"""Returns the feedback (design) matrix :math:`\mathbf{F}` of the state space model. Rescaled to the length scale."""
        powers = jnp.arange(int(self.order), dtype=jnp.result_type(self.scale))
        scale_diag = self.scale**powers

        F_ref = self._F

        return (1/self.scale) * jnp.diag(scale_diag) @ self.T @ F_ref @ self.T_inv @ jnp.diag(1/scale_diag)
        
    def transition_matrix(self, X1, X2):
        r"""Returns the transition matrix :math:`\mathbf{A}` which evolves the state from ``X1`` to ``X2``. Calculated from the modal kernels at :math:`x / l`, transformed back from the modal basis."""

        # F(scale)^T = D @ (F_ref^T / scale) @ D^{-1}, D=diag(scale^k)

        A_ref = self.total_kernel.transition_matrix(X1 / self.scale, X2 / self.scale)
        powers = jnp.arange(int(self.order), dtype=jnp.result_type(self.scale))
        d = self.scale**powers
    
        return jnp.diag(d) @ self.T @ A_ref @ self.T_inv @ jnp.diag(1/d) 
    
    def observation_model(self, X):
        r"""Returns :math:`\vec{h} = (\sigma, 0, \dots, 0)`."""
        
        H = jnp.zeros(self.order)
        H = H.at[0].set(self.sigma)

        return H

