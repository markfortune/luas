import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import jax
import jax.numpy as jnp
from typing import Any, Optional, Callable, Union, Dict, Tuple
from tqdm import tqdm

from luas.kernel_selector import kernel_selector
from luas.kernels.covtype import CovType
from luas.kernels.Conditional import Conditional
from luas.luas_types import Scalar, PyTree, JAXArray, Kernel
from luas.kronecker_fns import calc_data_shape
from luas.convenience_fns import (
    array_to_pytree_2D,
    pytree_to_array_2D,
    order_list,
    large_hessian_calc,
    transf_from_unbounded_params,
    transf_to_unbounded_params,
    varying_params_wrapper,
    get_corr_mat,
    check_params_in_bounds,
    bin_data_2D,
    interp_params,
    stack_params,
    stack_obs_and_pred_points,
)

__all__ = ["GP"]


class GP(object):
    r"""Gaussian process interface for 1D data sets, and for 2D (or higher) data sets on a grid.

    The covariance matrix of a 2D (or higher) data set is built from kernels along each dimension,
    e.g. one along wavelength and one along time. When the object is created, ``luas`` looks at the
    structure of these kernels and chooses an optimised method (one of the
    :doc:`Kronecker kernel classes <kronecker_kernels>`) for calculating the log likelihood and
    its gradients. ``verbose = True`` prints which one was chosen, and ``use_kernel`` can be
    used to choose a different one.

    Example:
        A 2D GP with a squared exponential kernel along each dimension plus white noise::

            def kf(p, X):
                Kl = luas.kernels.SquaredExp(p["l_l"], sigma = p["h"])
                Kt = luas.kernels.SquaredExp(p["l_t"])
                Sl = luas.kernels.Noise(p["sigma_l"])
                St = luas.kernels.Noise(1.)
                return (Sl, St), (Kl, Kt)

            gp = luas.GP(p, (x_l, x_t), kf = kf, mf = transit_2D)
            logL = gp.logL(p, Y)

    Args:
        p (PyTree): Parameters used to evaluate the kernel function once when the object is created,
            to check its structure and choose which GP optimisation to use. Must contain every parameter
            the kernel function needs.
        X (JAXArray | tuple): The input locations (regression variables). For 1D data this is a JAXArray
            of shape ``(N,)``, or ``(d, N)`` for ``d`` different input variables. For 2D or higher this is a
            tuple with one such JAXArray per dimension, in the same order as the dimensions of the
            observed data array.
        kf (Callable | CovType, optional): The kernel function, called as ``kf(p, X, *kf_args, **kf_kwargs)``.
            For 1D data it should return a single kernel from :doc:`luas.kernels <kernels>` (or a sum or
            product of kernels). For 2D or higher it should return either a tuple with one kernel per
            dimension, representing the Kronecker product of those kernels, or a tuple of such tuples
            which are summed. For example ``(Sl, St), (Kl, Kt)`` represents
            :math:`\mathbf{S}_l \otimes \mathbf{S}_t + \mathbf{K}_l \otimes \mathbf{K}_t`, where the
            first term is usually the noise term. It can also return a Kronecker kernel object directly,
            or a fixed covariance object can be given instead of a function. If not given, the
            covariance matrix is the identity matrix (unweighted least squares).
        mf (Callable, optional): The deterministic mean function, called as ``mf(p, X, *mf_args, **mf_kwargs)``.
            Must return a JAXArray with the same shape as the observed data. Defaults to zero.
        logPrior (Callable, optional): The log prior, called as
            ``logPrior(p, X, *logPrior_args, **logPrior_kwargs)`` and returning a scalar. Defaults to zero.
        jit (bool, optional): Whether to ``jax.jit`` compile :meth:`logL`, :meth:`logP`,
            :meth:`logL_stored`, :meth:`logP_stored`, :meth:`logL_hessianable` and
            :meth:`logP_hessianable`. Set to ``False`` to use a mean function which isn't written in
            ``JAX``, or to make debugging easier. Defaults to ``True``.
        use_kernel (optional): A Kronecker kernel class to use instead of the one ``luas`` chooses
            automatically, e.g. ``luas.GeneralKernel`` builds the full covariance matrix and uses a
            Cholesky decomposition with no optimisations. When more than one optimisation is valid
            (e.g. ``luas.LuasKernel`` and ``luas.LuasLasrachKernel``) it can be worth testing which is
            fastest. Any keyword arguments the class needs can be given as extra keyword arguments.
        kernel_select_kwargs (dict, optional): Keyword arguments passed to the automatic choice of GP
            optimisation for 2D data, e.g. ``max_gen_cholesky_blocks``.
        verbose (bool, optional): Whether to print which GP optimisation is being used. Defaults to ``True``.
        kf_args (tuple, optional): Extra positional arguments passed to the kernel function.
        kf_kwargs (dict, optional): Extra keyword arguments passed to the kernel function.
        mf_args (tuple, optional): Extra positional arguments passed to the mean function.
        mf_kwargs (dict, optional): Extra keyword arguments passed to the mean function.
        logPrior_args (tuple, optional): Extra positional arguments passed to the log prior function.
        logPrior_kwargs (dict, optional): Extra keyword arguments passed to the log prior function.
        **kernel_kwargs: Any other keyword arguments are passed to the Kronecker kernel class when it is
            built, e.g. ``fast_dim`` for ``luas.LuasLasrachKernel``. These override any values chosen
            automatically.

    Attributes:
        x (JAXArray | tuple): The input locations ``X``.
        dim (int): The number of dimensions of the data set.
        data_shape (tuple): The shape of the data set, e.g. ``(N_l, N_t)``.
        opt_name (str): The name of the GP optimisation (Kronecker kernel class) being used.

    """
    
    def __init__(
        self,
        p: PyTree,
        X: Tuple,
        kf: Optional[Callable] = None,
        mf: Optional[Callable] = None,
        logPrior: Optional[Callable] = None,
        jit: Optional[bool] = True,
        use_kernel: Kernel = None,
        kernel_select_kwargs: Optional[PyTree] = {},
        verbose = True,
        kf_args = (), kf_kwargs = {},
        mf_args = (), mf_kwargs = {},
        logPrior_args = (), logPrior_kwargs = {},
        **kernel_kwargs,
    ):
        self.x = X

        if isinstance(X, tuple):
            self.dim = len(X)
        else:
            self.dim = 1

        for d in range(self.dim):
            if (self.dim == 1 and X.ndim != 1) or (self.dim > 1 and X[d].ndim != 1):
                raise Exception(f"Currently input locations for dimension {d} must be a 1D array, got shape {X[d].shape}."\
                                " In future this will be extended to allow multiple input variables as shape (d, N).")
            
        self.data_shape = calc_data_shape(self.x)
        
        # Specify kernel function and include any args or kwargs needed
        if isinstance(kf, CovType) or kf is None:
            self.kf = kf
        elif isinstance(kf_args, tuple) and isinstance(kf_kwargs, dict):
            self.kf = lambda p, X, **kwargs: kf(p, X, *kf_args, **kf_kwargs, **kwargs)
        else:
            raise Exception("""kf_args must be a Tuple and kf_kwargs must be a dict.
                                For a single argument use kf_args = (arg1,)""")

        # Specify mean function and include any args or kwargs needed
        if mf is None:
            # Mean function returns zeros by default
            # Returns a zero array which varies in shape depending on the inputs
            self.mf = lambda p, X: jnp.zeros(calc_data_shape(X))
        else:
            # Ensure mean function is of the form mf(p, X) and for the
            # observed inputs X should return an array of the same shape
            # as the observed data Y.
            if isinstance(mf_args, tuple):
                self.mf = lambda p, X: mf(p, X, *mf_args, **mf_kwargs)
            else:
                raise Exception("mf_args must be a Tuple, for one argument use mf_args = (arg1,)")

        # Specify logprior function and include any args or kwargs needed
        if logPrior is None:
            # Log Prior function returns zero by default
            self.logPrior = lambda p, X: 0.
        else:
            # Custom logPrior function which must take only a single argument p
            if isinstance(logPrior_args, tuple) and isinstance(logPrior_kwargs, dict):
                self.logPrior = lambda p, X: logPrior(p, X, *logPrior_args, **logPrior_kwargs)
            else:
                raise Exception("""logPrior_args must be a Tuple and logPrior_kwargs must be a dict.
                                   For a single argument use logPrior_args = (arg1,)""")
        
        # Identify GP optimisation to use
        self.build_kf = kernel_selector(p, self.x, kf = self.kf, use_kernel = use_kernel,
                                        verbose = verbose, kernel_select_kwargs = kernel_select_kwargs,
                                        **kernel_kwargs)
        self.kf_eval = self.build_kf(p, self.x)
        
        self.opt_name = self.kf_eval.opt_name
        if verbose:
            print(f"Using: {self.opt_name}")

            if "orthogonalise" in kernel_kwargs.keys() and kernel_kwargs["orthogonalise"] == True:
                if self.kf_eval.opt_name == "MultiLowRankKernel":
                    print("Will orthogonalise components, note this will ignore any non-orthogonal component of the whitened alpha vectors!")
                else:
                    raise Exception("Orthogonalise components only implemented for MultiLowRankKernel with N_alpha > 1")

        if jit:
            # Have to option to avoid JIT compiling which can sometimes be useful
            # e.g. it can make debugging easier, or if using a mean function which cannot be JIT compiled
            # Note that NumPyro will perform JIT compilation anyway but PyMC will not
            self.logL = jax.jit(self.logL)
            self.logP = jax.jit(self.logP)
            self.logL_stored = jax.jit(self.logL_stored)
            self.logP_stored = jax.jit(self.logP_stored)
            self.logL_hessianable = jax.jit(self.logL_hessianable)
            self.logP_hessianable = jax.jit(self.logP_hessianable)


    def sample(self, p, add_mf = True, samples = 1, return_variance = False):
        r"""Draws random samples from the GP. Useful for visualising a GP model and generating
        synthetic data.

        Samples are drawn using the matrix square root of the covariance matrix
        :math:`\mathbf{K} = \mathbf{L}\mathbf{L}^T` as :math:`\vec{y} = \vec{\mu} + \mathbf{L}\vec{z}`,
        where :math:`\vec{z}` is drawn from a standard normal distribution using ``numpy.random``
        (so ``np.random.seed`` can be used for reproducible draws).

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            add_mf (bool, optional): Whether to add the mean function to the samples. Defaults to ``True``.
            samples (int, optional): The number of samples to draw. Defaults to 1.
            return_variance (bool, optional): If ``True`` (and ``samples > 1``), return a Monte Carlo estimate
                of the variance at each data point from the samples instead of the samples themselves.
                Used by :meth:`get_diagonal`. Defaults to ``False``.

        Returns:
            JAXArray: For ``samples = 1``, a single sample with shape ``GP.data_shape``. For ``samples > 1``,
            an array of shape ``(samples, *GP.data_shape)``, or the estimated variance with shape
            ``GP.data_shape`` if ``return_variance = True``.

        """

        self.kf_eval = self.build_kf(p, self.x)
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = {})

        # The Conditional CovType has a non-square matrix square root
        # which expects a random normal vector of a different shape to the output data
        z_shape = getattr(self.kf_eval, "sqrt_input_shape", self.data_shape)

        if samples == 1:
            z = np.random.normal(size = z_shape)
            R = self.kf_eval.matrix_sqrt(z)
        else:
            z = np.random.normal(size = (samples, *z_shape))
            matrix_sqrt_jit = jax.jit(self.kf_eval.matrix_sqrt)

            if return_variance:
                variance = jnp.zeros(self.data_shape)
                for i in tqdm(range(samples)):
                    # for loop because vmap can be super memory intensive
                    variance += matrix_sqrt_jit(z[i, ...])**2
                return variance/(samples - 1)
            else:
                R = jnp.zeros((samples, *self.data_shape))
                for i in tqdm(range(samples)):
                    # for loop because vmap can be super memory intensive
                    R = R.at[i, ...].set(matrix_sqrt_jit(z[i, ...]))

        if add_mf:
            R += self.mf(p, self.x)
        
        return R

    def get_diagonal(self, p, samples = None, **kwargs):
        r"""Returns the diagonal of the covariance matrix, i.e. the variance at each data point.

        Uses the exact diagonal if the kernel being used can calculate it. Otherwise, if ``samples``
        is given, the diagonal is estimated from that many random samples (see :meth:`sample`), with an
        error which decreases as ``1/sqrt(samples)``.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            samples (int, optional): The number of samples to use if the diagonal has to be estimated.
                If not given and the exact diagonal isn't available, an Exception is raised.
            **kwargs: Passed to the kernel's ``diagonal`` method.

        Returns:
            JAXArray: The variance at each data point, with shape ``GP.data_shape``.

        """

        self.kf_eval = self.build_kf(p, self.x)

        if hasattr(self.kf_eval, "diagonal"):
            return self.kf_eval.diagonal(self.x, **kwargs)
        else:
            if samples is not None:
                print("Estimating diagonal of GP covariance matrix stochastically, this may take a while for large models...")
                diagonal_est = self.sample(p, samples = samples, return_variance = True)
                return diagonal_est
            else:
                raise Exception("Evaluating the diagonal of the covariance matrix not implemented!\
                                You can evaluate it stochastically, e.g. by setting samples = 1000")
    
    def logL(
        self,
        p: PyTree,
        Y: JAXArray,
    ) -> Scalar:
        r"""Computes the log likelihood.

        .. math::

            \log \mathcal{L} = -\frac{1}{2} \vec{r}^T \mathbf{K}^{-1} \vec{r}
            - \frac{1}{2} \log |\mathbf{K}| - \frac{N}{2} \log 2\pi

        where :math:`\vec{r} = \vec{y} - \vec{\mu}` are the residuals of the data after subtracting the
        mean function, :math:`\mathbf{K}` is the covariance matrix and :math:`N` is the total number of data points.

        This method is JIT compiled unless the GP was created with ``jit = False``. Use
        :meth:`logL_hessianable` instead if second derivatives (e.g. ``jax.hessian``) are needed.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.

        Returns:
            Scalar: The log likelihood.

        """
        
        # Calculate the residuals after subtraction of the deterministic mean function
        R = Y - self.mf(p, self.x)
        self.kf_eval = self.build_kf(p, self.x)
        
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = {})
        
        logL_val = self.kf_eval.logL(R, stored_values = stored_values)
        
        return logL_val
    

    def pymc_model(
        self,
        p: PyTree,
        Y: JAXArray,
        param_bounds: PyTree = {},
        vars: Optional[list] = None,
        fixed_vars: Optional[list] = None,
        bound_tol = 1e-9,
    ):
        r"""Builds a ``PyMC`` model of the GP posterior, for optimisation or sampling with ``PyMC``.

        Each parameter being varied is given a uniform prior between its bounds in ``param_bounds``
        (``pm.Uniform``), or an improper flat prior if it has no bounds (``pm.Flat``). The log posterior
        from :meth:`logP_stored`, which includes the GP's own log prior function, is added to the model as a
        ``pm.Potential``. Prints which parameters are being varied and which are fixed.

        Requires ``PyMC`` to be installed.

        Example:
            ``param_bounds`` should be of the form ``param_bounds[par] = [lower_bound, upper_bound]``,
            where the bounds are scalars or arrays with the same shape as ``p[par]``.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function. Also passed to the log prior function. Used as the values of any fixed parameters and as the starting values of the
                parameters being varied.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.
            param_bounds (PyTree, optional): Lower and upper bounds for any of the parameters being varied.
            vars (:obj:`list` of :obj:`str`, optional): The names of the parameters to vary. All other
                parameters are kept fixed. Cannot be given together with ``fixed_vars``.
            fixed_vars (:obj:`list` of :obj:`str`, optional): The names of the parameters to keep fixed, as an
                alternative to ``vars``. If neither is given, all parameters are varied.
            bound_tol (Scalar, optional): Starting values which lie exactly on a bound are moved inside it by
                this amount (see :func:`luas.convenience_fns.check_params_in_bounds`). Defaults to ``1e-9``.

        Returns:
            (pm.Model, dict, PyTree): A tuple of three elements: the ``PyMC`` model; a dictionary of the
            ``PyMC`` variables for each parameter being varied, together with the values of the fixed
            parameters; and the starting values of the parameters being varied (as NumPy arrays), which can
            be used as the starting point for ``pm.find_MAP`` or ``pm.sample``.

        """
        # Lazy import to avoid PyMC being a dependency unless needed
        from luas.pymc_ext import build_pymc_model

        # This function returns a PyTree p_vary which has only the (key, value) pairs of
        # parameters being varied
        # make_p is also returned which is a function with returns the parameters in p_vary
        # with all fixed parameters added back in
        p_vary, make_p = varying_params_wrapper(p, vars = vars, fixed_vars = fixed_vars)
        p_fixed = {k:v for (k, v) in p.items() if k not in p_vary.keys()}
        
        ordered_param_list = order_list(list(p_vary.keys()))

        p_vary = check_params_in_bounds(p_vary, param_bounds, bound_tol = bound_tol)

        print(f"Varying parameters: {ordered_param_list}\nFixed parameters: {list(p_fixed.keys())}")

        # build a PyMC model for optimisation and inference
        model, var_dict = build_pymc_model(self, p_vary, Y, ordered_param_list, p_fixed = p_fixed, param_bounds = param_bounds)
        
        return model, var_dict, p_vary



    def optimise(self, p_init, Y, param_bounds = {}, vars = None, fixed_vars = None, maxeval = 5000, include_transformed = False, **kwargs):
        r"""Finds the parameters which maximise the posterior (the maximum a posteriori fit) using
        ``pm.find_MAP``.

        Builds a ``PyMC`` model with :meth:`pymc_model` and optimises it starting from ``p_init``. Requires
        ``PyMC`` to be installed.

        Args:
            p_init (PyTree): Starting values of all the parameters. Parameters which aren't being varied are
                kept fixed at these values.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.
            param_bounds (PyTree, optional): Lower and upper bounds for any of the parameters being varied,
                in the form ``param_bounds[par] = [lower_bound, upper_bound]``.
            vars (:obj:`list` of :obj:`str`, optional): The names of the parameters to vary. All other
                parameters are kept fixed. Cannot be given together with ``fixed_vars``.
            fixed_vars (:obj:`list` of :obj:`str`, optional): The names of the parameters to keep fixed, as an
                alternative to ``vars``. If neither is given, all parameters are varied.
            maxeval (int, optional): The maximum number of optimisation steps. Defaults to 5000.
            include_transformed (bool, optional): Whether to also return the values of the bounded parameters
                in the unbounded space ``PyMC`` optimises in. Defaults to ``False``.
            **kwargs: Passed to ``pm.find_MAP``, e.g. ``method``.

        Returns:
            PyTree: A copy of ``p_init`` with the parameters being varied replaced by their optimised values.

        """
        # Lazy import to avoid PyMC being a dependency unless needed
        import pymc as pm

        pymc_model, var_dict, p_pymc = self.pymc_model(p_init, Y, param_bounds = param_bounds, vars = vars, fixed_vars = fixed_vars)

        # Use PyMC's maximum posteriori optimisation function
        map_estimate = pm.find_MAP(
            model = pymc_model,               # PyMC model to optimise
            include_transformed = include_transformed, # If this is true it will also output the PyMC transformed values of bounded parameters
            start = p_pymc,              # Starting point of optimisations
            maxeval = maxeval,             # Maximum steps to run (normally will converge before this)
            **kwargs,
        )

        # Create a new dictionary of optimised values which includes our fixed parameters
        p_opt = deepcopy(p_init)
        p_opt.update(map_estimate)

        return p_opt
    

    def logL_stored(
        self,
        p: PyTree,
        Y: JAXArray,
        stored_values: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        r"""Computes the log likelihood and also returns values stored from the decomposition of
        the covariance matrix.

        Takes the same inputs and returns the same outputs as the function ``luas`` uses for inference with
        ``PyMC``, which passes the stored values from each call on to the next. The input ``stored_values``
        are passed on to the decomposition and reused by kernels which support this (``luas.LuasKernel``
        created with ``use_stored_values = True``), which saves time when some hyperparameters are fixed or
        are sampled separately (e.g. with blocked Gibbs sampling).

        Note:
            This method does not give accurate second derivatives (e.g. using ``jax.hessian``). Use
            :meth:`logL_hessianable_stored` if they are needed.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.
            stored_values (PyTree): Values stored from a previous decomposition of the covariance
                matrix, such as the output of a previous call to this method. Their contents depend on
                the kernel class being used.

        Returns:
            (Scalar, PyTree): A tuple of the log likelihood and the values stored from the decomposition of
            the covariance matrix.

        """
        
        R = Y - self.mf(p,  self.x)
        
        self.kf_eval = self.build_kf(p, self.x)
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = stored_values)  
        logL_val = self.kf_eval.logL(R, stored_values = stored_values)
    
        return logL_val, stored_values

    
    def logP(
        self,
        p: PyTree,
        Y: JAXArray,
    ) -> Scalar:
        r"""Computes the log posterior (up to a constant), i.e. the log prior plus the log likelihood.

        .. math::

            \log \mathcal{P} = \log \pi(\vec{p}) + \log \mathcal{L}

        where :math:`\log \pi(\vec{p})` is the log prior function the GP was created with, and the log
        likelihood is calculated as in :meth:`logL`.

        This method is JIT compiled unless the GP was created with ``jit = False``. Use
        :meth:`logP_hessianable` instead if second derivatives (e.g. ``jax.hessian``) are needed.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function. Also passed to the log prior function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.

        Returns:
            Scalar: The log posterior.

        """

        R = Y - self.mf(p,  self.x)

        self.kf_eval = self.build_kf(p, self.x)
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = {})  
        logL_val = self.kf_eval.logL(R, stored_values = stored_values)

        logPrior = self.logPrior(p, self.x)
        logP = logPrior + logL_val
        
        return logP


    def logP_stored(
        self,
        p: PyTree,
        Y: JAXArray,
        stored_values: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        r"""Computes the log posterior and also returns values stored from the decomposition of the
        covariance matrix.

        This is the function ``luas`` uses by default for inference with ``PyMC`` (see :meth:`pymc_model`),
        which passes the stored values from each call on to the next. The input ``stored_values`` are passed
        on to the decomposition and reused by kernels which support this (``luas.LuasKernel`` created with
        ``use_stored_values = True``), which saves time when some hyperparameters are fixed or are sampled
        separately (e.g. with blocked Gibbs sampling).

        Note:
            This method does not give accurate second derivatives (e.g. using ``jax.hessian``). Use
            :meth:`logP_hessianable_stored` if they are needed.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function. Also passed to the log prior function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.
            stored_values (PyTree): Values stored from a previous decomposition of the covariance
                matrix, such as the output of a previous call to this method. Their contents depend on
                the kernel class being used.

        Returns:
            (Scalar, PyTree): A tuple of the log posterior and the values stored from the decomposition of
            the covariance matrix.

        """

        
        R = Y - self.mf(p,  self.x)
        
        self.kf_eval = self.build_kf(p, self.x)
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = stored_values)  
        logL_val = self.kf_eval.logL(R, stored_values = stored_values)

        logPrior = self.logPrior(p, self.x)
        logP = logPrior + logL_val
        
        return logP, stored_values

    
    def crossval(self, p, Y, params_to_stack = None, train_ind = None, **kwargs):
        r"""Cross-validation for 2D data: fits the GP to a subset of the rows and/or columns of the
        data and predicts the rest.

        The GP is conditioned on the training rows and columns selected by ``train_ind``, and the remaining
        rows and columns are predicted using :meth:`predict`. Comparing the predictions with the held-out
        data tests how well the GP model predicts data it wasn't fit to.

        Example:
            To train on every other row and all columns, with a transit depth ``p["rp"]`` that has one value
            per row::

                train_l = np.arange(N_l) % 2 == 0
                gp_mean, M, gp_cond = gp.crossval(p, Y, params_to_stack = [["rp"], []],
                                                  train_ind = (train_l, None))

        Args:
            p (PyTree): Parameters for the full data set, including any hyperparameters needed by the kernel
                function and any parameters needed by the mean function.
            Y (JAXArray): Observed data for the full data set, with shape ``(N_l, N_t)``.
            params_to_stack (list): A list ``[params_l, params_t]`` of two lists of parameter names. The
                parameters in ``params_l`` have one value per row (e.g. a wavelength-dependent transit depth)
                and are split between the training and predicted rows, and likewise ``params_t`` for columns.
                Use an empty list for a dimension which isn't split.
            train_ind (tuple): A tuple ``(train_l, train_t)`` of boolean masks of shape ``(N_l,)`` and
                ``(N_t,)`` which are ``True`` for the rows and columns to train on. Either can be ``None`` to
                train on all rows or all columns, but not both.
            **kwargs: Passed to :meth:`predict`, e.g. ``return_predicted_loc``.

        Returns:
            (JAXArray, JAXArray, GP): The outputs of :meth:`predict`: the GP predictive mean, the mean function
            at the predicted locations and the conditional GP object.

        """

        x_l, x_t = self.x
        train_l, train_t = train_ind
        stack_p_dim1, stack_p_dim2 = params_to_stack
    
        p_obs = deepcopy(p)
        p_pred = deepcopy(p)

        if train_l is None and train_t is None:
            raise Exception("Both train_l and train_t cannot be None!")
    
        if train_l is not None:
            x_l_pred = x_l[..., ~train_l]
            x_l = x_l[..., train_l]
            Y = Y[train_l, :]
            
            for par in stack_p_dim1:
                p_pred[par] = p_obs[par][~train_l]
                p_obs[par] = p_obs[par][train_l]
        else:
            x_l_pred = None
    
        if train_t is not None:
            x_t_pred = x_t[..., ~train_t]
            x_t = x_t[..., train_t]
            Y = Y[:, train_t]
            
            for par in stack_p_dim2:
                p_pred[par] = p_obs[par][~train_t]
                p_obs[par] = p_obs[par][train_t]
        else:
            x_t_pred = None

        X_obs = (x_l, x_t)
        X_pred = (x_l_pred, x_t_pred)
    
        gp_mean_pred, M_pred, gp_cond_obj = self.predict(
            p_obs, Y,
            p_pred = p_pred,
            X_obs = X_obs,
            X_pred = X_pred,
            params_to_stack= params_to_stack,
            **kwargs,
        )
        return gp_mean_pred, M_pred, gp_cond_obj
    
    
    
    def predict(self, p, Y,
                p_pred = None,
                X_obs = None,
                X_pred = None,
                params_to_stack = None,
                params_to_interpolate = None,
                input_vecs_for_interp = None,
                sort = None,
                fast_dim = None, return_predicted_loc = True,
                return_pred_err = False, samples = 1000):
    
        r"""Makes GP predictions: computes the GP predictive mean, and optionally the predictive
        uncertainty, at a set of prediction locations given the observed data.

        The GP predictive mean and covariance at the prediction locations are

        .. math::

            \mathbb{E}[\vec{y}_*] = \vec{\mu}_* + \mathbf{K}_*^T \mathbf{K}^{-1} (\vec{y} - \vec{\mu})

        .. math::

            \mathrm{Var}[\vec{y}_*] = \mathbf{K}_{**} - \mathbf{K}_*^T \mathbf{K}^{-1} \mathbf{K}_*

        where :math:`\mathbf{K}` is the covariance matrix of the observed data, :math:`\mathbf{K}_*` is the
        covariance between the observed and prediction locations, :math:`\mathbf{K}_{**}` is the covariance
        matrix of the prediction locations, and :math:`\vec{\mu}` and :math:`\vec{\mu}_*` are the mean
        function at the observed and prediction locations.

        To do this, the observed and prediction locations along each dimension are stacked together (and
        sorted, see ``sort``), and the kernel is evaluated over the full stacked set of locations (see
        :func:`luas.convenience_fns.stack_obs_and_pred_points`). The hyperparameters are assumed to be the
        same at the observed and prediction locations, except for any parameters given in
        ``params_to_stack`` or ``params_to_interpolate``.

        Note:
            The predictive uncertainty is estimated from random samples of the conditional GP, which can be
            slow for large data sets. The returned conditional GP object can also be used to draw samples
            from the predictive distribution directly, e.g. ``gp_cond.sample(p_pred)``.

        Args:
            p (PyTree): Parameters at the observed locations, including any hyperparameters needed by the
                kernel function and any parameters needed by the mean function.
            Y (JAXArray): Observed data at the locations ``X_obs``.
            p_pred (PyTree, optional): Parameters at the prediction locations. Only needed for parameters
                which take different values at the prediction locations (see ``params_to_stack``).
                Defaults to a copy of ``p``.
            X_obs (JAXArray | tuple, optional): Observed input locations, in the same format as ``X`` when
                creating the GP. Defaults to the input locations the GP was created with.
            X_pred (JAXArray | tuple, optional): Prediction locations. For 1D data a JAXArray. For 2D data a
                tuple ``(x_l_pred, x_t_pred)``, where either entry can be ``None`` to make predictions only at
                the observed locations along that dimension. If not given, predictions are made at the
                observed locations.
            params_to_stack (list, optional): Names of parameters which have one value per input location
                (e.g. a wavelength-dependent transit depth). Their values from ``p`` and ``p_pred`` are joined
                to cover the full set of stacked locations (see
                :func:`luas.convenience_fns.stack_params`). For 1D data a list of names, for 2D data a list
                with one list of names per dimension.
            params_to_interpolate (list, optional): Names of parameters which have one value per input
                location, whose values at the prediction locations are linearly interpolated from their
                values in ``p`` (see :func:`luas.convenience_fns.interp_params`). For 1D data a list of
                names, for 2D data a list with one list of names (or ``None``) per dimension.
            input_vecs_for_interp (int | list, optional): If the input locations have more than one row
                (shape ``(d, N)``), the row to interpolate over. For 2D data a list with one entry per
                dimension, which must be given if ``params_to_interpolate`` is.
            sort (bool | list, optional): Whether to sort the stacked input locations along each dimension
                (one bool per dimension for 2D data). Most quasiseparable kernels need sorted inputs.
                Defaults to ``True`` for every dimension.
            fast_dim (int, optional): For 2D data when ``X_pred`` isn't given, the dimension (0 or 1) along
                which a copy of the observed locations is stacked to make predictions at the observed
                locations. Defaults to 0.
            return_predicted_loc (bool, optional): For 2D data, if ``True`` only return predictions at the
                prediction locations, otherwise return predictions at the full stacked set of observed and
                prediction locations. Defaults to ``True``.
            return_pred_err (bool, optional): Whether to also estimate the predictive standard deviation at
                each prediction location. Defaults to ``False``.
            samples (int, optional): The number of random samples used to estimate the predictive standard
                deviation if ``return_pred_err = True``. The error in the estimate decreases as
                ``1/sqrt(samples)``. Defaults to 1000.

        Returns:
            tuple: If ``return_pred_err = False``, a tuple ``(gp_mean, M_pred, gp_cond)``, otherwise
            ``(gp_mean, pred_err, M_pred, gp_cond)``, where:

            - **gp_mean** (JAXArray) - The GP predictive mean, including the mean function.
            - **pred_err** (JAXArray) - The estimated predictive standard deviation.
            - **M_pred** (JAXArray) - The mean function at the prediction locations.
            - **gp_cond** (GP) - A GP object describing the conditional (predictive) distribution at the
              prediction locations, which takes ``p_pred`` as its parameters.

        """
        if self.dim != 1 and self.dim != 2:
            raise Exception(f"GP prediction only implemented for 1D and 2D GPs, not {self.dim} dimensions")
        
        p_obs = deepcopy(p)

        if X_obs is None:
            # If observed locations not specified, assume training on the full dataset
            X_obs = deepcopy(self.x)
        
        if p_pred is None:
            # If predicted params not specified, assume they match the observed parameters
            # But may stack/interpolate to new predicted locations
            p_pred = deepcopy(p_obs)

        if sort is None:
            # Important to sort for most quasisep kernels
            sort = [True]*self.dim
        
        if X_pred is None:
            # If no prediction locations given then taken to be same as observed locations
            pred_loc_equals_obs = True

            if self.dim == 1:
                X_pred = X_obs.copy()
            elif fast_dim is None or fast_dim == 0:
                X_pred = [X_obs[0], None]
            elif fast_dim == 1:
                X_pred = [None, X_obs[1]]
            else:
                raise Exception("fast_dim keyword argument must be None, 0 or 1")
        else:
            X_pred = list(X_pred)
            pred_loc_equals_obs = False

    
        # Stacks or interpolates parameters from observed locations to predicted locations for each dimension
        # Works out location of predicted and observed locations within array
        # Will sort the full stacked regression variables if requested, typically necessary for quasisep dimensions
        if self.dim == 1:
            X_full, X_pred, obs_points, pred_points, sort_fn = stack_obs_and_pred_points(X_obs, X_pred, sort = sort)
            N_full = X_full.shape[-1]
            sort_fns = [sort_fn]

            if params_to_stack is not None:
                params_to_stack = [params_to_stack]
            observed_ind = jnp.arange(N_full)[obs_points]
            predicted_ind = jnp.arange(N_full)[pred_points]
            p_pred = interp_params(p_obs, p_pred, X_obs, X_pred,
                                    interp_param = params_to_interpolate,
                                    interp_row = input_vecs_for_interp)

        elif self.dim == 2:
            x_l_full, X_pred[0], obs_points_l, pred_points_l, sort_fn_l = stack_obs_and_pred_points(X_obs[0], X_pred[0], sort = sort[0])
            x_t_full, X_pred[1], obs_points_t, pred_points_t, sort_fn_t = stack_obs_and_pred_points(X_obs[1], X_pred[1], sort = sort[1])
            sort_fns = [sort_fn_l, sort_fn_t]
            X_full = (x_l_full, x_t_full)
            N_full = calc_data_shape(X_full)
            
            observed_ind = jnp.ix_(jnp.arange(N_full[0])[obs_points_l],
                                jnp.arange(N_full[1])[obs_points_t])
        
            if return_predicted_loc:
                # Will only output locations at grid inputs of (X_pred)
                predicted_ind = jnp.ix_(jnp.arange(N_full[0])[pred_points_l],
                                        jnp.arange(N_full[1])[pred_points_t])
            else:
                # Returns the full output locations at X_full
                predicted_ind = jnp.ix_(jnp.arange(N_full[0]), jnp.arange(N_full[1]))

            if params_to_interpolate is not None:
                for d in range(self.dim):
                    p_pred = interp_params(p_obs, p_pred, X_obs[d], X_pred[d],
                                        interp_param = params_to_interpolate[d],
                                        interp_row = input_vecs_for_interp[d])


        # GP mean calculation
        # Calculate mean function at observed locations
        M_obs = self.mf(p_obs, X_obs)
        R = Y - M_obs

        # Calculate K^-1 R
        self.kf_eval = self.build_kf(p_obs, X_obs)
        self.kf_eval, stored_values = self.kf_eval.decompose(X_obs)
        K_inv_R = self.kf_eval.inverse(R)
        logdet_K_train = stored_values["logdet"]

        # Keep the decomposed training kernel, the conditional covariance needs K^-1
        kf_eval_train = self.kf_eval

        # Calculate K_*^T K_inv_R by building the full ((K, K_*), (K_*^T, K_**))
        K_inv_R_to_mult = jnp.zeros(N_full)
        K_inv_R_to_mult = K_inv_R_to_mult.at[observed_ind].set(K_inv_R)

        # mf of conditional GP
        def gp_mean_cond_fn(p_pred, X_full):
            p_full = stack_params(p_obs, p_pred, stack_names = params_to_stack, sort_fns = sort_fns)

            self.kf_eval_full = self.build_kf(p_full, X_full)
            Ks_Kinv_R_term = self.kf_eval_full.matmul(X_full, X_full, K_inv_R_to_mult, wn = False, full = True)

            if pred_loc_equals_obs:
                M_pred = M_obs.copy()
            elif return_predicted_loc:
                M_pred = self.mf(p_full, X_full)[predicted_ind]
            else:
                M_pred = self.mf(p_full, X_full)

            gp_mean_pred = M_pred + Ks_Kinv_R_term[predicted_ind]
            return gp_mean_pred, M_pred

        def kf_cond(p_pred, X_full):
            p_full = stack_params(p_obs, p_pred, stack_names = params_to_stack, sort_fns = sort_fns)
            
            kf_eval_full = self.build_kf(p_full, X_full)
            return Conditional(kf_eval_full, kf_train = kf_eval_train, X_full = X_full,
                                     observed_ind = observed_ind, predicted_ind = predicted_ind,
                                     logdet_K_train = logdet_K_train)

        mf_cond = lambda p_pred, X_full: gp_mean_cond_fn(p_pred, X_full)[0]
        gp_cond = GP(p_pred, X_full, mf = mf_cond, kf = kf_cond)
        gp_cond.data_shape = calc_data_shape(X_pred)

        gp_mean_pred, M_pred = gp_mean_cond_fn(p_pred, X_full)

        if return_pred_err:
            # Currently only estimates diagonal of predicted covariance stochastically
            # Default is samples = 1000 which may be expensive for large models
            # Relative error in estimate is 1/sqrt(samples): 1000 samples -> ~3.2% error in estimate
            pred_err = jnp.sqrt(gp_cond.get_diagonal(p_pred, samples = samples))
            return gp_mean_pred, pred_err, M_pred, gp_cond
        else:
            return gp_mean_pred, M_pred, gp_cond
        

    def gp_mean_calc(
            self,
            p,
            Y,
            return_mf = False,
    ):
        r"""Computes the GP predictive mean at the observed locations.

        .. math::

            \mathbb{E}[\vec{y}] = \vec{\mu} + \mathbf{K}_{\mathrm{nw}} \mathbf{K}^{-1} (\vec{y} - \vec{\mu})

        where :math:`\mathbf{K}_{\mathrm{nw}}` is the covariance matrix without white noise. Faster than
        :meth:`predict` when only the fit at the observed locations is needed, and used by :meth:`plot`.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.
            return_mf (bool, optional): Whether to also return the mean function. Defaults to ``False``.

        Returns:
            JAXArray or (JAXArray, JAXArray): The GP predictive mean (including the mean function), with
            shape ``GP.data_shape``. If ``return_mf = True``, a tuple of the GP predictive mean and the mean
            function.

        """
        # Perform GP regression at the observed data locations
        M = self.mf(p, self.x)
        
        res = Y - M
        K_inv_R = self.apply_inverse(p, res)
        gp_mean = M + self.matmul(p, K_inv_R, wn = False, full = True)

        if return_mf:
            return gp_mean, M
        else:
            return gp_mean

    
    def laplace_approx(
        self,
        p: PyTree,
        Y: PyTree,
        regularise: Optional[bool] = True,
        regularise_const: Optional[Scalar] = 1e-10,
        vars: Optional[list] = None,
        fixed_vars: Optional[list] = None, 
        return_array: Optional[bool] = False,
        large: Optional[bool] = False,
        large_block_size: Optional[int] = 50,
        large_jit: Optional[bool] = True,
        logP_fn: Optional[Callable] = None,
        hessian_mat: Optional[JAXArray] = None,
    ) -> Tuple[Union[PyTree, JAXArray], list]:
        r"""Computes the Laplace approximation of the posterior at the location of ``p``, which should
        be the best-fit (maximum a posteriori) parameters.

        The Laplace approximation is a Gaussian approximation of the posterior centred on the best fit. Its
        covariance matrix is the negative inverse of the Hessian of the log posterior at the best fit, which is
        equivalent to a second-order Taylor series approximation of the log posterior. It gives a quick estimate
        of the posterior without running MCMC, and is also useful for initialising MCMC with a good tuning
        matrix when many correlated parameters are being sampled.

        Note:
            This calculation can be memory intensive for large data sets with many free parameters. Setting
            ``large = True`` with a small ``large_block_size`` reduces the memory cost by calculating the
            Hessian in blocks of rows.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function. Also passed to the log prior function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.
            regularise (bool, optional): Whether to regularise the covariance matrix so it is positive definite.
                Any eigenvalues of the covariance matrix smaller than ``regularise_const`` times the largest
                absolute eigenvalue are raised to that value, which handles poorly constrained parameters.
                Defaults to ``True``.
            regularise_const (Scalar, optional): The relative size of the smallest eigenvalue allowed when
                ``regularise = True``. Defaults to ``1e-10``.
            vars (:obj:`list` of :obj:`str`, optional): The names of the parameters to calculate the Laplace
                approximation for. All other parameters are kept fixed. Cannot be given together with
                ``fixed_vars``.
            fixed_vars (:obj:`list` of :obj:`str`, optional): The names of the parameters to keep fixed, as an
                alternative to ``vars``. If neither is given, all parameters are included.
            return_array (bool, optional): Whether to return the covariance matrix as a JAXArray, or as a nested
                PyTree where the covariance between parameters ``p1`` and ``p2`` is ``cov_mat[p1][p2]``.
                Defaults to ``False``.
            large (bool, optional): Whether to calculate the Hessian in blocks of rows to reduce the memory cost
                (see :func:`luas.convenience_fns.large_hessian_calc`). This can take significantly longer but
                makes no approximation. Defaults to ``False``.
            large_block_size (int, optional): The number of rows of the Hessian calculated at once when
                ``large = True``. Larger values may be faster but cost more memory. Defaults to 50.
            large_jit (bool, optional): Whether to JIT compile the Hessian calculation when ``large = True``.
                Defaults to ``True``.
            logP_fn (Callable, optional): The function to take the Hessian of, called as ``logP_fn(p, Y)``.
                Defaults to :meth:`logP_hessianable`.
            hessian_mat (JAXArray, optional): A Hessian matrix to use instead of calculating it, as a 2D JAXArray
                ordered as ``jax.flatten_util.ravel_pytree`` orders the parameters being varied. ``Y`` is then
                not used.

        Returns:
            (JAXArray, :obj:`list` of :obj:`str`) or (PyTree, :obj:`list` of :obj:`str`): A tuple of the
            covariance matrix of the Laplace approximation (as a JAXArray if ``return_array = True``, otherwise
            as a nested PyTree) and the list of parameter names in the order they appear in the JAXArray. The
            list matches how ``jax.flatten_util.ravel_pytree`` orders the parameters and is returned in both
            cases for consistency.

        """
        
        if logP_fn is None:
            logP_fn = self.logP_hessianable

        # This function returns a PyTree p_vary which has only the (key, value) pairs of
        # parameters being varied
        # make_p is also returned which is a function with returns the parameters in p_vary
        # with all fixed parameters added back in
        p_vary, make_p = varying_params_wrapper(p, vars = vars, fixed_vars = fixed_vars)

        if hessian_mat is None:
            # Generate a wrapper function which takes only the parameters being varied and
            # calculates the log posterior (this avoids calculating derivatives of fixed parameters)
            logP_hessianable_wrapper = lambda p_vary: logP_fn(make_p(p_vary), Y)
            
            if large:
                # For large data sets and many free parameters, breaking up the hessian calculation
                # into blocks of rows of size large_block_size has a much lower memory cost
                hessian_mat = large_hessian_calc(logP_hessianable_wrapper, p_vary, block_size = large_block_size,
                                                 return_array = True, jit = large_jit)
            else:
                # If memory cost is not an issue then directly calculating the full hessian is faster
                hessian_mat = jax.hessian(logP_hessianable_wrapper)(p_vary)
                
                # For inverting to a covariance matrix we need to convert the nested PyTree returned by
                # jax.hessian into a matrix which we can do with this helper function from luas.convenience_fns
                hessian_mat = pytree_to_array_2D(p_vary, hessian_mat)
                
        # Help symmetrise matrix which can help mitigate numerical errors
        hessian_mat = (hessian_mat + hessian_mat.T)/2.
        
        if regularise:
            hess_eigvals, eigvecs = jnp.linalg.eigh(hessian_mat)
            eigvals = -1/hess_eigvals

            epsilon = jnp.max(jnp.abs(eigvals))*regularise_const
            eigvals_clipped = jnp.maximum(eigvals, epsilon)
            cov_mat = eigvecs @ jnp.diag(eigvals_clipped) @ eigvecs.T
        else:
            # Performs the actual Laplace approximation by inverting the negative hessian
            cov_mat = jnp.linalg.inv(-hessian_mat)
        

        # Generate the list which gives the order of the parameters in the covariance matrix
        ordered_param_list = order_list(list(p_vary.keys()))
        
        if return_array:
            return cov_mat, ordered_param_list
        else:
            # If returning a nested PyTree use array_to_pytree_2D to convert
            return array_to_pytree_2D(p_vary, cov_mat), ordered_param_list
    
    
    def laplace_approx_with_bounds(
        self,
        p: PyTree,
        Y: JAXArray,
        param_bounds: PyTree,
        vars: Optional[list] = None,
        fixed_vars: Optional[list] = None,
        large: Optional[bool] = False,
        large_block_size: Optional[int] = 50,
        return_array: Optional[bool] = False,
        large_jit: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[Union[PyTree, JAXArray], list]:
        r"""Computes the Laplace approximation at the location of ``p`` in the unbounded space that
        ``PyMC`` and ``NumPyro`` sample bounded parameters in.

        ``PyMC`` and ``NumPyro`` sample a parameter bounded between :math:`a` and :math:`b` as an unbounded
        variable :math:`x`, which is mapped to the bounded parameter using a sigmoid transform (see
        :func:`luas.convenience_fns.transf_from_unbounded`). This method calculates the Laplace approximation
        with respect to these unbounded variables, including the log Jacobian of the transform, so the result
        can be used as a tuning matrix for sampling with these libraries.

        See :meth:`laplace_approx` for more details about the Laplace approximation.

        Example:
            ``param_bounds`` should be of the form ``param_bounds[par] = [lower_bound, upper_bound]``, where
            ``lower_bound`` and ``upper_bound`` have the same shape as ``p[par]``.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function. Also passed to the log prior function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.
            param_bounds (PyTree): Lower and upper bounds for any of the parameters in ``p``.
            vars (:obj:`list` of :obj:`str`, optional): The names of the parameters to calculate the Laplace
                approximation for. All other parameters are kept fixed. Cannot be given together with
                ``fixed_vars``.
            fixed_vars (:obj:`list` of :obj:`str`, optional): The names of the parameters to keep fixed, as an
                alternative to ``vars``. If neither is given, all parameters are included.
            large (bool, optional): Whether to calculate the Hessian in blocks of rows to reduce the memory cost.
                Defaults to ``False``.
            large_block_size (int, optional): The number of rows of the Hessian calculated at once when
                ``large = True``. Defaults to 50.
            return_array (bool, optional): Whether to return the covariance matrix as a JAXArray, or as a nested
                PyTree. Defaults to ``False``.
            large_jit (bool, optional): Whether to JIT compile the Hessian calculation when ``large = True``.
                Defaults to ``True``.
            **kwargs: Passed to :meth:`laplace_approx`, e.g. ``regularise``.

        Returns:
            (JAXArray, :obj:`list` of :obj:`str`) or (PyTree, :obj:`list` of :obj:`str`): A tuple of the
            covariance matrix of the Laplace approximation with respect to the unbounded variables (as a JAXArray
            if ``return_array = True``, otherwise as a nested PyTree) and the list of parameter names in the order
            they appear in the JAXArray.

        """

        # This function returns a PyTree p_vary which has only the (key, value) pairs of
        # parameters being varied
        # make_p is also returned which is a function with returns the parameters in p_vary
        # with all fixed parameters added back in
        p_vary, make_p = varying_params_wrapper(p, vars = vars, fixed_vars = fixed_vars)

        # Check that parameters to be varied are actually within the specified bounds
        p_vary = check_params_in_bounds(p_vary, param_bounds)
        
        # Transform the parameters being varied to the transformed values which are sampled by
        # PyMC and NumPyro
        p_transf = transf_to_unbounded_params(p_vary, param_bounds)
        
        # Create a function which returns the transformed values back to the full set of parameters
        # untransformed including fixed parameters
        def transf_back_to_p(p_transf):
            p_vary = transf_from_unbounded_params(p_transf, param_bounds)
            return make_p(p_vary)

        # Write a wrapper function which takes the transformed parameters and calculates the log Posterior
        pymc_logP_hessianable = lambda p_transf: self.logP_hessianable(transf_back_to_p(p_transf), Y)

        if large:
            # For large data sets and many free parameters, breaking up the hessian calculation
            # into blocks of rows of size large_block_size has a much lower memory cost
            hessian_mat = large_hessian_calc(pymc_logP_hessianable, p_transf, block_size = large_block_size,
                                             return_array = False, jit = large_jit)
        else:
            # If memory cost is not an issue then directly calculating the full hessian is faster
            hessian_mat = jax.hessian(pymc_logP_hessianable)(p_transf)
        
        # Loop over each parameter being varied
        for par in p_transf.keys():
            # Select just the bounded parameters
            if par in param_bounds.keys():

                # Reshape to ensure hessian wrt 2D arrays can be dealt with by flattening
                n = np.size(p_transf[par])
                hessian_mat[par][par] = jnp.reshape(hessian_mat[par][par], (n, n))

                # Add to the diagonal of the hessian an additional term
                # which is equal to the hessian of the jacobian of the transformation
                # performed by PyMC and NumPyro
                # This term is added to ensure the transformation these inference libraries perform
                # does not impact the choice of priors made
                exp_minus_p = jnp.exp(-p_transf[par])
                hessian_mat[par][par] += jnp.diag(jnp.ravel(-2*exp_minus_p/(1+exp_minus_p)**2))


        # Convert hessian from a nested PyTree to a 2D JAXArray for GP.laplace_approx to be able
        # to invert to calculate the covariance matrix
        hessian_mat = pytree_to_array_2D(p_transf, hessian_mat)
        cov_mat, ordered_param_list = self.laplace_approx(p_transf, Y, hessian_mat = hessian_mat,
                                                          return_array = return_array, **kwargs)
            
        return cov_mat, ordered_param_list

    
    def plot_cov_mat(
        self,
        p: PyTree,
        corr: Optional[bool] = False,
        wn: Optional[bool] = True,
        x_plot: Optional[JAXArray] = None,
        full: Optional[bool] = False,
        axes_labels = None,
        return_fig = False,
    ) -> plt.Figure:
        r"""Plots the covariance matrices along each dimension built from the parameters.

        For 1D data this plots the full covariance matrix. For 2D (or higher) data each column of the figure
        is one dimension, with the first row showing the :math:`\mathbf{\Sigma}` (noise) matrix along that
        dimension and each following row showing the matrix from one of the other Kronecker terms. This needs
        a Kronecker kernel class which stores these matrices (e.g. ``luas.LuasKernel``).

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function. Any extra
                parameters (e.g. for the mean function) are ignored.
            corr (bool, optional): If ``True``, plot correlation matrices instead of covariance matrices.
                Defaults to ``False``.
            wn (bool, optional): Whether to include white noise in the covariance matrices. Defaults to ``True``.
            x_plot (tuple, optional): The values along each axis used by ``plt.pcolormesh``, with one JAXArray
                per dimension. Defaults to the input locations the GP was created with (the first row, if there
                are several input variables along a dimension).
            full (bool, optional): Plotting the full covariance matrix of a 2D data set is not implemented, so
                this must be ``False``.
            axes_labels (:obj:`list` of :obj:`str`, optional): Axis labels for each dimension. Defaults to
                ``$x_1$``, ``$x_2$``, ...
            return_fig (bool, optional): Whether to return the figure. Defaults to ``False``.

        Returns:
            plt.Figure: The figure, if ``return_fig = True``.

        """
    
        # If no x and y axes for the plots specified, defaults to x_l, x_t
        # If x_l or x_t contain multiple rows then pick the first row
        if x_plot is None:
            x_plot = ()
            if self.dim == 1:
                x_plot = (self.x,)
            else:
                for d in range(self.dim):
                    if self.x[d].ndim > 1:
                        x_plot += (self.x[d][0],)
                    else:
                        x_plot += (self.x[d],)

        if axes_labels is None:
            axes_labels = [rf"$x_{d+1}$" for d in range(self.dim)]
        
        self.kf_eval = self.build_kf(p, self.x)
        N_kron_terms = 1 + len(self.kf_eval.K_list)

        if self.dim == 1:
            K_list = [[self.kf_eval.evaluate(self.x, self.x, wn = wn, full = True)]]
        else:
            K_list = []
            for d in range(self.dim):
                K_list.append([self.kf_eval.Sigma[d].evaluate(self.x[d], self.x[d], wn = wn, full = True)])
                
                # Build each component matrix
                for i in range(N_kron_terms-1):
                    K_list[-1].append(self.kf_eval.K_list[i][d].evaluate(self.x[d], self.x[d], wn = wn, full = True))


        fig, ax = plt.subplots(N_kron_terms, self.dim, figsize = (5*self.dim, 5*N_kron_terms), squeeze=False)

        if full:
            raise Exception("Plotting full covariance matrix not implemented")
        else:
            # Individually plot each of the 4 component covariance matrices

            for d in range(self.dim):
                for i in range(N_kron_terms):
                    if corr:
                        K_list[d][i] = get_corr_mat(K_list[d][i])
    
                    if i == 0:
                        ax[0][d].set_title(rf"$\Sigma_d$$_{d+1}$")
                    elif i == 1 and N_kron_terms == 2:
                        ax[1][d].set_title(rf"K$_d$$_{d+1}$")
                    else:
                        ax[i][d].set_title(rf"K$_d$$_{d+1}$[{i}]")
                        
                    ax[i][d].set_ylabel(axes_labels[d])
                    ax[i][d].set_xlabel(axes_labels[d])
                    
                    img1 = ax[i][d].pcolormesh(x_plot[d], x_plot[d], K_list[d][i])
                    ax[i][d].invert_yaxis()
                    plt.colorbar(mappable = img1, ax = ax[i][d])
            
            plt.tight_layout()

        if return_fig:
            return fig

    

    def plot(
        self,
        p: PyTree,
        Y: JAXArray,
        x_plot = None,
        return_fig = False,
        return_fit = False,
        axes_labels = [r"$x_1$", r"$x_2$"],
        sharey = None,
        **fig_kwargs,
    ) -> plt.Figure:
        r"""Visualises the fit to the data.

        Plots four panels: the observed data, the mean function, the GP predictive mean without the mean
        function, and the residuals after subtracting the GP predictive mean (including the mean function).
        For 1D data the first panel also overlays the mean function and the GP predictive mean on the data.
        For 2D data each panel is a ``plt.pcolormesh`` image.

        For a good fit, the residuals should look like white noise with no remaining correlations, and the
        GP predictive mean without the mean function should only be fitting correlated noise, not structure
        which belongs in the mean function.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function and any
                parameters needed by the mean function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``).
            x_plot (tuple, optional): The values along each axis of the plot, with one JAXArray per dimension.
                Defaults to the input locations the GP was created with (the first row, if there are several
                input variables along a dimension).
            return_fig (bool, optional): Whether to return the figure. Defaults to ``False``.
            return_fit (bool, optional): Whether to return the GP predictive mean and mean function (only if
                ``return_fig = False``). Defaults to ``False``.
            axes_labels (:obj:`list` of :obj:`str`, optional): Axis labels for each dimension.
                Defaults to ``["$x_1$", "$x_2$"]``.
            sharey (bool, optional): Whether the panels share a y-axis. Defaults to ``False`` for 1D data and
                ``True`` for 2D data.
            **fig_kwargs: Passed to ``plt.figure``.

        Returns:
            plt.Figure or (JAXArray, JAXArray): The figure if ``return_fig = True``, or a tuple of the GP
            predictive mean and the mean function if ``return_fit = True``. Otherwise nothing is returned.

        """
        
        # If no x and y axes for the plots specified, defaults to x_l, x_t
        # If x_l or x_t contain multiple rows then pick the first row
        if x_plot is None:
            x_plot = ()
            if self.dim == 1:
                x_plot = (self.x,)
            else:
                for x in self.x:
                    if x.ndim == 1:
                        x_plot += (x.copy(),)
                    else:
                        x_plot += (x[0, :],)

        if sharey is None:
            # default to sharey = False for 1D fits, but True for 2D fits
            if self.dim == 1:
                sharey = False
            else:
                sharey = True
    
        # Perform GP regression at the observed data locations
        gp_mean, M = self.gp_mean_calc(p, Y, return_mf = True)
        
        fig = plt.figure(figsize = (20, 5), **fig_kwargs)
        ax = fig.subplots(1, 4, sharey = sharey)

        titles = ["Data", "Mean function", "GP mean (excl. mean function)", "Residual noise"]
        data = [Y, M, gp_mean - M, Y - gp_mean]

        if self.dim == 1:
            ax[0].plot(x_plot[0], Y, label = "Data")
            ax[0].plot(x_plot[0], M, label = "Mean function")
            ax[0].plot(x_plot[0], gp_mean, label = "GP mean")
            ax[0].legend()

            for i in range(1, 4):
                ax[i].plot(x_plot[0], data[i])
                ax[i].set_xlabel(axes_labels[0])
                ax[i].set_title(titles[i])
        
        elif self.dim == 2:

            ims = [ax[i].pcolormesh(x_plot[1], x_plot[0], data[i], shading="nearest")
                   for i in range(4)]

            for i, (im, title) in enumerate(zip(ims, titles)):
                cbar = fig.colorbar(im, ax=ax[i], location="top", pad=0.02,
                                    fraction=0.03, aspect=40)
                cbar.ax.set_title(title, pad=4, fontsize = 14)

            # Label axes
            ax[0].set_ylabel(axes_labels[0], fontsize = 14)
            for i in range(4):
                ax[i].set_xlabel(axes_labels[1], fontsize = 14)

            # pcolormesh defaults to having the y-axis decrease with height which is weird so invert it
            plt.gca().invert_yaxis()

        else:
            raise Exception(f"Plotting of {self.dim}D data not yet implemented")

        plt.tight_layout()

        if return_fig:
            return fig
        elif return_fit:
            return gp_mean, M

    
    def plot_lightcurves(self, p, Y, sep = None, time_dim = 1, xlabel = r"Time", ylabel = "Values + offset", bin = None,
                         figsize = (12, 8), sharey = True, **fig_kwargs):
        r"""Plots each row of a 2D data set as a light curve, with the fit and the residuals.

        Plots three panels with the light curves offset vertically from each other: the data with the mean
        function (dashed) and the GP predictive mean (solid); the data and the GP predictive mean after
        subtracting the mean function; and the residuals after subtracting the GP predictive mean.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function and any
                parameters needed by the mean function.
            Y (JAXArray): Observed data, with shape ``(N_l, N_t)``.
            sep (Scalar, optional): The vertical offset between light curves. Defaults to four times the
                largest standard deviation of any light curve.
            time_dim (int, optional): Which dimension of the data is time (0 or 1). Defaults to 1.
            xlabel (str, optional): Label of the x-axis. Defaults to ``"Time"``.
            ylabel (str, optional): Label of the y-axis. Defaults to ``"Values + offset"``.
            bin (tuple, optional): Bin sizes ``(l_bin, t_bin)`` to bin the data along each dimension before
                plotting (see :func:`luas.convenience_fns.bin_data_2D`).
            figsize (tuple, optional): Figure size. Defaults to ``(12, 8)``.
            sharey (bool, optional): Whether the panels share a y-axis. Defaults to ``True``.
            **fig_kwargs: Passed to ``plt.subplots``.

        Returns:
            plt.Figure: The figure.

        """

        if self.dim != 2:
            raise Exception("plot_lightcurves only implemented for 2D data")

        gp_mean, M = self.gp_mean_calc(p, Y, return_mf=True)

        if bin is not None:
            _, _, Y = bin_data_2D(self.x[0], self.x[1], Y, bin[0], bin[1])
            _, _, M = bin_data_2D(self.x[0], self.x[1], M, bin[0], bin[1])
            x_bin0, x_bin1, gp_mean = bin_data_2D(self.x[0], self.x[1], gp_mean, bin[0], bin[1])
            x_t = [x_bin0, x_bin1][time_dim]
        else:
            x_t = self.x[time_dim]

        if time_dim == 0:
            gp_mean = gp_mean.T
            M = M.T
            Y = Y.T

        N_l = Y.shape[0]

        if sep is None:
            sep = 4 * Y.std(1).max()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = figsize, sharey = sharey, **fig_kwargs)
        for i in range(N_l):
            ax1.plot(x_t, Y[i, :] + np.arange(0, -sep*N_l, -sep)[i], 'o', color = "tab:blue", ms = 2)
            ax1.plot(x_t, M[i, :] + np.arange(0, -sep*N_l, -sep)[i], '--', color = "tab:orange")
            ax1.plot(x_t, gp_mean[i, :] + np.arange(0, -sep*N_l, -sep)[i], 'k-', ms = 3)

            ax2.plot(x_t, Y[i, :] - M[i, :] + np.arange(1, 1-sep*N_l, -sep)[i], 'o', color = "tab:blue", ms = 2)
            ax2.plot(x_t, gp_mean[i, :] - M[i, :] + np.arange(1, 1-sep*N_l, -sep)[i], 'k-')

            ax3.plot(x_t, Y[i, :] - gp_mean[i, :] + np.arange(1, 1-sep*N_l, -sep)[i], 'o', color = "tab:blue", ms = 2)
            ax3.plot(x_t, np.arange(1, 1-sep*N_l, -sep)[i]*np.ones_like(x_t), 'k-', ms = 3)
            
        ax1.set_xlabel(xlabel)
        ax2.set_xlabel(xlabel)
        ax3.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        plt.tight_layout()
        return fig
    
    
    def autocorrelate(
        self,
        p: PyTree,
        Y: JAXArray,
        max_sep_l: Optional[int] = None,
        max_sep_t: Optional[int] = None,
        include_gp_mean: Optional[bool] = True,
        mat: Optional[JAXArray] = None,
        plot: Optional[bool] = True,
        plot_kwargs: Optional[dict] = None,
        zero_centre: Optional[bool] = False,
        cov: Optional[bool] = False,
    ) -> Union[JAXArray, Tuple[JAXArray, plt.Figure]]:
        r"""Calculates (and plots) a quick, approximate 2D autocorrelation of the residuals using
        ``jax.scipy.signal.correlate2d``, to check if there is any correlation left in the residuals
        after subtracting the GP predictive mean.

        If ``include_gp_mean = False``, the autocorrelation of the data minus the mean function (without the
        GP predictive mean) is calculated instead, which can help choose which kernels to fit with.
        Alternatively any matrix ``mat`` can be given, in which case ``p`` and ``Y`` are not used.

        Note:
            This assumes the data is evenly spaced along both dimensions. It is also not an exact
            autocorrelation: the residuals are assumed to have zero mean, and all values are divided by the
            overall variance of the residuals rather than by the standard deviations of the specific residuals
            being multiplied together. This means some values can lie outside the interval [-1, 1], but it is
            fast and should be reasonably accurate except for widely separated parts of the data. For this
            reason, only separations up to half the size of the data are plotted by default.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function and any
                parameters needed by the mean function.
            Y (JAXArray): Observed data, with shape ``(N_l, N_t)``.
            max_sep_l (int, optional): The largest separation in rows to plot. Defaults to half the number of rows.
            max_sep_t (int, optional): The largest separation in columns to plot. Defaults to half the number of
                columns.
            include_gp_mean (bool, optional): Whether to subtract the GP predictive mean as well as the mean
                function from the data. Defaults to ``True``.
            mat (JAXArray, optional): A matrix to calculate the autocorrelation of instead of the residuals.
            plot (bool, optional): Whether to plot the autocorrelation. Defaults to ``True``.
            plot_kwargs (dict, optional): Passed to ``plt.imshow``, replacing the default plot settings.
            zero_centre (bool, optional): Whether to set the value at zero separation (which is always 1) to 0,
                which makes weak correlations at other separations easier to see. Defaults to ``False``.
            cov (bool, optional): If ``True``, return the autocovariance instead of the autocorrelation.
                Defaults to ``False``.

        Returns:
            JAXArray or (JAXArray, AxesImage): The autocorrelation matrix. If ``plot = True``, a tuple of the
            autocorrelation matrix and the image returned by ``plt.imshow``.

        """
        
        # Sets the maximum separation to visualise the autocorrelation for
        # Once separations are too large the autocorrelation becomes very noisy
        # as it is based off of very few values
        if mat is not None:
            # If a general matrix mat is given, defaults to autocorrelation of half its length
            # in each dimension
            if max_sep_l is None:
                max_sep_l = mat.shape[0]//2
            if max_sep_t is None:
                max_sep_t = mat.shape[1]//2
                
            # Set residuals matrix equal to mat
            res = mat
            
        else:
            # For a matrix of residuals will also default to half the length of the data in
            # each dimension
            if max_sep_l is None:
                max_sep_l = self.data_shape[0]//2
            if max_sep_t is None:
                max_sep_t = self.data_shape[1]//2
        
        
            if include_gp_mean:
                # Perform GP regression
                gp_mean = self.gp_mean_calc(p, Y)
            
                # Includes GP mean fit to data when subtracting from observed data
                res = Y - gp_mean
            else:
                M = self.mf(p, self.x)
                # Only calculates the observed data minus the deterministic mean function
                res = Y - M
            
        # Performs the autocorrelation of the res matrix
        auto_corr = jax.scipy.signal.correlate2d(res, res)
        
        # jax.scipy.signal.correlate2d will pad with zeros as necessary for the autocorrelation
        # which will artificially reduce the strength of correlation at large separations.
        # These lines autocorrelate a matrix of ones to divide auto_corr by so that a values
        # with more zeros used for padding will be divided through by lower numbers
        # This should normalise things correctly and ensures if the residuals are constant
        # then they will produce a correlation of ones everywhere.
        ones = jnp.ones_like(res)
        auto_corr_ones = jax.scipy.signal.correlate2d(ones, ones)
        auto_corr /= auto_corr_ones
        
        # Find the centre of the auto_corr 2D array
        n_l, n_t = auto_corr.shape
        auto_corr_centre = ((n_l-1)//2, (n_t-1)//2)
        
        if not cov:
            # Unless calculating the autocovariance we divide by the variance at zero separation
            auto_corr /= auto_corr[auto_corr_centre[0], auto_corr_centre[1]]
        
        if zero_centre:
            # Can be helpful to zero the centre (which will always show correlation = 1)
            # to help visualise weaker correlation at non-zero separations
            auto_corr = auto_corr.at[auto_corr_centre[0], auto_corr_centre[1]].set(0.)

        if plot:
            if mat is None and plot_kwargs is None:
                # Calculate the x and y axes values assuming equal separation of data points
                # It is assumed if giving values to plot_kwargs that this will be handled by the user
                
                # First check if x_l and x_t contain multiple rows in which case pick the top row
                if self.x.ndim > 1:
                    x_l_plot = self.x[0][0, :]
                else:
                    x_l_plot = self.x[0]

                if self.x.ndim > 1:
                    x_t_plot = self.x[1][0, :]
                else:
                    x_t_plot = self.x[1]

                # Calculates the average separation between points along each dimension
                l_step = x_l_plot.ptp()/(x_l_plot.size-1)
                t_step = x_t_plot.ptp()/(x_t_plot.size-1)

                # Calculate the maximum separations in each dimension in values given in x_l and x_t
                extent = [t_step*-(max_sep_t+0.5), t_step*(max_sep_t+0.5),
                            l_step*(max_sep_l+0.5), l_step*-(max_sep_l+0.5)]
            else:
                extent = None
            
            # Select the correct range of separations to plot
            l_plot_range = [auto_corr_centre[0]-max_sep_l, auto_corr_centre[0]+max_sep_l+1]
            t_plot_range = [auto_corr_centre[1]-max_sep_t, auto_corr_centre[1]+max_sep_t+1]
            
            if plot_kwargs:
                fig = plt.imshow(auto_corr[l_plot_range[0]:l_plot_range[1], t_plot_range[0]:t_plot_range[1]],
                            **plot_kwargs)
            else:
                fig = plt.imshow(auto_corr[l_plot_range[0]:l_plot_range[1], t_plot_range[0]:t_plot_range[1]],
                            aspect = 'auto', interpolation = "none", extent = extent)
            
            plt.xlabel(r"$\Delta$t")
            plt.ylabel(r"$\Delta$l")
            plt.colorbar()
            
            return auto_corr, fig

        else:
            # If not plotting then just returns the autocorrelation matrix
            return auto_corr
    


    def transform_fn(
        self,
        p: PyTree,
        Y: JAXArray,
        inverse = True,
        transpose = 0,
    ) -> Scalar:
        r"""Applies the transformation the kernel uses along all dimensions except its fast dimension to
        the residuals of the data.

        Some Kronecker kernel classes (``luas.LuasKernelND``, ``luas.LuasLasrachKernel``,
        ``luas.ExtendedLuasKernel``, ``luas.MultiLowRankKernel`` and ``luas.DoubleMultiLowRankKernel``) factorise
        the matrix square root of the covariance matrix as :math:`\mathbf{L} = \mathbf{T}\tilde{\mathbf{L}}`,
        where :math:`\mathbf{T}` acts along every dimension except the fast dimension and
        :math:`\tilde{\mathbf{L}}` acts along the fast dimension. This method subtracts the mean function from
        ``Y`` and applies :math:`\mathbf{T}^{-1}` (or :math:`\mathbf{T}` if ``inverse = False``) to the
        residuals. Useful for looking at the data in the basis the kernel uses.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.
            inverse (bool, optional): If ``True``, apply :math:`\mathbf{T}^{-1}`, otherwise apply
                :math:`\mathbf{T}`. Defaults to ``True``.
            transpose (int, optional): If ``1``, apply the transpose (:math:`\mathbf{T}^{-T}` or
                :math:`\mathbf{T}^T`) instead. Defaults to ``0``.

        Returns:
            JAXArray: The transformed residuals, with shape ``GP.data_shape``.

        """
        
        # Calculate the residuals after subtraction of the deterministic mean function
        R = Y - self.mf(p, self.x)
        self.kf_eval = self.build_kf(p, self.x)
        
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = {})
        
        if inverse:
            logL_val = self.kf_eval.inv_transform_fn(R, transpose = transpose)
        else:
            logL_val = self.kf_eval.transform_fn(R, transpose = transpose)
        
        return logL_val



    def apply_inverse(
        self,
        p: PyTree,
        R: JAXArray,
    ) -> Scalar:
        r"""Multiplies an array by the inverse of the covariance matrix,
        :math:`\mathbf{K}^{-1} \vec{r}`.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            R (JAXArray): The array to multiply, with shape ``GP.data_shape``.

        Returns:
            JAXArray: :math:`\mathbf{K}^{-1} \vec{r}`, with shape ``GP.data_shape``.

        """
        
        self.kf_eval = self.build_kf(p, self.x)
        
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, full = True, stored_values = {})
        K_inv_R = self.kf_eval.inverse(R)
        
        return K_inv_R

    
    def matmul(
        self,
        p: PyTree,
        R: JAXArray,
        full = True,
        **kwargs,
    ) -> Scalar:
        r"""Multiplies an array by the covariance matrix, :math:`\mathbf{K} \vec{r}`.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            R (JAXArray): The array to multiply, with shape ``GP.data_shape``.
            full (bool, optional): Whether to use the full covariance matrix. Defaults to ``True``.
            **kwargs: Passed to the kernel's ``matmul`` method, e.g. ``wn = False`` to leave out white noise.

        Returns:
            JAXArray: :math:`\mathbf{K} \vec{r}`, with shape ``GP.data_shape``.

        """
        
        self.kf_eval = self.build_kf(p, self.x)

        if len(self.x) == 1:
            K_R = self.kf_eval.matmul(self.x[0], self.x[0], R, full = full, **kwargs)
        else:
            K_R = self.kf_eval.matmul(self.x, self.x, R, full = full, **kwargs)
        
        return K_R
        

    def orthogonalise(self, p, A):
        r"""Removes the non-orthogonal part of a set of low-rank component vectors.

        Only available for ``luas.MultiLowRankKernel``. The columns of ``A`` are whitened by the noise
        covariance matrix along the dimension other than the fast dimension, orthogonalised using a
        Householder QR decomposition, and then transformed back.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            A (JAXArray): The component vectors, with one vector per column.

        Returns:
            JAXArray: The orthogonalised component vectors, with the same shape as ``A``.

        """
        if self.opt_name == "MultiLowRankKernel":
            self.kf_eval = self.build_kf(p, self.x)
            return self.kf_eval.make_orthogonal(A, self.x)
        else:
            raise Exception("Orthogonalising components only compatible with MultiLowRankKernel")

        
    def apply_matrix_sqrt(
        self,
        p: PyTree,
        R: JAXArray,
        transpose = 0,
    ) -> Scalar:
        r"""Multiplies an array by the matrix square root of the covariance matrix,
        :math:`\mathbf{L} \vec{r}`, where :math:`\mathbf{K} = \mathbf{L} \mathbf{L}^T`.

        If :math:`\vec{z}` is drawn from a standard normal distribution then :math:`\mathbf{L} \vec{z}` is a
        random sample from the GP (without the mean function).

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            R (JAXArray): The array to multiply, with shape ``GP.data_shape``.
            transpose (int, optional): If ``1``, multiply by :math:`\mathbf{L}^T` instead. Defaults to ``0``.

        Returns:
            JAXArray: :math:`\mathbf{L} \vec{r}` (or :math:`\mathbf{L}^T \vec{r}`), with shape ``GP.data_shape``.

        """
        
        self.kf_eval = self.build_kf(p, self.x)
        
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = {})
        L_R = self.kf_eval.matrix_sqrt(R, transpose = transpose)
        
        return L_R


    def apply_matrix_inv_sqrt(
        self,
        p: PyTree,
        R: JAXArray,
        transpose = 0,
    ) -> Scalar:
        r"""Multiplies an array by the inverse of the matrix square root of the covariance matrix,
        :math:`\mathbf{L}^{-1} \vec{r}`, where :math:`\mathbf{K} = \mathbf{L} \mathbf{L}^T`.

        Applied to the residuals of the data, this whitens them: if the GP model is correct the result should
        look like independent draws from a standard normal distribution.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            R (JAXArray): The array to multiply, with shape ``GP.data_shape``.
            transpose (int, optional): If ``1``, multiply by :math:`\mathbf{L}^{-T}` instead. Defaults to ``0``.

        Returns:
            JAXArray: :math:`\mathbf{L}^{-1} \vec{r}` (or :math:`\mathbf{L}^{-T} \vec{r}`), with shape
            ``GP.data_shape``.

        """
        
        self.kf_eval = self.build_kf(p, self.x)
        
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = {})
        L_inv_R = self.kf_eval.matrix_inv_sqrt(R, transpose = transpose)
        
        return L_inv_R

    
    
    def logL_hessianable(
        self,
        p: PyTree,
        Y: JAXArray,
    ) -> Scalar:
        r"""Computes the log likelihood in a way which gives accurate second derivatives.

        Returns the same value as :meth:`logL`, but uses a calculation which is more numerically stable for
        second derivatives such as ``jax.hessian``, at the cost of being slower for gradients. This method
        only returns the log likelihood, so ``jax.hessian`` must still be applied to it.

        This method is JIT compiled unless the GP was created with ``jit = False``.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.

        Returns:
            Scalar: The log likelihood.

        """
        
        # Subtract mean function from observed data
        R = Y - self.mf(p, self.x)
        
        # Calculate log likelihood and stored values from decomposition
        self.kf_eval = self.build_kf(p, self.x)
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = {})  
        logL_val = self.kf_eval.logL_hessianable(R, stored_values = stored_values)
        
        return logL_val

    
    def logL_hessianable_stored(
        self,
        p: PyTree,
        Y: JAXArray,
        stored_values: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        r"""Computes the log likelihood in a way which gives accurate second derivatives, and also
        returns values stored from the decomposition of the covariance matrix.

        The ``hessianable`` version of :meth:`logL_stored`. The input ``stored_values`` are passed on to
        the decomposition and reused by kernels which support this (``luas.LuasKernel`` created with
        ``use_stored_values = True``).

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.
            stored_values (PyTree): Values stored from a previous decomposition of the covariance
                matrix, such as the output of a previous call to this method. Their contents depend on
                the kernel class being used.

        Returns:
            (Scalar, PyTree): A tuple of the log likelihood and the values stored from the decomposition of
            the covariance matrix.

        """
        
        # Subtract mean function from observed data
        R = Y - self.mf(p, self.x)

        self.kf_eval = self.build_kf(p, self.x)
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = stored_values)  
        logL_val = self.kf_eval.logL_hessianable(R, stored_values = stored_values)
    
        return logL_val, stored_values
    
    
    def logP_hessianable(
        self,
        p: PyTree,
        Y: JAXArray,
    ) -> Scalar:
        r"""Computes the log posterior in a way which gives accurate second derivatives.

        Returns the same value as :meth:`logP`, but uses a calculation which is more numerically stable for
        second derivatives such as ``jax.hessian``, at the cost of being slower for gradients. Used by
        :meth:`laplace_approx`. This method only returns the log posterior, so ``jax.hessian`` must still be
        applied to it.

        This method is JIT compiled unless the GP was created with ``jit = False``.

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function. Also passed to the log prior function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.

        Returns:
            Scalar: The log posterior.

        """
        
        logPrior = self.logPrior(p, self.x)
        R = Y - self.mf(p, self.x)
        
        self.kf_eval = self.build_kf(p, self.x)
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = {})  
        logL_val = self.kf_eval.logL_hessianable(R, stored_values = stored_values)
        logP = logPrior + logL_val
        
        return logP

    
    def logP_hessianable_stored(
        self,
        p: PyTree,
        Y: JAXArray,
        stored_values: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        r"""Computes the log posterior in a way which gives accurate second derivatives, and also
        returns values stored from the decomposition of the covariance matrix.

        The ``hessianable`` version of :meth:`logP_stored`. The input ``stored_values`` are passed on to
        the decomposition and reused by kernels which support this (``luas.LuasKernel`` created with
        ``use_stored_values = True``).

        Args:
            p (PyTree): Parameters, including any hyperparameters needed by the kernel function
                and any parameters needed by the mean function. Also passed to the log prior function.
            Y (JAXArray): Observed data, with the same shape as the data set (``GP.data_shape``),
                e.g. ``(N_l, N_t)`` for 2D data.
            stored_values (PyTree): Values stored from a previous decomposition of the covariance
                matrix, such as the output of a previous call to this method. Their contents depend on
                the kernel class being used.

        Returns:
            (Scalar, PyTree): A tuple of the log posterior and the values stored from the decomposition of
            the covariance matrix.

        """

        
        R = Y - self.mf(p, self.x)

        self.kf_eval = self.build_kf(p, self.x)
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = stored_values)  
        logL_val = self.kf_eval.logL_hessianable(R, stored_values = stored_values)
        
        logPrior = self.logPrior(p, self.x)
        logP = logPrior + logL_val
        
        return logP, stored_values
