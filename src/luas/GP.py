import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import jax
import jax.numpy as jnp
from typing import Any, Optional, Callable, Union, Dict, Tuple

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
    """Gaussian process interface specialised to make the analysis of grid-based data sets simple and efficient.
    
    Must have separable input dimensions which are used to compute the covariance matrix and may additionally
    be used to compute the mean function and log prior. The observed data ``Y`` used for calculations is assumed to be a JAXArray
    matching the dimensions of the input regression tuple and input kernel function and mean function.

    Args:
        p (PyTree | dict): A PyTree of parameters which 
        X (JAXArray | tuple): For 1D this should be a JAXArray containing the regression variable(s).
            May be of shape ``(N,)`` or ``(d, N)`` for ``d`` different regression variables.
            For 2D or higher this should be a tuple of JAXArrays containing the regression variable(s) in each dimension.
            The order of the regression variables should match the shape of the observed data array.
        kf (Callable, optional): Function which returns the kernel function to use. Needs to be in the format ``kf(p, X, *kf_args, **kf_kwargs)``
            For 1D this should return a kernel or sum/product of kernels from luas.kernels directly.
            For 2D or higher, this should return a tuple or multiple tuples containing the kernel functions in each dimension,
            in the same order as X and the observed data array.
        mf (Callable, optional): The deterministic mean function, by default returns a JAXArray of zeros.
            Needs to be in the format ``mf(p, X, *mf_args, **mf_kwargs)`` and returns a JAXArray of the same shape
            as the observed data array.
        logPrior (Callable, optional): Log prior function, by default returns zero.
            Needs to be in the format ``logPrior(params, X, *logPrior_args, **logPrior_kwargs)`` and return a scalar.
        jit (bool, optional): Whether to ``jax.jit`` compile the likelihood, posterior and GP prediction
            functions. If set to ``False`` then mean functions not written in ``JAX`` are supported and
            can still be used with ``PyMC`` (but not ``NumPyro`` which requires JIT compilation).
            Defaults to ``True``.
        use_kernel (CovType, optional): Whether to use a specific GP optimisation, rather than the one automatically identified as fast by luas.
            For example, luas.GeneralKernel will force luas to avoid any optimisations and just build the full covariance matrix and use Cholesky decomposition.
            In some cases, multiple GP optimisations are valid and it may be worth testing which is fastest,
            such as luas.LuasKernel and luas.LuasLasrachKernel which should always be valid for the same kernel functions.
            Some kernels require certain keyword arguments to be specified, which can be input as any additional keyword arguments within kernel_kwargs.
        verbose (bool, optional): Print information about which optimisations are being used at initialisation.
        kf_args (tuple, optional): Any additional parameters to input to the kernel function ``kf(p, X, *kf_args, **kf_kwargs)``.
        kf_kwargs (dict, optional): Any keyword arguments to input to the kernel function ``kf(p, X, *kf_args, **kf_kwargs)``.
        mf_args (tuple, optional): Any additional parameters to input to the mean function ``mf(p, X, *mf_args, **mf_kwargs)``.
        mf_kwargs (dict, optional): Any keyword arguments to input to the mean function ``mf(p, X, *mf_args, **mf_kwargs)``.
        logPrior_args (tuple, optional): Any additional parameters to input to the logPrior function ``logPrior(p, X, *logPrior_args, **logPrior_kwargs)``.
        logPrior_kwargs (dict, optional): Any keyword arguments to input to the logPrior function ``logPrior(p, X, *logPrior_args, **logPrior_kwargs)``.
            
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


    def sample(self, p, add_mf = True):
        """Sample from the GP prior distribution. Useful for visualising a GP model and generating synthetic data.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
            add_mf (bool, optional): Whether to add the mean function to the GP draw, default is True.
        
        Returns:
            JAXArray: A random draw from the GP prior distribution.
            
        """

        self.kf_eval = self.build_kf(p, self.x)
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = {})

        # The Conditional CovType has a non-square matrix square root
        # which expects a random normal vector of a different shape to the output data
        z_shape = getattr(self.kf_eval, "sqrt_input_shape", self.data_shape)
        z = np.random.normal(size = z_shape)
        R = self.kf_eval.matrix_sqrt(z)

        if add_mf:
            R += self.mf(p, self.x)
        
        return R

    
    def logL(
        self,
        p: PyTree,
        Y: JAXArray,
    ) -> Scalar:
        """Computes the log likelihood without returning any stored values from the
        decomposition of the covariance matrix.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
        
        Returns:
            Scalar: The value of the log likelihood.
            
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
        """Computes the log likelihood and also returns any stored values from the decomposition of
        the covariance matrix. This allows time to be saved in future log likelihood calculations in
        which some hyperparameters are either fixed or being sampled separately with Gibbs/Blocked Gibbs
        sampling.
        
        Note:
            This function will not give correct second order derivatives/hessian values (e.g. calculated
            using `jax.hessian`). Make sure to use `GP.logL_hessianable_stored` if any hessian calculations
            are required.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
            stored_values (PyTree): Stored values from the decomposition of the covariance matrix. The specific
                values contained in this PyTree depend on the choice of :class:`Kernel` object and are returned by
                ``Kernel.decomp_fn``.
        
        Returns:
            (Scalar, PyTree): A tuple where the first element is the value of the log likelihood.
            The second element is a PyTree which contains stored values from the decomposition of the
            covariance matrix.
            
        """
        
        R = Y - self.mf(p,  self.x)
        
        self.kf_eval = self.build_kf(p, self.x)
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = {})  
        logL_val = self.kf_eval.logL(R, stored_values = stored_values)
    
        return logL_val, stored_values

    
    def logP(
        self,
        p: PyTree,
        Y: JAXArray,
    ) -> Scalar:
        """Computes the log posterior without returning any stored values from the
        decomposition of the covariance matrix.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
                Also input to the logPrior function for the calculation of the log priors.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
        
        Returns:
            Scalar: The value of the log posterior.
            
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
        """Computes the log posterior and also returns any stored values from the decomposition of the
        covariance matrix. This allows time to be saved in future log likelihood calculations in
        which some hyperparameters are either fixed or being sampled separately with Gibbs/Blocked Gibbs
        sampling.
        
        Note:
            This function will not give correct second order derivatives/hessian values (e.g. calculated
            using `jax.hessian`). Make sure to use `GP.logP_hessianable_stored` if any hessian calculations
            are required.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
                Also input to the logPrior function for the calculation of the log priors.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
            stored_values (PyTree): Stored values from the decomposition of the covariance matrix. The specific
                values contained in this PyTree depend on the choice of :class:`Kernel` object and are returned by
                ``Kernel.decomp_fn``.
        
        Returns:
            (Scalar, PyTree): A tuple where the first element is the value of the log posterior.
            The second element is a PyTree which contains stored values from the decomposition of the
            covariance matrix.
            
        """

        
        R = Y - self.mf(p,  self.x)
        
        self.kf_eval = self.build_kf(p, self.x)
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = {})  
        logL_val = self.kf_eval.logL(R, stored_values = stored_values)

        logPrior = self.logPrior(p, self.x)
        logP = logPrior + logL_val
        
        return logP, stored_values

    
    def crossval(self, p, Y,
                 train_l = None, train_t = None,
                 stack_p_dim1 = [], stack_p_dim2 = [],
                 return_predicted_loc = False,
                 **kwargs):

        x_l, x_t = self.x
    
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
    
        gp_mean_pred, M_pred, gp_cond_obj = self.predict(
            p_obs, Y,
            p_pred = p_pred,
            X_obs = (x_l, x_t),
            X_pred = (x_l_pred, x_t_pred),
            stack_p_dim1 = stack_p_dim1, stack_p_dim2 = stack_p_dim2,
            return_predicted_loc = return_predicted_loc,
            **kwargs,
        )
        return gp_mean_pred, M_pred, gp_cond_obj
    
    
    
    def predict(self, p, Y,
                p_pred = None,
                X_obs = None,
                X_pred = None,
                interp_params_dim1 = [], interp_params_dim2 = [],
                interp_row_l = None, interp_row_t = None,
                stack_p_dim1 = [], stack_p_dim2 = [], 
                sort_l = True, sort_t = True,
                fast_dim = None, return_predicted_loc = True):
    
        r"""Performs GP regression and computes the GP predictive mean and the GP predictive
        uncertainty as the standard devation at each location or else can return the full
        covariance matrix. Requires the input kernel function(s) to have a ``wn`` keyword
        argument that defines the kernel when white noise is included (``wn = True``) and
        when white noise isn't included (``wn = False``).
        
        Currently assumes the same input hyperparameters for both the observed and predicted
        locations. The predicted locations ``x_l_pred`` and ``x_t_pred`` may deviate from
        the observed locations ``x_l`` and ``x_t`` however.
        
        The GP predictive mean is defined as:
        
        .. math::
        
            \mathbb{E}[\vec{y}_*] = \vec{\mu}_* + \mathbf{K}_*^T \mathbf{K}^{-1} \vec{r}
        
        And the GP predictive covariance is given by:
        
        .. math::
        
            Var[\vec{y}_*] = \mathbf{K}_{**} - \mathbf{K}_*^T \mathbf{K}^{-1} \mathbf{K}_*
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
            x_l_pred (JAXArray, optional): Prediction locations along the row dimension,
                defaults to observed input locations.
            x_t_pred (JAXArray, optional): Prediction locations along the column dimension,
                defaults to observed input locations.
            return_std_dev (bool, optional): If ``True`` will return the standard deviation ofuncertainty
                at the predicted locations. Otherwise will return the full predictive covariance matrix.
                Defaults to ``True``.
        
        Returns:
            (JAXArray, JAXArray, JAXArray): Returns a tuple of three elements, where the first element is
            the GP predictive mean at the prediction locations, the second element is either the
            standard deviation of the predictions if ``return_std_dev = True``, otherwise it will be
            the full covariance matrix of the predicted values. The third element will be the mean function
            evalulated at the prediction locations.
        
        """
        p_obs = deepcopy(p)

        if X_obs is None:
            X_obs = deepcopy(self.x)
        
        if p_pred is None:
            p_pred = deepcopy(p_obs)
        
        pred_loc_equals_obs = False
        if X_pred is None:
            # If no prediction locations given then taken to be same as observed locations
            pred_loc_equals_obs = True
            
            if fast_dim is None or fast_dim == 0:
                X_pred = [X_obs[0], None]
            elif fast_dim == 1:
                X_pred = [None, X_obs[1]]
            else:
                raise Exception("fast_dim keyword argument must be None, 0 or 1")
    
        # Stack observed and predicted locations together for each dimension
        # Works out location of predicted and observed locations within array
        # Will sort the full stacked regression variables if requested, necessary for celerite dimensions
        x_l_full, X_pred[0], obs_points_l, pred_points_l, sort_fn_l = stack_obs_and_pred_points(X_obs[0], X_pred[0], sort = sort_l)
        x_t_full, X_pred[1], obs_points_t, pred_points_t, sort_fn_t = stack_obs_and_pred_points(X_obs[1], X_pred[1], sort = sort_t)

        X_full = (x_l_full, x_t_full)
        N_full = (X_full[0].shape[-1], X_full[1].shape[-1])
        
        observed_ind = jnp.ix_(jnp.arange(N_full[0])[obs_points_l],
                               jnp.arange(N_full[1])[obs_points_t])
    
        if return_predicted_loc:
            # Will only output locations at grid inputs of (X_pred)
            predicted_ind = jnp.ix_(jnp.arange(N_full[0])[pred_points_l],
                                    jnp.arange(N_full[1])[pred_points_t])
        else:
            # Returns the full output locations at X_full
            predicted_ind = jnp.ix_(jnp.arange(N_full[0]), jnp.arange(N_full[1]))
        
        p_pred = interp_params(p_obs, p_pred, X_obs[0], X_pred[0],
                               interp_param = interp_params_dim1,
                               interp_row = interp_row_l)
        p_pred = interp_params(p_obs, p_pred, X_obs[1], X_pred[1],
                               interp_param = interp_params_dim2,
                               interp_row = interp_row_t)
    
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
        K_inv_R_to_mult = jnp.zeros((N_full[0], N_full[1]))
        K_inv_R_to_mult = K_inv_R_to_mult.at[observed_ind].set(K_inv_R)

        # mf of conditional GP
        def gp_mean_cond_fn(p_pred, X_full):
            p_full = stack_params(p_obs, p_pred,
                                    stack_names_dim1 = stack_p_dim1, sort_fn_dim1 = sort_fn_l,
                                    stack_names_dim2 = stack_p_dim2, sort_fn_dim2 = sort_fn_t)

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
            p_full = stack_params(p_obs, p_pred, stack_names_dim1 = stack_p_dim1, sort_fn_dim1 = sort_fn_l,
                                    stack_names_dim2 = stack_p_dim2, sort_fn_dim2 = sort_fn_t)
            
            kf_eval_full = self.build_kf(p_full, X_full)
            return Conditional(kf_eval_full, kf_train = kf_eval_train, X_full = X_full,
                                     observed_ind = observed_ind, predicted_ind = predicted_ind,
                                     logdet_K_train = logdet_K_train)

        mf_cond = lambda p_pred, X_full: gp_mean_cond_fn(p_pred, X_full)[0]
        gp_cond = GP(p_pred, X_full, mf = mf_cond, kf = kf_cond)
        gp_cond.data_shape = (X_pred[0].shape[-1], X_pred[1].shape[-1])

        gp_mean_pred, M_pred = gp_mean_cond_fn(p_pred, X_full)
        
        return gp_mean_pred, M_pred, gp_cond
        

    def gp_mean_calc(
            self,
            p,
            Y,
            return_mf = False,
    ):
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
        r"""Computes the Laplace approximation at the location of ``p`` with options to regularise
        values which are poorly constrained. The parameters in ``p`` should be best-fit values of the posterior.
        
        The Laplace approximation is an estimate of the posterior distribution at the location of best-fit.
        It assumes the best-fit location is the mean of the Gaussian and calculates the covariance matrix
        based on approximating the value of the Hessian at the location of best-fit. By taking the negative
        inverse of the Hessian matrix this should give an approximate covariance matrix assuming the posterior
        is close to a Gaussian distribution. It is equivalent to a second-order Taylor series approximation of
        the posterior at the location of best-fit.
        
        The Laplace approximation is useful to get a quick approximation of the posterior without having to
        run an expensive MCMC calculation. Can also be useful for initialising MCMC inference with a good
        tuning matrix when large numbers of parameters which may contain strong correlations are being sampled.
        
        Note:
            This calculation can be memory intensive for large data sets with many free parameters and so
            setting ``large = True`` and ensuring ``large_block_size`` is a low integer can help reduce memory
            costs by breaking up the hessian calculation into blocks of rows.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
                Also input to the ``logPrior`` function for the calculation of the log priors.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
            regularise (bool, optional): Whether to add a regularisation constant to the diagonal of the hessian matrix
                corresponding to diagonals which are negative along the diagonals of the resulting covariance matrix.
                Defaults to ``True``.
            regularise_const (bool, optional): The constant added to diagonals of the hessian matrix to regularise it
                given that regularise is set to ``True``. Defaults to 100.
            vars (:obj:`list` of :obj:`str`, optional): The ``list`` of keys names corresponding to
                the parameters we want to calculate the Laplace approximation with respect to.
                The remaining parameters will be assumed to be fixed. If specified in addition to
                fixed_vars will raise an Exception.
            fixed_vars (:obj:`list` of :obj:`str`, optional): Alternative to vars, may specify instead
                the parameters being kept fixed which will not be marginalised over in the Laplace approximation.
                If specified in addition to vars will raise an Exception.
            return_array (bool, optional): Whether to return the approximated covariance matrix as a JAXArray or
                as a nested PyTree where e.g. the covariance between parameters named p1 and p2 is given by
                ``cov_mat[p1][p2]`` and ``cov_mat[p2][p1]``.
            large (bool, optional): Calculating the hessian matrix for large data sets with many parameters can be
                very memory intensive. If this is set to True then the hessian will be calculated in groups of rows
                instead of all at once which reduces the memory cost but can take significantly longer to run.
                The calculation is otherwise the same with no approximation made. Defaults to False.
            large_block_size (int, optional): If large is set to True and the hessian is being calculated in groups of rows
                can specify how many rows are being calculated simultaneously. Large numbers may calculate the overall hessian
                faster but at greater memory cost.
            large_jit (bool, optional): Whether to JIT compile the hessian function when ``large = True``,
                can speed up the calculation assuming the function can be JIT compiled. Defaults to ``True``.
            hessian_mat (JAXArray, optional): Instead of calculating the hessian matrix (needed for the Laplace approximation)
                from the input parameters ``p`` and ``Y`` just provide the hessian matrix directly.
                Assumed to be a JAXArray and not a PyTree. The input parameters ``p`` and ``Y`` will be ignored.
        
        Returns:
            (JAXArray, :obj:`list` of :obj:`str`) or (PyTree, :obj:`list` of :obj:`str`): Returns a tuple of two elements,
            if ``return_array = True`` the first element will be the covariance matrix from the Laplace approximation
            as a JAXArray, otherwise it will be as a nested PyTree. The second element will be the order of the parameters
            in the returned covariance matrix if it is a JAXArray. This list is also returned when ``return_array = False``
            for consistency. The order of the list matches how ``jax.flatten_util.ravel_pytree`` will order keys from a PyTree.
        
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
                # jax.hessian into a matrix which we can do with this helper function from luas.jax_convenience_fns
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
        """Computes the Laplace approximation at the location of ``p`` but within the transformed
        parameter space used by ``PyMC`` and ``NumPyro`` to deal with parameters bounded by a lower and upper bound.
        
        Example:
            ``param_bounds`` should be of the form ``param_bounds[par] = [lower_bound, upper_bound]`` where
            ``lower_bound`` and ``upper_bound`` are of the same shape as ``p[par]``.
        
        See ``GP.laplace_approx`` for more details about the Laplace approximation.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
                Also input to the ``logPrior`` function for the calculation of the log priors.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
            param_bounds (PyTree): Contains any bounds for the parameters in ``p``.
            vars (:obj:`list` of :obj:`str`, optional): The ``list`` of key names corresponding to
                the parameters we want to calculate the Laplace approximation with respect to.
                The remaining parameters will be assumed to be fixed. If specified in addition to
                fixed_vars will raise an Exception.
            fixed_vars (:obj:`list` of :obj:`str`, optional): Alternative to vars, may specify instead
                the parameters being kept fixed which will not be marginalised over in the Laplace approximation.
                If specified in addition to vars will raise an ``Exception``.
            large (bool, optional): Calculating the hessian matrix for large data sets with many parameters can be
                very memory intensive. If this is set to True then the hessian will be calculated in groups of rows
                instead of all at once which reduces the memory cost but can take significantly longer to run.
                The calculation is otherwise the same with no approximation made. Defaults to ``False``.
            large_block_size (int, optional): If large is set to True and the hessian is being calculated in groups of rows
                can specify how many rows are being calculated simultaneously. Large numbers may calculate the overall hessian
                faster but at greater memory cost.
            large_jit (bool, optional): Whether to JIT compile the hessian function when ``large = True``,
                can speed up the calculation assuming the function can be JIT compiled. Defaults to ``True``.
            return_array (bool, optional): Whether to return the approximated covariance matrix as a JAXArray or
                as a nested PyTree where e.g. the covariance between parameters named p1 and p2 is given by
                ``cov_mat[p1][p2]`` and ``cov_mat[p2][p1]``.
                
        Returns:
            (JAXArray, :obj:`list` of :obj:`str`) or (PyTree, :obj:`list` of :obj:`str`): Returns a tuple of two elements,
            if ``return_array = True`` the first element will be the covariance matrix from the Laplace approximation
            as a JAXArray, otherwise it will be as a nested PyTree. The second element will be the order of the parameters
            in the returned covariance matrix if it is a JAXArray. This list is also returned when ``return_array = False``
            for consistency. The order of the list matches how ``jax.flatten_util.ravel_pytree`` will order keys from a PyTree.
            
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
        r"""Visualise the covariance matrix/matrices generated by the input hyperparameters.
        
        Note:
            Default behaviour is to separately visualise each of the component covariance matrices in each dimension.
            For example, ``Sl``, ``St``, ``Kl`` and ``Kt`` for the 2D `LuasKernel` or `LuasLasrachKernel`.
            If ``full = True`` then it will instead build the full covariance matrix ``K``, but this is very
            memory intensive as it requires creating a JAXArray with ``(N_l*N_t, N_l*N_t)`` entries 
            (and is not currently implemented)! For 1D GPs this will simply generate the full covariance matrix.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
            corr (bool, optional): If ``True`` will plot the correlation between points instead of the
                covariance. Defaults to ``False``.
            wn (bool, optional): Whether to include white noise in the calculation of covariance.
                Defaults to ``True``.
            x_plot (JAXArray | tuple, optional): The values on the axes used by ``plt.pcolormesh`` for the plot.
                If not included, will default to the input vector(s) in ``X`` during GP object intialisation.
                If these input vectors consist of multiple regression vectors i.e. of shape ``(d, N)``, 
                then the first vector will be used.
            full (bool, optional): If ``True`` will build and visualise the full constructed covariance matrix
        
        Returns:
            plt.Figure: A figure displaying each of the component covariance matrices in the Kronecker product model.
        
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
        """Visualises the fit to the data. Displays the observed data as well as the mean function,
        the GP predictive mean (not including the mean function) and the residuals of the data
        after subtraction of the GP predictive mean (including the mean function).
        
        For a good fit to the data, the data minus the GP predictive mean should consist of
        white noise with no remaining correlations. The GP predictive mean (not including the mean function)
        should also just be fitting correlated noise and should not look like its fitting the mean function.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
            x_l_plot (JAXArray, optional): The values on the y-axis used by ``plt.pcolormesh`` for the plot.
                If not included will default to ``x_l`` if ``x_l`` is of shape ``(N_l,)`` or to ``x_l[0, :]``
                if ``x_l`` is of shape ``(d_l, N_l)``.
            x_t_plot (JAXArray, optional): The values on the x-axis used by ``plt.pcolormesh`` for the plot.
                If not included will default to ``x_t`` if ``x_t`` is of shape ``(N_t,)`` or to ``x_t[0, :]``
                if ``x_t`` is of shape ``(d_t, N_t)``.
        
        Returns:
            plt.Figure: The figure object containing the plot produced.
            
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
        """Quick function to visualise light curves and the residuals after subtraction of transit model
        """

        if self.dim != 2:
            raise Exception("plot_lightcurves only implemented for 2D data")

        gp_mean, M = self.gp_mean_calc(p, Y, return_mf=True)

        if bin is not None:
            _, _, Y = bin_data_2D(self.x[0], self.x[1], Y, bin[0], bin[1])
            _, _, M = bin_data_2D(self.x[0], self.x[1], M, bin[0], bin[1])
            x_bin0, x_bin1, gp_mean = bin_data_2D(X[0], X[1], gp_mean, bin[0], bin[1])
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
        """Performs a quick (and approximate) 2D autocorrelation using ``jax.scipy.signal.correlate2d``
        on the observed data after subtraction of the GP predictive mean to examine if there is any
        remaining correlation in the residuals.
        
        Note:
            This function also assumes the data is evenly spaced in both dimensions. It is also
            not an exact autocorrelation as the mean is not subtracted for each set of residuals and therefore
            it is assumed the residuals always have mean zero. Also instead of dividing by the standard deviations
            of the specific residuals being multiplied together, all values are divided by the overall variance
            of the residuals. This can result in some values having correlation lying outside the interval [-1, 1]
            but runs very efficiently and should be reasonably accurate unless considering correlations between
            widely separated parts of the data. For this reason, by default only half the separation of the data
            is visualised in the plots.
        
        If ``include_gp_mean = False`` then shows the autocorrelation of the observed data minus
        the mean function (without the GP predictive mean) which is useful for visualising
        what kernel function to use when fitting with a GP.
        
        Can also just input a general matrix ``mat`` to run an autocorrelation on, in which case the inputs
        ``p`` and ``Y`` are ignored.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
            max_sep_l (int, optional): The maximum separation of wavelengths/rows to visualise the correlation of.
                This is given as an integer referring to the number of rows apart to show. Defaults to half the
                number of rows in the observed data ``Y``.
            max_sep_t (int, optional): The maximum separation of time/columns to visualise the correlation of. This
                is given as an integer referring to the number of columns apart to show. Defaults to half the
                number of columns in the observed data ``Y``.
            include_gp_mean (bool, optional): Whether to subtract the GP predictive mean from the observed data
                when calculating the residuals. If ``False`` will still subtract the deterministic mean function.
                Useful for visualising correlation in a data set to aid in kernel choice before fitting with a GP.
            mat (JAXArray, optional): Instead of using the residuals of the observed data, providing a
                matrix for this argument will calculate the autocorrelation of this given matrix instead.
            plot (bool, optional): If ``True`` then will produce a plot visualising the autocorrelation.
                Defaults to ``True``.
            zero_centre (bool, optional): Whether to set the correlation at zero separation to 0.
                Since the correlation at zero separation is always 1 it can make it hard to visualise
                correlation values which are very small so setting this to ``True`` can aid visualisation.
                Defaults to ``False``.
            cov (bool, optional): If ``True`` will return the autocovariance matrix instead.
            
        Returns:
            JAXArray or (JAXArray, plt.Figure): Returns the autocorrelation matrix of the residuals. If
            ``plot = True`` then will return a tuple with the generated ``plt.Figure`` also added.
        
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
                max_sep_l = self.N_l//2
            if max_sep_t is None:
                max_sep_t = self.N_t//2
        
        
            if include_gp_mean:
                # Perform GP regression
                gp_mean, sigma_diag, M = self.predict(p, Y)
            
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
                if self.x_l.ndim > 1:
                    x_l_plot = self.x_l[:, 0]
                else:
                    x_l_plot = self.x_l

                if self.x_t.ndim > 1:
                    x_l_plot = self.x_t[:, 0]
                else:
                    x_l_plot = self.x_t

                # Calculates the average separation between points along each dimension
                l_step = x_l_plot.ptp()/(self.N_l-1)
                t_step = x_l_plot.ptp()/(self.N_t-1)

                
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
        """Computes the log likelihood without returning any stored values from the
        decomposition of the covariance matrix.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
        
        Returns:
            Scalar: The value of the log likelihood.
            
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
        
        self.kf_eval = self.build_kf(p, self.x)

        if len(self.x) == 1:
            K_R = self.kf_eval.matmul(self.x[0], self.x[0], R, full = full, **kwargs)
        else:
            K_R = self.kf_eval.matmul(self.x, self.x, R, full = full, **kwargs)
        
        return K_R
        

    def orthogonalise(self, p, A):
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
        """Computes the log likelihood without returning any stored values from the
        decomposition of the covariance matrix.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
        
        Returns:
            Scalar: The value of the log likelihood.
            
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
        """Computes the log likelihood without returning any stored values from the
        decomposition of the covariance matrix. This function is slower for gradient calculations
        than ``GP.logL`` but is more numerically stable for second-order derivative calculations as
        required when calculating the hessian. This function still only returns the log likelihood
        so ``jax.hessian`` must be applied to return the hessian of the log likelihood.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
        
        Returns:
            Scalar: The value of the log likelihood.
            
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
        """Computes the log likelihood and also returns any stored values from the
        decomposition of the covariance matrix. This function is slower for gradient calculations
        than ``GP.logL_stored`` but is more numerically stable for second-order derivative calculations as
        required when calculating the hessian. This function still only returns the log likelihood
        so jax.hessian must be applied to return the hessian of the log likelihood.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
            stored_values (PyTree): Stored values from the decomposition of the covariance matrix. The specific
                values contained in this PyTree depend on the choice of :class:`Kernel` object and are returned by
                ``Kernel.decomp_fn``.
        
        Returns:
            (Scalar, PyTree): A tuple where the first element is the value of the log likelihood.
            The second element is a PyTree which contains stored values from the decomposition of the
            covariance matrix.
            
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
        """Computes the log posterior without returning any stored values from the
        decomposition of the covariance matrix. This function is slower for gradient calculations
        than ``GP.logP`` but is more numerically stable for second-order derivative calculations as
        required when calculating the hessian. This function still only returns the log posterior
        so jax.hessian must be applied to return the hessian of the log posterior.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
                Also input to the logPrior function for the calculation of the log priors.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
        
        Returns:
            Scalar: The value of the log posterior.
            
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
        """Computes the log posterior and also returns any stored values from the decomposition of the
        covariance matrix.
        
        Note: 
            This function is slower for gradient calculations than ``GP.logP_stored`` but is more numerically
            stable for second-order derivative calculations as required when calculating the hessian.
            This function still only returns the log posterior so ``jax.hessian`` must be applied to return
            the hessian of the log posterior.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
                Also input to the logPrior function for the calculation of the log priors.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
            stored_values (PyTree): Stored values from the decomposition of the covariance matrix. The specific
                values contained in this PyTree depend on the choice of :class:`Kernel` object and are returned by
                ``Kernel.decomp_fn``.
        
        Returns:
            (Scalar, PyTree): A tuple where the first element is the value of the log posterior.
            The second element is a PyTree which contains stored values from the decomposition of the
            covariance matrix.
            
        """

        
        R = Y - self.mf(p, self.x)

        self.kf_eval = self.build_kf(p, self.x)
        self.kf_eval, stored_values = self.kf_eval.decompose(self.x, stored_values = stored_values)  
        logL_val = self.kf_eval.logL_hessianable(R, stored_values = stored_values)
        
        logPrior = self.logPrior(p, self.x)
        logP = logPrior + logL_val
        
        return logP, stored_values
