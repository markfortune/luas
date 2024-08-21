import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from .jax_convenience_fns import (
    array_to_pytree_2D,
    pytree_to_array_2D,
    order_list,
    large_hessian_calc,
    transf_from_unbounded_params,
    transf_to_unbounded_params,
    varying_params_wrapper,
)

from typing import Any, Optional, Callable, Union, Dict, Tuple
from .luas_types import Scalar, PyTree, JAXArray, Kernel

__all__ = ["GP"]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


class GP(object):
    """Gaussian process class specialised to make the analysis of 2D data sets simple and efficient.
    Can be used with any :class:`Kernel` such as :class:`LuasKernel` to perform
    the required log-likelihood calculations in addition to performing GP regression.
    Support for calculating Laplace approximations of the posterior with respect to
    the input parameters is also provided as it can be very useful when sampling large numbers of parameters
    (such as for tuning the MCMC).
    
    Must have two separate input dimensions which are used to compute the covariance matrix and may additionally
    be used to compute the deterministic mean function. The observed data ``Y`` is assumed to have shape ``(N_l, N_t)``
    and will be a parameter of most methods.
    
    The first input ``x_l`` is the wavelength/vertical dimension of the observed data ``Y`` and is expected to
    have shape ``(N_l,)`` or ``(d_l, N_l)`` where N_l is the number of rows of ``Y`` and ``d_l`` is the number
    of regression variables along the wavelength/vertical dimension used for kernel inputs and/or mean function inputs.
    Similarly ``x_t`` is assumed to lie along the time/horizontal dimension of the observed data with shape ``(N_t,)`` or
    ``(d_t, N_t)`` where ``N_t`` is the number of columns of ``Y`` and ``d_t`` is the number of regression variables
    along the column dimension used as kernel inputs and/or mean function inputs.

    Args:
        kf (Kernel): Kernel object which has already been initialised with the desired kernel function.
        x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s).
            May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l`` different wavelength/vertical
            regression variables.
        x_t (JAXArray): Array containing time/horizontal dimension regression variable(s).
            May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different time/horizontal
            regression variables.
        mf (Callable, optional): The deterministic mean function, by default returns a JAXArray of zeros.
            Needs to be in the format ``mf(params, x_l, x_t)`` and returns a JAXArray of shape ``(N_l, N_t)``.
            matching the shape of the observed data Y.
        logPrior (Callable, optional): Log prior function, by default returns zero.
            Needs to be in the format ``logPrior(params)`` and return a scalar.
        jit (bool, optional): Whether to ``jax.jit`` compile the likelihood, posterior and GP prediction
            functions. If set to ``False`` then mean functions not written in ``JAX`` are supported and
            can still be used with ``PyMC`` (but not ``NumPyro`` which requires JIT compilation).
            Defaults to ``True``.
            
    """
    
    def __init__(
        self,
        kf: Kernel,
        x_l: JAXArray,
        x_t: JAXArray,
        mf: Optional[Callable] = None,
        logPrior: Optional[Callable] = None,
        jit: Optional[bool] = True,
    ):
        # Initialise variables. Due to the way JAX's JIT compilation works,
        # any variables initialised here should not be modified but instead
        # a new GP object should be initialised.
        self.kf = kf
        self.x_l = x_l
        self.x_t = x_t
        
        # The accepted shapes for each dimension are (N_l,) and (d_l, N_l)
        # so take the length of the last dimension to get the dimension of the
        # wavelength and time dimensions
        self.N_l = self.x_l.shape[-1]
        self.N_t = self.x_t.shape[-1]
        
        if mf is None:
            # Mean function returns zeros by default
            # Returns a zero array which can vary in shape depending on the inputs
            # x_l and x_t which permits GP regression to predict unobserved points
            # of different array sizes
            self.mf = lambda p, x_l, x_t: jnp.zeros((x_l.shape[-1], x_t.shape[-1]))
        else:
            # Ensure mean function is of the form mf(p, x_l, x_t) and for the
            # observed inputs x_l, x_t should return an array of the same shape
            # as the observed data Y.
            self.mf = mf
            
            
        if logPrior is None:
            # Log Prior function returns zero by default
            self.logPrior = lambda p: 0.
        else:
            # Custom logPrior function which must take only a single argument p
            self.logPrior = logPrior
        
        if jit:
            # Have to option to avoid JIT compiling which can sometimes be useful
            # e.g. Using a mean function which is either in JAX but cannot be JIT compiled
            # Or a mean function which contains code not written in JAX
            # Note that NumPyro will perform JIT compilation anyway but PyMC will not
            self.logL = jax.jit(self.logL)
            self.logL_stored = jax.jit(self.logL_stored)
            self.logP = jax.jit(self.logP)
            self.logP_stored = jax.jit(self.logP_stored)
            self.logL_hessianable = jax.jit(self.logL_hessianable)
            self.logL_hessianable_stored = jax.jit(self.logL_hessianable_stored)
            self.logP_hessianable = jax.jit(self.logP_hessianable)
            self.logP_hessianable_stored = jax.jit(self.logP_hessianable_stored)
    
    
    def calculate_stored_values(self, p: PyTree) -> PyTree:
        """Calculate a PyTree of stored values from the decomposition of the covariance matrix.
        The values stored depend on the choice of Kernel object and are returned by its decomp_fn method.
        E.g. for :class:`LuasKernel` this will include eigenvalues and matrices calculated from the
        eigendecompositions of its component covariance matrices.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix.
                Any mean function parameters included will not affect this function.
        
        Returns:
            PyTree: Stored values from the decomposition of the covariance matrix. The specific
            values contained in this PyTree depend on the choice of Kernel object and are returned by the
            decomp_fn method of each :class:`Kernel` class.
            
        """
        
        return self.kf.decomp_fn(p, self.x_l, self.x_t, stored_values = {})
    
    
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
        R = Y - self.mf(p, self.x_l, self.x_t)
        
        # Use the specific log likelihood calculation of the chosen Kernel object
        # to compute the log likelihood and any stored values from the decomposition
        # are also returned by default but not returned by this method
        logL, stored_values = self.kf.logL(p, self.x_l, self.x_t, R, {})
        
        return logL

    
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
        
        R = Y - self.mf(p, self.x_l, self.x_t)
        logL, stored_values = self.kf.logL(p, self.x_l, self.x_t, R, stored_values)
    
        return logL, stored_values

    
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
        
        R = Y - self.mf(p, self.x_l, self.x_t)
        logL, stored_values = self.kf.logL(p, self.x_l, self.x_t, R, {})
        
        logPrior = self.logPrior(p)
        logP = logPrior + logL
        
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

        
        R = Y - self.mf(p, self.x_l, self.x_t)
        logL, stored_values = self.kf.logL(p, self.x_l, self.x_t, R, stored_values)
        
        logPrior = self.logPrior(p)
        logP = logPrior + logL
        
        return logP, stored_values

    
    def predict(
        self,
        p: PyTree,
        Y: JAXArray,
        x_l_pred: Optional[JAXArray] = None,
        x_t_pred: Optional[JAXArray] = None,
        return_std_dev: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[JAXArray, JAXArray, JAXArray]:
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
        
        # If no prediction locations specified, predict at observed locations
        if x_l_pred is None:
            x_l_pred = self.x_l
        if x_t_pred is None:
            x_t_pred = self.x_t

        # Generate mean function and compute residuals
        R = Y - self.mf(p, self.x_l, self.x_t)
        M_pred = self.mf(p, x_l_pred, x_t_pred)
        
        # Kernel object computes GP regression as the most efficient method depends on the form of
        # the kernel function
        gp_mean, pred_err = self.kf.predict(p, self.x_l, x_l_pred, self.x_t, x_t_pred, R, M_pred,
                                            return_std_dev = return_std_dev, **kwargs)
        
        return gp_mean, pred_err, M_pred
    
    
    def sigma_clip(
        self,
        p: PyTree,
        Y:JAXArray,
        sigma: Scalar,
        plot: Optional[bool] = True,
        use_gp_mean: Optional[bool] = True,
    ) -> JAXArray:
        """Performs GP regression and replaces any outliers above a given number of standard deviations
        with the GP predictive mean evaluated at those locations. If ``use_gp_mean = False`` then will instead
        replace outliers with the mean function evaluated at each location.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
            sigma (Scalar): Significance value in standard deviations above which outliers will be clipped.
            plot (bool, optional): Whether to produce plots which visualise the outliers in the data.
            use_gp_mean (bool, optional): Will replace outliers with values from the GP predictive mean if ``True``,
                otherwise will replace with values from the mean function.
        
        Returns:
            JAXArray: The observed data with outliers cleaned.
            
        """
        
        # First perform the GP regression with the predicted locations defaulting to the observed locations
        gp_mean, sigma_diag, M = self.predict(p, Y)
        
        # Residuals after subtraction of the GP mean are normalised based on their predicted uncertainties
        res = Y - gp_mean
        Z = jnp.abs(res/sigma_diag)
        
        # Identify outliers above given significance level
        outliers = Z > sigma
        
        # Create a copy of the observed data with outliers replaced with the GP predictive mean values
        Y_clean = jnp.array(Y.copy())
        
        if use_gp_mean:
            Y_clean = Y_clean.at[outliers].set(gp_mean[outliers])
        else:
            Y_clean = Y_clean.at[outliers].set(M[outliers])
        
        print("Number of outliers clipped = ", (outliers).sum())
        
        # Some convenient plots to visualise the locations of the outliers
        if plot:
            plt.title("Std. Dev. of Residuals")
            plt.imshow(Z, aspect = 'auto')
            plt.colorbar()
            plt.show()
            
            if outliers.sum() > 0:
                plt.title("Locations of Outliers Removed")
                plt.imshow(Y, aspect = 'auto')
                y, x = jnp.where(outliers)
                plt.scatter(x, y, color='red', marker='x')
                plt.show()

        return Y_clean
    
    
    def plot(
        self,
        p: PyTree,
        Y: JAXArray,
        x_l_plot = None, 
        x_t_plot = None,
        **kwargs,
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
        if x_l_plot is None:
            if self.x_l.ndim == 1:
                x_l_plot = self.x_l
            else:
                x_l_plot = self.x_l[0, :]
                
        if x_t_plot is None:
            if self.x_t.ndim == 1:
                x_t_plot = self.x_t
            else:
                x_t_plot = self.x_t[0, :]
    
        # Perform GP regression at the observed data locations
        gp_mean, gp_cov, M = self.predict(p, Y, **kwargs)
        
        fig = plt.figure(figsize = (20, 5))
        ax = fig.subplots(1, 4, sharey = True)
        
        # First plot is just the observed data
        ax[0].set_title("Data")
        ax[0].pcolormesh(x_t_plot, x_l_plot, Y, shading = "nearest")
        
        # Second plot is just the deterministic mean function
        ax[1].set_title("Mean function")
        ax[1].pcolormesh(x_t_plot, x_l_plot, M, shading = "nearest")
        
        # Third plot is the GP mean fit to the data without the mean function included
        ax[2].set_title("GP mean (excl. mean function)")
        ax[2].pcolormesh(x_t_plot, x_l_plot, gp_mean - M, shading = "nearest")
        
        # Final plot is the residuals of the observed data after subtraction of the GP
        # predictive mean (including the deterministic mean function)
        ax[3].set_title("Residual noise")
        ax[3].pcolormesh(x_t_plot, x_l_plot, Y - gp_mean, shading = "nearest")

        # Label axes
        ax[0].set_ylabel('x_l')
        for i in range(4):
            ax[i].set_xlabel('x_t')

        # pcolormesh defaults to having the y-axis decrease with height which is weird so invert it
        plt.gca().invert_yaxis()
        
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
                M = self.mf(p, self.x_l, self.x_t)
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
        R = Y - self.mf(p, self.x_l, self.x_t)
        
        # Calculate log likelihood and stored values from decomposition
        logL, stored_values = self.kf.logL_hessianable(p, self.x_l, self.x_t, R, {})
        
        return logL

    
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
        R = Y - self.mf(p, self.x_l, self.x_t)
        
        # Calculate log likelihood and stored values from decomposition
        logL, stored_values = self.kf.logL_hessianable(p, self.x_l, self.x_t, R, {})
    
        return logL, stored_values
    
    
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
        
        logPrior = self.logPrior(p)
        R = Y - self.mf(p, self.x_l, self.x_t)
        logL, stored_values = self.kf.logL_hessianable(p, self.x_l, self.x_t, R, {})
        logP = logPrior + logL
        
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

        
        R = Y - self.mf(p, self.x_l, self.x_t)
        logL, stored_values = self.kf.logL_hessianable(p, self.x_l, self.x_t, R, stored_values)
        
        logPrior = self.logPrior(p)
        logP = logPrior + logL
        
        return logP, stored_values
    

    
    def laplace_approx(
        self,
        p: PyTree,
        Y: PyTree,
        regularise: Optional[bool] = True,
        regularise_const: Optional[Scalar] = 100.,
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
        
        # Performs the actual Laplace approximation by inverting the negative hessian
        cov_mat = jnp.linalg.inv(-hessian_mat)
        
        if regularise:
            # Test if the diagonals of the covariance matrix are positive
            cov_diag = jnp.diag(cov_mat)
            neg_ind = cov_diag < 0.
            num_neg_diag_vals = neg_ind.sum()
            
            if num_neg_diag_vals == 0:
                print("No regularisation needed to remove negative values along diagonal of covariance matrix.")
            else:
                # Subtract regularise_const from the diagonal hessian elements which correspond to negative
                # values in the covariance matrix which will help to regularise for large enough regularise_const
                regularise_vec = regularise_const*neg_ind
                hessian_mat -= jnp.diag(regularise_vec)

                # Calculate the new Laplace approximation with regularisation
                cov_mat = jnp.linalg.inv(-hessian_mat)

                # Help to describe which values were regularised
                # Identifies which parameters the negative diagonal elements correspond to
                p_arr, make_p_dict = ravel_pytree(p_vary)
                regularised_values = make_p_dict(neg_ind)

                # Only include values which needed to be regularised when printing
                for par in p_vary.keys():
                    if not jnp.any(regularised_values[par]):
                        del regularised_values[par]

                # Check if there are any remaining negative values along the diagonal of the covariance matrix
                cov_diag = jnp.diag(cov_mat)
                neg_ind = cov_diag < 0.
                num_neg_diag_vals_remaining = neg_ind.sum()
                
                if num_neg_diag_vals_remaining > 0:
                    # Identify which parameters are still resulting in negatives along the diagonal
                    values_still_negative = make_p_dict(neg_ind)
                    
                    # Only include values which still need to be regularised when printing
                    for par in p_vary.keys():
                        if not jnp.any(values_still_negative[par]):
                            del values_still_negative[par]
                            
                    print(f"Initial number of negative values on diagonal of covariance matrix = {num_neg_diag_vals}\n" \
                          f"Corresponding to parameters: {regularised_values}.\n" \
                          f"Remaining number of negative values = {num_neg_diag_vals_remaining}\n" \
                          f"Corresponding to parameters: {values_still_negative}.\n"
                          f"Try increasing regularise_const to ensure the covariance matrix is positive definite " \
                          f"or double check that the input parameters are close to a best-fit location."
                         )
                else:
                    print(f"Initial number of negative values on diagonal of covariance matrix = {num_neg_diag_vals}\n" \
                          f"Corresponding to parameters: {regularised_values}.\n" \
                          f"No remaining negative values."
                         )
        
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
                # Add to the diagonal of the hessian an additional term
                # which is equal to the hessian of the jacobian of the transformation
                # performed by PyMC and NumPyro
                # This term is added to ensure the transformation these inference libraries perform
                # does not impact the choice of priors made
                exp_minus_p = jnp.exp(-p_transf[par])
                hessian_of_transform_jacobian = jnp.diag(-2*exp_minus_p/(1+exp_minus_p)**2)
                hessian_mat[par][par] += hessian_of_transform_jacobian

        # Convert hessian from a nested PyTree to a 2D JAXArray for GP.laplace_approx to be able
        # to invert to calculate the covariance matrix
        hessian_mat = pytree_to_array_2D(p_transf, hessian_mat)
        cov_mat, ordered_param_list = self.laplace_approx(p_transf, Y, hessian_mat = hessian_mat,
                                                          return_array = return_array, **kwargs)
            
        return cov_mat, ordered_param_list
