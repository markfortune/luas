import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import jax
import jax.numpy as jnp
import jax.flatten_util
from jax import jit, grad, hessian
from kronecker_fn import *

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


def default_param_transform(p_vary, p_fixed, log_params):
    
    # Copy to avoid transformation affecting stored values
    p = deepcopy(p_fixed)
    
    # Update fixed values with values being varied
    p.update(p_vary)

    # Example of a polynomial parameterisation of a parameter being transformed to the polynomial
    # p["h_CM"] = jnp.polyval(p["h_CM"], x = jnp.linspace(-1, 1, self.N_l))

    # Transform log parameters
    for name in log_params:
        p[name] = jnp.power(10, p[name])

    return p


def transform_wrapper(p, transf_args, transf_fn, compute_fn, *args, **kwargs):
    # Wrapper function which creates functions with parameter transformation performed first
    
    p = transf_fn(p, *transf_args)
    
    return compute_fn(p, *args, **kwargs)


def transform_wrapper_hessian(p, transf_args, transf_fn, compute_fn, *args, **kwargs):
    # Wrapper function needed for the hessian calculation
    
    p["p1"] = transf_fn(p["p1"], *transf_args)
    p["p2"] = transf_fn(p["p2"], *transf_args)
    
    return compute_fn(p, *args, **kwargs)


def logL(p, x_l, x_t, Y, eigen_dict, mf):
    # Computes the log-likelihood 
        
    # Generate mean function and compute residuals
    M = mf(p, x_l, x_t)
    R = Y - M

    # Compute r.T K^-1 r
    alpha1 = kron_prod(eigen_dict["W_l"].T, eigen_dict["W_t"].T, R)
    alpha2 = jnp.multiply(eigen_dict["D_inv"], alpha1)
    r_K_inv_r = jnp.multiply(alpha1, alpha2).sum()

    # Can make use of stored logdetK from eigendecomposition
    logL = - 0.5*r_K_inv_r  - 0.5*eigen_dict["logdetK"] - 0.5*R.size*jnp.log(2*jnp.pi)

    return logL



def grad_logL(p, x_l, x_t, Y, eigen_dict, mf, kf):
    
    # Generate mean function and compute residuals
    M = mf(p, x_l, x_t)
    R = Y - M
    
    # Compute outer part of r^T K^-1 r derivative
    alpha1 = kron_prod(eigen_dict["W_l"].T, eigen_dict["W_t"].T, R)
    alpha2 = jnp.multiply(eigen_dict["D_inv"], alpha1)
    
    # Build kernels, necessary to do here for JAX to know how to calculate gradients
    Kl = kf.Kl(p, x_l, x_l)
    Kt = kf.Kt(p, x_t, x_t)
    Sl = kf.Sl(p, x_l, x_l)
    St = kf.St(p, x_t, x_t)
    
    # This transformation is used for both the r^T K^-1 r and logdetK derivatives
    W_Kl_W = eigen_dict["W_l"].T @ Kl @ eigen_dict["W_l"]
    W_Kt_W = eigen_dict["W_t"].T @ Kt @ eigen_dict["W_t"]
    W_Sl_W = eigen_dict["W_l"].T @ Sl @ eigen_dict["W_l"]
    W_St_W = eigen_dict["W_t"].T @ St @ eigen_dict["W_t"]
    
    K_alpha = kron_prod(W_Kl_W, W_Kt_W, alpha2)
    K_alpha += kron_prod(W_Sl_W, W_St_W, alpha2)

    # Diagonal of these terms is used for logdetK transformation
    Kl_diag = jnp.diag(W_Kl_W)
    Kt_diag = jnp.diag(W_Kt_W)
    Sl_diag = jnp.diag(W_Sl_W)
    St_diag = jnp.diag(W_St_W)
    
    # Computes diagonal of W.T K W for calculation of logdetK
    W_K_W_diag = jnp.outer(Kl_diag, Kt_diag) + jnp.outer(Sl_diag, St_diag)
    logdetK = jnp.multiply(eigen_dict["D_inv"], W_K_W_diag).sum()
    
    return + 0.5 * jnp.multiply(alpha2, K_alpha).sum() - 0.5 * logdetK
  
    
def hessian_logL(p, mfp, x_l, x_t, Y, eigen_dict, mf, kf):
    
    # 
    p1 = p["p1"]
    p2 = p["p2"]
    
    # Generate mean function and compute residuals
    M = mf(mfp, x_l, x_t)
    R = Y - M
    
    alpha1 = kron_prod(eigen_dict["W_l"].T, eigen_dict["W_t"].T, R)
    alpha2 = jnp.multiply(eigen_dict["D_inv"], alpha1)
    
    Kl1 = kf.Kl(p1, x_l, x_l)
    Kt1 = kf.Kt(p1, x_t, x_t)
    Sl1 = kf.Sl(p1, x_l, x_l)
    St1 = kf.St(p1, x_t, x_t)
    
    W_Kl_W1 = eigen_dict["W_l"].T @ Kl1 @ eigen_dict["W_l"]
    W_Kt_W1 = eigen_dict["W_t"].T @ Kt1 @ eigen_dict["W_t"]
    W_Sl_W1 = eigen_dict["W_l"].T @ Sl1 @ eigen_dict["W_l"]
    W_St_W1 = eigen_dict["W_t"].T @ St1 @ eigen_dict["W_t"]
    
    Kl2 = kf.Kl(p2, x_l, x_l)
    Kt2 = kf.Kt(p2, x_t, x_t)
    Sl2 = kf.Sl(p2, x_l, x_l)
    St2 = kf.St(p2, x_t, x_t)
    
    W_Kl_W2 = eigen_dict["W_l"].T @ Kl2 @ eigen_dict["W_l"]
    W_Kt_W2 = eigen_dict["W_t"].T @ Kt2 @ eigen_dict["W_t"]
    W_Sl_W2 = eigen_dict["W_l"].T @ Sl2 @ eigen_dict["W_l"]
    W_St_W2 = eigen_dict["W_t"].T @ St2 @ eigen_dict["W_t"]
    
    K_alpha1 = kron_prod(W_Kl_W1, W_Kt_W1, alpha2)
    K_alpha1 += kron_prod(W_Sl_W1, W_St_W1, alpha2)
    
    D_K_alpha1 = jnp.multiply(eigen_dict["D_inv"], K_alpha1)
    
    
    K_alpha2 = kron_prod(W_Kl_W2, W_Kt_W2, alpha2)
    K_alpha2 += kron_prod(W_Sl_W2, W_St_W2, alpha2)
    
    rKr = jnp.multiply(K_alpha2, D_K_alpha1).sum()
    
    K_diag = kron_prod(W_Kl_W1 * W_Kl_W2.T, W_Kt_W1 * W_Kt_W2.T, eigen_dict["D_inv"])
    K_diag += kron_prod(W_Sl_W1 * W_Sl_W2.T, W_St_W1 * W_St_W2.T, eigen_dict["D_inv"])
    K_diag += kron_prod(W_Kl_W1 * W_Sl_W2.T, W_Kt_W1 * W_St_W2.T, eigen_dict["D_inv"])
    K_diag += kron_prod(W_Sl_W1 * W_Kl_W2.T, W_St_W1 * W_Kt_W2.T, eigen_dict["D_inv"])
    
    K_logdet = jnp.multiply(eigen_dict["D_inv"], K_diag).sum()
    
    return - rKr + 0.5 * K_logdet
    
    
# def grad_logL(p, x_l, x_t, Y, eigen_dict, mf, kf):
    
#     # Generate mean function and compute residuals
#     M = mf(p, x_l, x_t)
#     R = Y - M
    
#     # Compute outer part of r^T K^-1 r derivative
#     alpha1 = kron_prod(eigen_dict["W_l"].T, eigen_dict["W_t"].T, R)
#     alpha2 = jnp.multiply(eigen_dict["D_inv"], alpha1)
    
#     # Build kernels, necessary to do here for JAX to know how to calculate gradients
#     Kl = kf.Kl(p, x_l, x_l)
#     Kt = kf.Kt(p, x_t, x_t)
#     Sl = kf.Sl(p, x_l, x_l)
#     St = kf.St(p, x_t, x_t)
    
#     # This transformation is used for both the r^T K^-1 r and logdetK derivatives
#     W_Kl_W = eigen_dict["W_l"].T @ Kl @ eigen_dict["W_l"]
#     W_Kt_W = eigen_dict["W_t"].T @ Kt @ eigen_dict["W_t"]
#     W_Sl_W = eigen_dict["W_l"].T @ Sl @ eigen_dict["W_l"]
#     W_St_W = eigen_dict["W_t"].T @ St @ eigen_dict["W_t"]
    
#     K_alpha = kron_prod(W_Kl_W, W_Kt_W, alpha2)
#     K_alpha += kron_prod(W_Sl_W, W_St_W, alpha2)

#     # Diagonal of these terms is used for logdetK transformation
#     Kl_diag = jnp.diag(W_Kl_W)
#     Kt_diag = jnp.diag(W_Kt_W)
#     Sl_diag = jnp.diag(W_Sl_W)
#     St_diag = jnp.diag(W_St_W)
    
#     # Computes diagonal of W.T K W for calculation of logdetK
#     W_K_W_diag = jnp.outer(Kl_diag, Kt_diag) + jnp.outer(Sl_diag, St_diag)
#     logdetK = jnp.multiply(eigen_dict["D_inv"], W_K_W_diag).sum()
    
#     return  + 0.5 * eigen_dict["logdetK"] + 0.5 * eigen_dict["grad_logdetK"] + 0.5*R.size*jnp.log(2*jnp.pi) + 0.5 * jnp.multiply(alpha2, K_alpha).sum() - 0.5 * logdetK
    

class GP(object):
    def __init__(self, initial_params, mfp_to_fit, hp_to_fit, x_l, x_t, Y, kf, mf = None, log_prior_fn = None,
                 eigen_fn = eigendecomp_rakitsch_general, log_params = [],
                 transform_fn = default_param_transform, transform_args = None):
        
        """
        GP class using the Kronecker product structure of the covariance matrix to extend GPs to
        2D regular structures relatively cheaply.

                   --------------------------------
                  |                                |
                  |                                |
                  |                                |
        x_l (N_l) |       Y (N_l x N_t) data       |
        input     |                                |
                  |                                |
                  |                                |
                   --------------------------------  
                            x_t (N_t) input

        Must be two input dimensions, and each are treated independently to create a kernel in
        each dimension. These are then combined with the sum of two Kronecker products to produce
        the full covariance matrix, although using eigendecomposition allows us to avoid computing and
        inverting huge matrices.

        Kl and Sl (both MxM) will be the covariance matrices in the wavelength/vertical direction,
        Kt and St (both NxN) will be the covariance matrices in the time/horizontal direction.
        The full covariance is given by K = (Kl KRON Kt + Sl KRON St).

        :inputs
        -------
        
        initial_params - dictionary of starting guesses for mean function parameters and hyperparameters
        mfp_to_fit - a list of the names of mean function parameters which are being fit (i.e. not fixed)
        hp_to_fit - a list of the names of hyperparameters which are being fit (i.e. not fixed)
        x_l,x_t - arrays containing the two input variables: np.array([x1,x2]).T
        Y - 2D array containing the observed data
        kf - Kernel object containing functions to calculate Kl, Kt, Sl, St as methods
        mf - mean function, by default returns zeros. Needs to be in the format mf(p, x_l, x_t)
        logPrior - log prior function, optional as PyMC can also be used for priors but more complex priors
            can be used with this
        eigen_fn - function used for calculating eigendecompositions of covariance matrices. Default will
            work for any general covariance matrix formed from K = (Kl KRON Kt + Sl KRON St).
        log_params - list of variable names which it is desired to fit for the log of the parameter (uses log base 10)
        transform_fn - function for transforming variables being fit to variable values. Where log parameters are transformed
        transform_args - arguments to transform_fn. If using default transform_fn then this can be left as None

        """
        
        # Initialise variables
        self.p_initial = initial_params
        self.p = {k:initial_params[k] for k in (mfp_to_fit + hp_to_fit)}
        self.x_l = x_l
        self.x_t = x_t
        self.Y = Y
        self.mfp = mfp_to_fit
        self.hp = hp_to_fit
        self.kf = kf
        self.log_params = log_params
        self.eigen_dict = {}
        self.transform_fn = transform_fn
        self.N_l = self.Y.shape[0]
        self.N_t = self.Y.shape[1]
        
        # Flatten parameter dict into an array to provide log-likelihood functions which
        # take array inputs which is required for some optimisers and samplers including by PyMC
        # Also returns a function to convert an array back into a dictionary
        self.p_arr, self.make_p_dict = jax.flatten_util.ravel_pytree(self.p)
        
        # Mean function returns zeros by default
        if mf is None:
            self.mf = lambda p, x_l, x_t: jnp.zeros((self.N_l, self.N_t))
            
        # Log Prior function returns zero by default
        if log_prior_fn is None:
            self.log_prior_fn = lambda p: 0.
            
        # Use default transform_fn arguments if not specified
        if transform_args is None:
            transform_args = (self.p_initial, self.log_params)
           
        # Convenient function for transforming untransformed parameter dict into a transformed dict
        self.transf_p = lambda p: self.transform_fn(p, *transform_args)
        
        # Wrap some functions with the parameter transformation function
        self.eigen_fn = lambda p, *args, **kwargs: transform_wrapper(p, transform_args, self.transform_fn,
                                                                     eigen_fn, *args, **kwargs)
        logL_transf = lambda p, *args, **kwargs: transform_wrapper(p, transform_args, self.transform_fn,
                                                                   logL, *args, **kwargs)
        grad_logL_transf = lambda p, *args, **kwargs: transform_wrapper(p, transform_args, self.transform_fn,
                                                                        grad_logL, *args, **kwargs)
        hessian_logL_transf = lambda p, *args, **kwargs: transform_wrapper_hessian(p, transform_args, self.transform_fn,
                                                                                   hessian_logL, *args, **kwargs)
        
        # JIT compile log-likelihood function
#         self.compute_logL = jit(logL_transf, static_argnums=(5,))
        self.compute_logL = jit(logL_transf, static_argnums=(5,6,))
        
        # Use JAX's grad and hessian functions on the log-likelihood in combination with JIT compilation
        self.compute_grad_logL = jit(grad(grad_logL_transf), static_argnums=(5,6,))
        self.compute_hessian_logL1 = jit(hessian(grad_logL_transf), static_argnums=(5,6,))
        self.compute_hessian_logL2 = jit(hessian(hessian_logL_transf), static_argnums=(6, 7,))
        
    
    def logL(self, p):
        
        # Calculate any necessary eigendecompositions
        self.eigen_dict = self.eigen_fn(p, self.x_l, self.x_t, self.kf, eigen_dict = self.eigen_dict)
        
        # Calculate log-likelihood
        return self.compute_logL(p, self.x_l, self.x_t, self.Y, self.eigen_dict, self.mf)
    
    
    def logP(self, p):
        
        logPrior = self.log_prior_fn(p)
        
        if logPrior == -jnp.inf:
            return -jnp.inf
        else:
            # Calculate any necessary eigendecompositions
            self.eigen_dict = self.eigen_fn(p, self.x_l, self.x_t, self.kf, eigen_dict = self.eigen_dict)

            # Calculate log-likelihood
            logL = self.compute_logL(p, self.x_l, self.x_t, self.Y, self.eigen_dict, self.mf)
        
            return logPrior + logL
    
    
    def grad_logP(self, p):
        
        logPrior = self.log_prior_fn(p)
        
        if logPrior == -jnp.inf:
            return -jnp.inf
        else:
            # Calculate any necessary eigendecompositions
            self.eigen_dict = self.eigen_fn(p, self.x_l, self.x_t, self.kf, eigen_dict = self.eigen_dict)

            # Calculate dictionary of gradients of log-likelihood
            grad_dict = self.compute_grad_logL(p, self.x_l, self.x_t, self.Y, self.eigen_dict, self.mf, self.kf)

            # Function used for gradient calculations actually returns negative gradients of mean function parameters
            grad_dict.update({k: -grad_dict[k] for k in self.mfp})

            return grad_dict
    
    
    def grad_logL(self, p):
        
        # Calculate any necessary eigendecompositions
        self.eigen_dict = self.eigen_fn(p, self.x_l, self.x_t, self.kf, eigen_dict = self.eigen_dict)
        
        # Calculate dictionary of gradients of log-likelihood
        grad_dict = self.compute_grad_logL(p, self.x_l, self.x_t, self.Y, self.eigen_dict, self.mf, self.kf)
        
        # Function used for gradient calculations actually returns negative gradients of mean function parameters
        grad_dict.update({k: -grad_dict[k] for k in self.mfp})
        
        return grad_dict
    
    
    
    def hessian_logL(self, p):
        
        # Calculate any necessary eigendecompositions
        self.eigen_dict = self.eigen_fn(p, self.x_l, self.x_t, self.kf, eigen_dict = self.eigen_dict)
        
        # Calculate nested dictionary of hessian of log-likelihood
        hessian_dict = self.compute_hessian_logL1(p, self.x_l, self.x_t, self.Y, deepcopy(self.eigen_dict), self.mf, self.kf)
        
        p_transf = self.transf_p(p)
        mfp = {k:p_transf[k] for k in self.mf.mfp}
        hp = {k:p[k] for k in self.hp}
        p_cross = {"p1":hp, "p2":deepcopy(hp)}
        hessian_dict2 = self.compute_hessian_logL2(p_cross, mfp, self.x_l, self.x_t, self.Y, self.eigen_dict, self.mf, self.kf)
        
        for i in self.mfp:
            for j in self.mfp:
                hessian_dict[i][j] = -hessian_dict[i][j]
                
        for i in self.hp:
            for j in self.hp:
                hessian_dict[i][j] += hessian_dict2["p1"][i]["p2"][j]
        
        return hessian_dict
    
    
    def hessian_to_covariance_mat(self, hessian_dict):
        
        self.cov_order = {}

        i = 0
        for param in hessian_dict.keys():
            if type(self.p_initial[param]) in [float, np.float32, np.float64, jnp.float32, jnp.float64]:
                self.cov_order[param] = np.arange(i, i+1)
                i += 1
            else:
                param_size = self.p_initial[param].size
                self.cov_order[param] = np.arange(i, i+param_size)
                i += param_size
        
        hess_mat = np.zeros((i, i))

        for (k1, v1) in self.cov_order.items():
            for (k2, v2) in self.cov_order.items():
                if v1.size > 1 and v2.size == 1:
                    hess_mat[np.ix_(v1, v2)] = hessian_dict[k1][k2].reshape((v1.size, v2.size))
                else:
                    hess_mat[np.ix_(v1, v2)] = hessian_dict[k1][k2]

        hess_mat = 0.5 * (hess_mat + hess_mat.T)
        
        self.cov_mat = jnp.linalg.inv(-hess_mat)
        
        self.cov_dict = {p:{} for p in hessian_dict.keys()}
        
        for (k1, v1) in self.cov_order.items():
            for (k2, v2) in self.cov_order.items():
                self.cov_dict[k1][k2] = self.cov_mat[np.ix_(v1, v2)]
        
        return self.cov_dict
    
    
    def get_cov(self, p_list):
        
        order_dict = {}

        i = 0
        for param in p_list:
            if type(self.p_initial[param]) in [float, np.float32, np.float64, jnp.float32, jnp.float64]:
                order_dict[param] = np.arange(i, i+1)
                i += 1
            else:
                param_size = self.p_initial[param].size
                order_dict[param] = np.arange(i, i+param_size)
                i += param_size
        
        cov_mat = np.zeros((i, i))

        for (k1, v1) in order_dict.items():
            for (k2, v2) in order_dict.items():
                if v1.size > 1 and v2.size == 1:
                    cov_mat[np.ix_(v1, v2)] = self.cov_dict[k1][k2].reshape((v1.size, v2.size))
                else:
                    cov_mat[np.ix_(v1, v2)] = self.cov_dict[k1][k2]

        return cov_mat
    
    
    def predict(self, p_untransf):
        
        self.eigen_dict = self.eigen_fn(p_untransf, self.x_l, self.x_t, self.kf, eigen_dict = self.eigen_dict)
        
        p = self.transf_p(p_untransf)
        
        x_l_s = self.x_l.copy()
        x_t_s = self.x_t.copy()
        

        Kl_s = self.kf.Kl(p, self.x_l, x_l_s, wn = False)
        Kt_s = self.kf.Kt(p, self.x_t, x_t_s, wn = False)
        Sl_s = self.kf.Sl(p, self.x_l, x_l_s, wn = False)
        St_s = self.kf.St(p, self.x_t, x_t_s, wn = False)

        
        Kl_ss = self.kf.Kl(p, x_l_s, x_l_s, wn = True)
        Kt_ss = self.kf.Kt(p, x_t_s, x_t_s, wn = True)
        Sl_ss = self.kf.Sl(p, x_l_s, x_l_s, wn = True)
        St_ss = self.kf.St(p, x_t_s, x_t_s, wn = True)

        # Generate mean function and compute residuals
        M = self.mf(p, self.x_l, self.x_t)
        R = self.Y - M
        alpha = kronecker_inv_vec(R, self.eigen_dict)

        gp_mean = M
        gp_mean += kron_prod(Kl_s.T, Kt_s.T, alpha)
        gp_mean += kron_prod(Sl_s.T, St_s.T, alpha)

        Y_l = Kl_s.T @ self.eigen_dict["W_l"]
        Y_t = Kt_s.T @ self.eigen_dict["W_t"]
        Z_l = Sl_s.T @ self.eigen_dict["W_l"]
        Z_t = St_s.T @ self.eigen_dict["W_t"]

        sigma_diag = np.outer(np.diag(Kl_ss), np.diag(Kt_ss))
        sigma_diag += np.outer(np.diag(Sl_ss), np.diag(St_ss))

        sigma_diag -= kron_prod(Y_l**2, Y_t**2, self.eigen_dict["D_inv"])
        sigma_diag -= kron_prod(Z_l**2, Z_t**2, self.eigen_dict["D_inv"])
        sigma_diag -= 2*kron_prod(Y_l * Z_l, Y_t * Z_t, self.eigen_dict["D_inv"])

        return gp_mean, sigma_diag, M
    
    
    def clip_outliers(self, p, sigma):
        
        gp_mean, sigma_diag, M = self.predict(p)
        
        R = self.Y - gp_mean
        Z = jnp.abs(R/jnp.sqrt(sigma_diag))
        
        Y_clean = self.Y.copy()
        
        plt.imshow(Z)
        plt.colorbar()
        plt.show()
        
        Y_clean = Y_clean.at[Z > sigma].set(gp_mean[Z > sigma])
        
        return Y_clean
    
    
    def plot(self, p_untransf, fig=None):
    
        #run prediction
        gp_mean, gp_cov, M = self.predict(p_untransf)
        
        if fig is None: fig = plt.figure(figsize = (20, 5))
        ax = fig.subplots(1, 4)
        
        ax[0].imshow(self.Y, aspect = 'auto')
        ax[1].imshow(gp_mean, aspect = 'auto')
        ax[2].imshow(gp_mean - M, aspect = 'auto')
        ax[3].imshow(self.Y - gp_mean, aspect = 'auto')

        ax[0].set_ylabel('x_l')
        for i in range(4):
            ax[i].set_xlabel('x_t')
        
        
    def logL_arr(self, p_arr):
        # Same as logL but with the input parameters as an array
        
        # Convert array to a parameter dictionary and then perform as logL as normal
        p = self.make_p_dict(p_arr)
        
        return self.logL(p)
    
    
    def grad_logL_arr(self, p_arr):
        # Same as grad_logL but with the input and output parameters as an array
        
        p = self.make_p_dict(p_arr)
        grad_dict = self.grad_logL(p)
        return jax.flatten_util.ravel_pytree(grad_dict)[0]
