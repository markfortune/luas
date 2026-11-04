(when_to_use)=

# Why use luas?

While there are already plenty of popular Gaussian process libararies - such as [`sklearn.gaussian_process`](https://scikit-learn.org/stable/modules/gaussian_process.html), [`george`](https://george.readthedocs.io), [`celerite`](https://celerite2.readthedocs.io) and [`tinygp`](https://tinygp.readthedocs.io/en/stable/index.html) - currently none of them implement the optimisations described in [Fortune et al. (2026)](https://github.com/markfortune/luas_dev). They allow for the exact calculation of the log likelihood for 2D data set of dimensions `(N_l, N_t)` in $\mathcal{O}(N_l^3 + N_t^3 + N_l N_t(N_l + N_t))$ runtime and $\mathcal{O}(N_l^2 + N_t^2)$ memory for any valid kernel function $k$ of the form:

$$
k(\lambda_1, \lambda_2, t_1, t_2) = s_{l}(\lambda_1, \lambda_2)s_{t}(t_1, t_2) + k_{l}(\lambda_1, \lambda_2)k_{t}(t_1, t_2)
$$

Where $s_{l}$, $k_{\lambda}$ are kernel functions which describe the covariance as a function of regression variables that lie along the first dimension of the data set and $k_{t_1}$, $k_{t_2}$ describe the covariance as a function of the second dimension.

This is a particularly general method as these kernel functions $k_{l_1}$, $k_{l_2}$, $k_{t_1}$ and $k_{t_2}$ can be any valid kernel function, the main restriction is that the covariance must be separable between the two dimensions. We also require that our 2D data set is complete (i.e. there is no missing data) and that it lies on a (potentially non-uniform) grid (i.e. that all data points lie on a 2D grid but we do not require uniform spacing along either dimension). This form of kernel function is implemented in the class [`LuasKernel`](api-luaskernel).

## Why not just implement in an existing GP library?

The purpose of `luas` is not to simply add another Gaussian process library to the mix but to specialise in the analysis of multidimensional data sets with GPs, and the interface has been designed to make this as intuitive as possible. The optimisation above is considered a Kronecker product method as a kernel of its form results in a covariance matrix which can be described as:

$$
\mathbf{K} = \mathbf{K}_\mathrm{\lambda} \otimes \mathbf{K}_\mathrm{t} + \mathbf{\Sigma}_\mathrm{\lambda} \otimes \mathbf{\Sigma}_\mathrm{t}
$$

Where $\otimes$ denotes the Kronecker product (i.e. each element in the first matrix is multiplied by every element in the second matrix). As these methods have limited restrictions other than separating out the kernel functions to individual dimensions they can have quite general applications. In addition, only some of the current Gaussian process libraries have implementations written using `jax` - which permits efficient automatic differentiation. This allows them to be used in combination with gradient-based optimisation methods and gradient-based MCMC methods. Lately there has been a rapid adoption of these techniques due to their efficiency (especially with large numbers of parameters) and thanks to the release of popular inference libraries such as [`NumPyro`](https://num.pyro.ai/en/latest/index.html) and [`PyMC`](https://www.pymc.io/welcome.html). The ability of these inference methods to efficently sample large numbers of parameters is particularly useful as extending GPs to 2D data sets often results in needing to fit more parameters simultaneously. `jax` also makes it easy to run code on GPUs, which can dramatically speed up log likelihood calculations for kernels such as the [`LuasKernel`](api-luaskernel).
