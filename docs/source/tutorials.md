(tutorials_index)=

# Tutorials

Included below are tutorials introducing Gaussian processes and their extension to 2D data sets.

The optimised method used by `luas` to calculate the log-likelihood and its derivatives is first demonstrated using an implementation in NumPy in ["An Introduction to 2D Gaussian Processes"](numpy_tutorial). This should be useful if you want to understand the mathematics behind the optimisation in [`LuasKernel`](api-luaskernel) but isn't necessary to use `luas`.

If you would like to see a sample analysis using `luas` then there are examples demonstrating a re-analysis of the VLT/FORS2 archival data of WASP-31b (originally analysed in [Gibson et al. 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.467.4591G/abstract)) using either [PyMC](pymc_example) or [NumPyro](numpyro_example) as the choice of inference library.

```{toctree}
:maxdepth: 1

tutorials/2D_GP_intro
tutorials/PyMC Example
tutorials/NumPyro Example
```
