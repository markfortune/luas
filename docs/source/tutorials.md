(tutorials_index)=

# Tutorials

Included below are tutorials introducing Gaussian processes and their extension to 2D data sets.

The optimised method used by `luas` to calculate the log-likelihood and its derivatives is first demonstrated using an implementation in NumPy in ["An Introduction to 2D Gaussian Processes"](numpy_tutorial). This should be useful if you want to understand the mathematics behind the optimisation in [`LuasKernel`](api-luaskernel) but isn't necessary to use `luas`.

If you would like to see a sample analysis using `luas` then there is an example demonstrating an analysis of synthetic spectroscopic transit light curves containing time and wavelength correlated noise using [PyMC](transit_spec_pymc) as the inference library.

```{toctree}
:maxdepth: 1

tutorials/Walking_through_luas_optimisations
tutorials/Simple_1D_example
tutorials/joint_transit_spec_example_pymc
tutorials/2D_GP_intro
```
