(tutorials_index)=

# Tutorials

Included below are tutorials introducing Gaussian processes and their extension to 2D data sets.

We first walk through the different Kronecker product optimisations available in luas, demonstrating how to build these models using luas' GP and kernel function interface. We then include an example fitting a simple 1D transit time-series, showing how luas can be integrated with PyMC or NumPyro, and how it can be used for 1D GPs. We then demonstrate how luas can be scaled to joint-fitting a transmission spectroscopy time-series, similar to Fortune et al. (2024), and shows how inference can be perfomed using either PyMC or NumPyro. Finally, we give a NumPy example implementing the LuasKernel from scratch, in case you want to dig into some of the maths of Kronecker product-based optimisations.

```{toctree}
:maxdepth: 1

tutorials/Walking_through_luas_optimisations
tutorials/Simple_1D_example
tutorials/joint_transit_spec_example
tutorials/2D_GP_intro
```
