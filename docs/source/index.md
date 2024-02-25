(index)=

# luas

`luas` (from the Irish word for speed) is a small library aimed at building Gaussian processes (GPs) primarily for two-dimensional data sets. By utilising different optimisations - such as using kronecker product algebra - we can make the application of GPs to 2D data sets which may have dimensions of 100s-1000s along both dimensions possible within a reasonable timeframe. `luas` can be used with popular inference frameworks such as [`NumPyro`](https://num.pyro.ai/en/latest/index.html) and [`PyMC`](https://www.pymc.io/welcome.html) for which there are [tutorials](tutorials_index) to help you get started.

:::{note}
This package has been implemented using [`jax`](https://github.com/google/jax), which helps to calculate derivatives of the log-likelihood as well as permitting the code to be easily run on either CPU or GPU. See [Working with JAX](things_to_know) for more details.
:::

These methods could have broad uses for the interpolation of 2D data sets and are likely underutilised within fields such as astronomy where 2D data sets can appear not just from images but also from multiple time-series being joint analysed. The [`LuasKernel`](api-luaskernel) class contains an optimisation which has already been frequently used within computational biology since its introduction in [Rakitsch et al. (2013)](https://proceedings.neurips.cc/paper/2013/hash/59c33016884a62116be975a9bb8257e3-Abstract.html) but has only recently been applied to astronomy in the context of exoplanet transmission spectroscopy (Fortune et al. 2024). See [Why Use Luas?](why_use_luas) for more details on this optimisation and its advantages and see [Luas for Exoplanets](luas_for_exo) for more information on how to use this package for applications within exoplanet astronomy.

## Overview of Documentation

```{toctree}
:maxdepth: 2
getting_started
tutorials
api
GitHub repository <https://github.com/markfortune/luas>
```

## License and Citing

`luas` is licensed under an MIT license, feel free to use. We hope by making this package freely available and open source it will make it easier for people to account for systematics correlated across two dimensions in data sets, in addition to being helpful for any other applications (e.g. interpolation). If you are using `luas` then please cite our work Fortune et al. (2024).

We hope to expand the functionality of `luas` over time and welcome any help to do so. Also, if you encounter any issues, have any requests or questions then feel free to [raise an issue](https://github.com/markfortune/luas/issues) or [send an email](mailto:fortunma@tcd.ie).

