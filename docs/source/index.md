(index)=

# luas

'luas' is a small library aimed at building Gaussian processes (GPs) primarily for two-dimensional data sets. It contains an implementation of the methods described in Fortune et al. *in review* to efficiently calculate the log-likelihood and it's derivatives for kronecker structured covariance matrices when the input data lies on a (potentially non-uniform) grid. The particular optimisation used was originally introduced in Rakitsch et al. 2013 and can perform exact inference for any covariance matrix which is the sum of two kronecker products of covariance matrices for each dimension. It is a generalisation of the more widely-used method introduced in Saatchi 2011 which can only perform exact inference on multi-dimensional data sets when the white noise in the data is uniform.

:::{note}
This package has been implemented using [`jax`](https://github.com/google/jax), which helps to calculate derivatives of the log-likelihood as well as allowing the same code to be run on either a CPU or a GPU.
:::

This package was originally developed for use in fitting systematics in transiting exoplanet spectroscopy where the two-dimensional data sets are across wavelength and time. While the tutorials included mainly focus on this use case, the luas package itself has been written to be a general-purpose 2D GP package which could have broad applications for interpolation and inference on many large 2D data sets. The 'luas.exoplanet' submodule has however been included to help get people using the module for exoplanet spectroscopy started - which uses [`jaxoplanet`](https://github.com/exoplanet-dev/jaxoplanet) as a backend for transit modelling - but this is not a dependency for the rest of the 'luas' module.

## Where to get started

```{toctree}
:maxdepth: 3
getting_started
tutorials
api
```

## License and Citing

'luas' is licensed under an MIT license, feel free to use
