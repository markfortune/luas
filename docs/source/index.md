(index)=

# luas

'luas' is a small library aimed at building Gaussian processes (GPs) primarily for two-dimensional data sets. It contains an implementation of the methods described in Fortune et al. (2024) to efficiently calculate the log-likelihood and it's derivatives for kronecker structured covariance matrices when the input data lies on a (potentially non-uniform) grid. The particular optimisation implemented was originally introduced in Rakitsch et al. (2013) as an extension of work in Saatchi (2011) and can perform exact inference for any covariance matrix which is the sum of two kronecker products of covariance matrices for each dimension of the data set. `luas` also supports Gaussian processes with any general covariance matrix applied to 2D data sets for problems where the kronecker product structure is not suitable.

:::{note}
This package has been implemented using [`jax`](https://github.com/google/jax), which helps to calculate derivatives of the log-likelihood as well as permitting the code to be easily run on either CPU or GPU.
:::

This package was originally developed for use in fitting systematics in transiting exoplanet spectroscopy where the two-dimensional data sets may be correlated across wavelength and time. While the tutorials included mainly focus on this use case, the luas package itself has been written to be a general-purpose 2D GP package which could have broad applications for interpolation and inference on many large 2D data sets. The 'luas.exoplanet' submodule has however been included to help get people started if using `luas` for exoplanet spectroscopy. This submodule uses [`jaxoplanet`](https://github.com/exoplanet-dev/jaxoplanet) as a backend for transit modelling which if used should also be cited. Other exoplanet packages with implementations in `jax` currently include [`kelp`](https://github.com/bmorris3/kelp) - which can be used for computing exoplanet phase curves in both reflected and emitted light - as well as [`harmonica`](https://github.com/DavoGrant/harmonica/tree/main) - which can be used for mapping the shapes of exoplanets from transmission spectroscopy.

## Where to get started

```{toctree}
:maxdepth: 2
getting_started
tutorials
api
GitHub repository <https://github.com/markfortune/luas>
```

## License and Citing

'luas' is licensed under an MIT license, feel free to use
