(luas_for_exo)=

# Luas for Exoplanets

`luas` has been designed to be a general-purpose GP library which should be simple to apply to any 2D data set. However, exoplanet astronomy is a particular focus and `luas` has already been used with [`jaxoplanet`](https://github.com/exoplanet-dev/jaxoplanet) for transmission spectroscopy in Fortune et al. (2024). This was for the analysis of ground-based data from VLT/FORS2 but an upcoming paper will be demonstrating how it can help analyse JWST data.

Currently `luas` only supports `jax` functions to be used as the deterministic mean function within the GP. However, this should not be a major issue as there are plenty of exoplanet packages with implementations in `jax`. In addition to `jaxoplanet` which can be used for efficient transit modelling, [`kelp`](https://github.com/bmorris3/kelp) can be used for computing exoplanet phase curves in both reflected and emitted light and [`harmonica`](https://github.com/DavoGrant/harmonica/tree/main) can be used for mapping the shapes of exoplanets (e.g. identifying limb-asymmetries) using transmission spectroscopy. While functions to build light curves using `jaxoplanet` have been included in the `luas.exoplanet` submodule, make sure to separately cite this package as it is independent from `luas`.
