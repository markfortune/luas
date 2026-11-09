# luas

luas is a library focused on implementing fast and flexible 2D Gaussian processes using Kronecker product algebra, which assumes the data is rectangular and without missing values, but doesn't assume uniform sampling. We have also implemented ``luas`` for 1D datasets (e.g. time-series) and also N-dimensional datasets provided the kernel inputs lie on a grid. ``luas`` has been written in ``jax`` and can be easily used in combination with inference libraries such as ``PyMC`` and ``NumPyro``. Documentation and tutorials are located at https://luas.readthedocs.io/en/latest/.

See [Fortune et al. (2026)]([https://arxiv.org/abs/2611.09198]) which explains the GP optimisations implemented in `luas`, while  [Fortune et al. (2024)]([https://ui.adsabs.harvard.edu/abs/2024A%26A...686A..89F/abstract]) is the original paper for ``luas`` which outlines the use of this package to analyse transmission spectroscopy observations.

## Installation

See the installation page on readthedocs [here]([[https://arxiv.org/abs/2611.09198](https://luas.readthedocs.io/en/latest/installation_info.html)]).

## License & Citing

`luas` is licensed under an MIT license, feel free to use. We hope by making this package freely available and open source it will make it easier for people to account for systematics correlated across two dimensions in data sets, in addition to being helpful for any other applications (e.g. interpolation). See the [docs](https://luas.readthedocs.io/en/latest/index.html) for more info on citing.

We hope to expand the functionality of `luas` over time and welcome any help to do so. Also, if you encounter any issues, have any requests or questions then feel free to [raise an issue](https://github.com/markfortune/luas/issues) or [send an email](mailto:mark.fortune@mq.edu.au).
