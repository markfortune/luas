(installation)=

# Installation

`luas` requires `jax` to run so see the [`jax installation page`](https://github.com/google/jax/#installation) for instructions on how to install it for your specific choice of hardware. A GPU is not required to run any of the code, but it can significantly speed up many of the calculations, so be sure to follow the instructions for installing with GPU if you have one available.

:::{note}
Due to the required numerical precision for matrix decomposition, double-precision floating point arithmetic is required which many GPUs either do not support or will perform poorly at. Make sure to check the specific hardware capabilities of the GPU being used e.g. number of double-precision FLOPS. GPUs which will perform exceptionally well include common HPC hardware such as NVIDIA TESLA V100/A100/H100s.
:::

`luas` also does not have its own inference library but can be used with `PyMC` and `NumPyro`, so if you want to perform inference it is likely you will need to install one of these. See the installation guides for [`PyMC`](https://www.pymc.io/projects/docs/en/latest/installation.html) and [`NumPyro`](https://num.pyro.ai/en/latest/getting_started.html#installation).

If performing exoplanet transit modelling with the `luas.exoplanet` submodule then you will need the additional dependency `jaxoplanet` which should be installable using pip (see [`jaxoplanet`](https://github.com/exoplanet-dev/jaxoplanet) for the latest information).

Installation of `luas` can be performed by cloning the GitHub repository and using pip.

```bash
git clone https://github.com/markfortune/luas.git
cd luas
python -m pip install -e .
```

