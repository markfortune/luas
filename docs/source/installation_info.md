(installation)=
# Installation

Currently supported Python versions are 3.12 - 3.14. From my testing, I've found jax 0.10.1 to run particularly quickly for some tinygp functionality, so you may want to start out by running ``pip install jax==0.10.1``. That said, testing has shown it works with jax >= 0.4.38 and the latest versions of jax are automatically tested on every pull request.

For installing tinygp, you currently need to install the latest version from its GitHub repo , as the latest named release is quite out of date and will break on some luas functions + lacks some new speedups which provide a performance boost (particularly on GPU). You can do this as follows:
```bash
pip install git+https://github.com/dfm/tinygp.git
```
Once you have tinygp installed, you can decide whether you want to install luas along with jaxoplanet, PyMC and NumPyro, which I recommend if using luas for exoplanet research. In this case you can run:
```bash
pip install luas-gp[exoplanet]
```
However, if you just want luas for your own inference setup and don't need jaxoplanet, then you can just run:
```bash
pip install luas-gp
```
If running Jupyter/IPython notebooks then you will also want to make sure both ipykernel and ipywidgets (for PyMC progress bars) are installed, both of which can be installed using pip. If progress bars are not appearing properly when running gp.optimise/MCMC sampling with PyMC it can be that it loads too slow on the first run, and runs correctly afterwards. Some of the tutorial notebooks pre-load it to avoid that.

## Installing on GPU

:::{note}
Due to the required numerical precision for matrix decomposition, double-precision floating point arithmetic is required which many GPUs either do not support or will perform poorly at. Make sure to check the specific hardware capabilities of the GPU being used e.g. number of double-precision FLOPS. GPUs which will perform exceptionally well include common HPC hardware such as NVIDIA TESLA V100/A100/H100s.
:::

`luas` requires `jax` to run, if you're running on CPU then a simple `pip install jax` will likely suffice, which luas will install automatically as a dependency. However, if installing for GPU I'd recommend first visiting the [`jax installation page`](https://github.com/google/jax/#installation) for instructions on how to install it for your specific choice of hardware, before then installing tinygp and luas as above. A GPU is not required to run any of the code, but it can significantly speed up many of the calculations, so be sure to follow the instructions for installing with GPU if you have one available.
