# luas
A library designed for implementing fast and flexible 2D Gaussian processes. It has been written in ``jax`` and can be easily used in combination with inference libraries such as ``PyMC`` and ``NumPyro``. Documentation and tutorials are located at https://luas.readthedocs.io/en/latest/.

The paper outlining the use of this package to analyse transmission spectroscopy observations has been accepted and is in the process of being published. It is currently available on [arXiv](https://arxiv.org/abs/2402.15204).

## Installation

Install by cloning and navigating into the downloaded directory, followed by entering the command:
```
$ pip install .
```

## License & Citing

`luas` is licensed under an MIT license, feel free to use. We hope by making this package freely available and open source it will make it easier for people to account for systematics correlated across two dimensions in data sets, in addition to being helpful for any other applications (e.g. interpolation).

If you are using `luas` then please cite our work [Fortune et al. (2024)](https://arxiv.org/abs/2402.15204) with the BibTeX provided below:

```
@ARTICLE{2024arXiv240215204F,
       author = {{Fortune}, Mark and {Gibson}, Neale P. and {Foreman-Mackey}, Daniel and {Mikal-Evans}, Thomas and {Maguire}, Cathal and {Ramkumar}, Swaetha},
        title = "{How do wavelength correlations affect your transmission spectrum? Application of a new fast and flexible 2D Gaussian process framework to transiting exoplanet spectroscopy}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = feb,
          eid = {arXiv:2402.15204},
        pages = {arXiv:2402.15204},
archivePrefix = {arXiv},
       eprint = {2402.15204},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240215204F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

We hope to expand the functionality of `luas` over time and welcome any help to do so. Also, if you encounter any issues, have any requests or questions then feel free to [raise an issue](https://github.com/markfortune/luas/issues) or [send an email](mailto:fortunma@tcd.ie).
