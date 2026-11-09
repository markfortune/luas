(index)=

# luas

:::{note}
This is a new update to luas which is still in development. Feel free to play around with the tutorials but the interface has not yet stabilised and some functions may still fail silently. You can expect a more stable interface with better documentation and tutorials in the next 1-2 months.
:::

`luas`, from the Irish word for speed, is a library designed for building efficient Gaussian processes (GPs) primarily for two-dimensional rectangular data sets, although it can be used for one-dimensional (or even n-dimensional) datasets. By utilising different optimisations - such as using Kronecker product algebra - we can make the application of GPs to 2D data sets which can contain millions of points with log-likelihood calculations often measured in milliseconds (see [Benchmarking](benchmarking.md)). `luas` can be used with popular inference frameworks such as [`PyMC`](https://www.pymc.io/welcome.html) and [`NumPyro`](https://num.pyro.ai/en/latest/index.html), for which there are [tutorials](tutorials_index) to help you get started.

:::{note}
This package has been implemented using [`jax`](https://github.com/google/jax), which helps to calculate derivatives of the log-likelihood, as well as permitting the code to be easily run on either CPU or GPU. See [Working with JAX](things_to_know) for more details.
:::

These optimisations should have broad applications across astronomy, since 2D data sets frequently appear in the form of images and spectroscopic time-series, and even 3D data cubes could be modelled with particular `luas` kernels. Initially `luas` only implemented the [`LuasKernel`](api-luaskernel) optimisation, which has been in use within computational biology since its introduction in [Rakitsch et al. (2013)](https://proceedings.neurips.cc/paper/2013/hash/59c33016884a62116be975a9bb8257e3-Abstract.html), but was first applied to astronomy in the context of exoplanet transmission spectroscopy ([Fortune et al. 2024](https://www.aanda.org/articles/aa/full_html/2024/06/aa47613-23/aa47613-23.html)), and modelling JWST/MIRI photometric time-series ([Fortune et al. 2025](https://www.aanda.org/articles/aa/full_html/2025/09/aa54198-25/aa54198-25.html)). More recently, we have introduced many new Kronecker product-based optimisations in [Fortune et al. (2026)](https://arxiv.org/pdf/2611.09198), which this latest version of `luas` implements. This allows `luas` to work in combination with fast GP optimisations like [celerite](https://iopscience.iop.org/article/10.3847/1538-3881/aa9332) and quasiseparable kernels. We have added support for quasiseparable kernels by using [`tinygp`](https://tinygp.readthedocs.io/en/stable/) as a backend, with these kernels wrapped in the `luas.kernels.quasisep` submodule for ease of use. See [Why use luas?](why_use_luas) for more details on these optimisations and their advantages, and see [luas for exoplanets](luas_for_exo) for more information on how to use this package for applications within exoplanet astronomy.

## License and citing

`luas` is licensed under an MIT license, feel free to use. We hope by making this package freely available and open source it will make it easier for people to account for systematics correlated across multiple dimensions in data sets, in addition to being helpful for any other applications (e.g. interpolation).

If you are using `luas` then please cite our original 2024 paper introducing `luas` from [Fortune et al. (2024)](https://www.aanda.org/articles/aa/full_html/2024/06/aa47613-23/aa47613-23.html), as well as the new updated version documented in [Fortune et al. (2026)](https://arxiv.org/pdf/2611.09198) with the BibTeX provided below:

```
@ARTICLE{fortune2024,
       author = {{Fortune}, Mark and {Gibson}, Neale P. and {Foreman-Mackey}, Daniel and {Evans-Soma}, Thomas M. and {Maguire}, Cathal and {Ramkumar}, Swaetha},
        title = "{How do wavelength correlations affect transmission spectra? Application of a new fast and flexible 2D Gaussian process framework to transiting exoplanet spectroscopy}",
      journal = {\aap},
     keywords = {methods: data analysis, methods: statistical, techniques: spectroscopic, planets and satellites: atmospheres, stars: individual: WASP-31, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = jun,
       volume = {686},
          eid = {A89},
        pages = {A89},
          doi = {10.1051/0004-6361/202347613},
archivePrefix = {arXiv},
       eprint = {2402.15204},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024A&A...686A..89F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@misc{fortune2026,
      title={New optimisations for exact, scalable 2D Gaussian process inference, with applications to astronomical time-series}, 
      author={Mark Fortune and Neale P. Gibson and Benjamin Pope},
      year={2026},
      eprint={2611.09198},
      archivePrefix={arXiv},
      primaryClass={astro-ph.EP},
      url={https://arxiv.org/abs/2611.09198}, 
}
```

If using the kernels in `luas.kernels.quasisep` for fast quasiseparable kernels, then it's important you also cite the package [`tinygp`](https://tinygp.readthedocs.io/en/stable/), since `luas` wraps this package to implement them. I would also suggest citing the original [celerite](https://iopscience.iop.org/article/10.3847/1538-3881/aa9332) paper, as it introduces a special case of quasiseparable kernels in all but name. There's not really a clear `quasiseparble` kernel paper to cite, they can be viewed as extending mathematics from celerite to Kalman filtering/state-space models (e.g. [Hartikainen & Sarkka 2010](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5589113), but described more recently in an astronomy context in [Rubenzahl & Hattori et al. 2026](https://iopscience.iop.org/article/10.3847/1538-3881/ae4d0b/meta)), and also to quasiseparable matrix structure (as described in [Eidelman & Gohberg 1999](https://link.springer.com/article/10.1007/BF01300581)).

```
@software{2024zndo..10463641F,
       author = {{Foreman-Mackey}, Daniel and {Yu}, Weixiang and {Yadav}, Sachin and {Reynolds Becker}, McCoy and {Caplar}, Neven and {Huppenkothen}, Daniela and {Killestein}, Thomas and {Tronsgaard}, Ren{\'e} and {Rashid}, Theo and {Schmerler}, Steve},
        title = "{dfm/tinygp: The tiniest of Gaussian Process libraries}",
         year = 2024,
        month = jan,
          eid = {10.5281/zenodo.10463641},
          doi = {10.5281/zenodo.10463641},
      version = {v0.3.0},
    publisher = {Zenodo},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024zndo..10463641F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Mackey2017,
       author = {{Foreman-Mackey}, Daniel and {Agol}, Eric and {Ambikasaran}, Sivaram and {Angus}, Ruth},
        title = "{Fast and Scalable Gaussian Process Modeling with Applications to Astronomical Time Series}",
      journal = {\aj},
     keywords = {asteroseismology, methods: data analysis, methods: statistical, planetary systems, stars: rotation, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Solar and Stellar Astrophysics, Physics - Data Analysis, Statistics and Probability, Statistics - Applications},
         year = 2017,
        month = dec,
       volume = {154},
       number = {6},
          eid = {220},
        pages = {220},
          doi = {10.3847/1538-3881/aa9332},
archivePrefix = {arXiv},
       eprint = {1703.09710},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

We hope to expand the functionality of `luas` over time and welcome any help to do so. Also, if you encounter any issues, have any requests or questions then feel free to [raise an issue](https://github.com/markfortune/luas/issues) or [send an email](mailto:markfortuneis@gmail.com).


## Overview of documentation

```{toctree}
:maxdepth: 2
getting_started
tutorials
api
GitHub repository <https://github.com/markfortune/luas>
```