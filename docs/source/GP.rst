.. _api-gp:

GP
==

.. currentmodule:: luas

.. autoclass:: GP
   :show-inheritance:

.. The method tables import the methods through the module luas.GP, see the note in conf.py.
.. Only the public methods are listed here.
.. currentmodule:: luas.GP

Log likelihood and log posterior
--------------------------------

.. autosummary::
   :toctree: summary
   :template: autosummary/gp_method.rst

   ~GP.logL
   ~GP.logP

Predictions and sampling
------------------------

.. autosummary::
   :toctree: summary
   :template: autosummary/gp_method.rst

   ~GP.predict
   ~GP.sample
   ~GP.crossval

Optimisation and inference
--------------------------

.. autosummary::
   :toctree: summary
   :template: autosummary/gp_method.rst

   ~GP.optimise
   ~GP.pymc_model
   ~GP.laplace_approx
   ~GP.laplace_approx_with_bounds

Plotting
--------

.. autosummary::
   :toctree: summary
   :template: autosummary/gp_method.rst

   ~GP.plot
   ~GP.plot_lightcurves
   ~GP.plot_cov_mat
   ~GP.autocorrelate

Kernel-specific methods
-----------------------

.. autosummary::
   :toctree: summary
   :template: autosummary/gp_method.rst

   ~GP.transform_fn
   ~GP.orthogonalise
