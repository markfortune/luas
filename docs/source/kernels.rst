.. _api-kernels:

kernels package
===============

.. automodule:: luas.kernels

Kernel functions build the covariance matrix along one dimension of the data
(e.g. wavelength or time) and are the main way of building a model in luas.
Each function returns a covariance object (see :doc:`covtype`). These objects
can be added and multiplied together, and can be passed to one of the 2D
kernel classes (e.g. :class:`luas.LuasKernel`) for building Kronecker kernels.

The kernels in this package build dense covariance matrices. Versions that
exploit quasiseparable or low-rank structure are in the
:doc:`kernels_quasisep` and :doc:`kernels_lowrank` subpackages.

.. currentmodule:: luas.kernels

.. autosummary::
   :toctree: summary

   Noise
   KroneckerDelta

Stationary Kernels
------------------

Stationary kernels depend only on the distance between two input points. Most
take a length scale ``scale``, written as :math:`l` in the equations, and an
amplitude ``sigma`` (:math:`\sigma`, default ``1``).

.. currentmodule:: luas.kernels

.. autosummary::
   :toctree: summary

   Exp
   SquaredExp
   Matern32
   Matern52
   MaternHalfInt
   RationalQuadratic
   PoweredExp
   Cosine
   ExpSineSquared

Other Kernels
-------------

.. currentmodule:: luas.kernels

.. autosummary::
   :toctree: summary

   Constant
   Fixed
   Custom

Subpackages
-----------

.. toctree::
   :maxdepth: 1

   kernels_quasisep
   kernels_lowrank
   covtype
