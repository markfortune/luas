.. _api-kernels-lowrank:

kernels.lowrank package
=======================

.. automodule:: luas.kernels.lowrank

These kernels return a :class:`~luas.kernels.lowrank.LowRank` covariance
object, which represents the covariance matrix along that dimension with a
limited number of components (set with the ``rank`` argument where available).

.. currentmodule:: luas.kernels.lowrank

.. autosummary::
   :toctree: summary

   LowRank
   SquaredExp
   Cosine
   Constant
   Amplitude
   MixingMat
   ExpSineSquaredApprox
