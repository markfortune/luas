.. _api-kernels-quasisep:

kernels.quasisep package
========================

.. automodule:: luas.kernels.quasisep

These kernels return quasiseparable covariance objects, built on tinygp's
quasiseparable kernels. Calculations along that dimension scale linearly with
the number of points, rather than cubically for the dense kernels in
:doc:`kernels`. The ``*Approx`` kernels are quasiseparable approximations to
kernels that are not exactly quasiseparable.

.. currentmodule:: luas.kernels.quasisep

.. autosummary::
   :toctree: summary

   Exp
   Matern32
   Matern52
   MaternHalfInt
   SHO
   Cosine
   SquaredExpApprox
   ExpSineSquaredApprox
   QuasiperiodicApprox
   Banded
   Constant
   ConstantBlocks
   CustomTinygp
