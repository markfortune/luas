.. _api-kronecker-kernels:

Kronecker kernels
=================

Each of these classes represents the covariance matrix of a whole 2D data set,
built from covariance matrices along each dimension, and uses its structure to
compute the log likelihood efficiently.

.. currentmodule:: luas

.. autosummary::

   SeparableKernel
   LuasKernel
   LuasKernelND
   LuasLasrachKernel
   MultiLowRankKernel
   DoubleMultiLowRankKernel
   LuasPlusMultiLowRankKernel
   ExtendedLuasKernel
   GeneralKernel
   WhiteNoiseKernel

.. toctree::
   :hidden:

   SeparableKernel
   LuasKernel
   LuasKernelND
   LuasLasrachKernel
   MultiLowRankKernel
   DoubleMultiLowRankKernel
   LuasPlusMultiLowRankKernel
   ExtendedLuasKernel
   GeneralKernel
   WhiteNoiseKernel
