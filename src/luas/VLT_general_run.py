import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import logging
import jax
import jax.numpy as jnp

logging.getLogger().setLevel(logging.INFO)
os.chdir("/Users/mark/Code/luas")

from luas.GPyMCClass import GPyMC, GPyMC_from_file
from luas import build_VLT_kernel, build_VLT_GeneralKernel
from luas import transit_2D, transit_param_transform
from luas import *

gp2 = GPyMC_from_file("analyses/default_loc/181153_08_09_23_374", "GP_save_80668031.pkl",
                      build_VLT_GeneralKernel(), mf = transit_2D, transform_fn = transit_param_transform)
print("Starting logP = ", gp2.logP(gp2.p))
gp2.run_NUTS(tune = 1000, draws = 1000, chains = 2, slice_sample = [])
gp2.save()