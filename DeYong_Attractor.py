#=================================================================================
#----------------------      De Jong Attractor     -------------------------------
#=================================================================================

#-------------------     X = sin(a * Y) - cos(b * X)     -------------------------
#-------------------     Y = sin(c * X) - cos(d * Y)     -------------------------

#               This program must be run in a Jupyter notebook.
#---------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib as mpl
import pandas as pd
import sys
import datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import Greys9, inferno, viridis
from datashader.utils import export_image
from functools import partial
from numba import jit
import numba
from colorcet import palette

#---------------------------------------------------------------------------------

background = "white"
img_map = partial(export_image, export_path="dejong_maps", background=background)

n = 10000000

#---------------------------------------------------------------------------------

@jit
def trajectory(fn, a, b, c, d, x0=0, y0=0, n=n):

    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0

    for i in np.arange(n-1):
        x[i+1], y[i+1] = fn(a, b, c, d, x[i], y[i])

    return pd.DataFrame(dict(x=x, y=y))

@jit
def dejong(a, b, c, d, x, y):

    return np.sin(a*y) - np.cos(b*x),   np.sin(c*x) - np.cos(d*y)

#---------------------------------------------------------------------------------

cmaps =  [palette[p][::-1] for p in ['bgy', 'bmw', 'bgyw', 'bmy', 'fire', 'gray', 'kbc', 'kgy']]
cmaps += [inferno[::-1], viridis[::-1]]
cvs = ds.Canvas(plot_width = 500, plot_height = 500)
ds.transfer_functions.Image.border=0

#---------------------------------------------------------------------------------

# Parameter  :            a=xxx,   b=xxx,   c=xxx,  d=xxx, 
df = trajectory(dejong,     -2,      -2,    -1.2,      2,   0,   0)
#df = trajectory(dejong,   2.01,   -2.53,    1.61,  -0.33,   0,   0)
#df = trajectory(dejong,  1.641,   1.902,   0.316,  1.525,   0,   0)
#df = trajectory(dejong,    1.4,    -2.3,     2.4,   -2.1,   0,   0)

# Try to put a value in xxx.
#df = trajectory(dejong,    xxx,     xxx,     xxx,    xxx,   0,   0)

agg = cvs.points(df, 'x', 'y')
img = tf.shade(agg, cmap = cmaps[4], how='linear', span = [0, n/60000])
img_map(img,"attractor")

