---
layout: post
title:  "Jupyter import"
date:   2016-09-02 18:30:00 +0100
categories: fast_copy_paste
---

{% highlight python %}
# Public libraries
import fitwrap as fw
import glob, os, sys, shutil # system packages
import matplotlib.pyplot as plt
import mpmath as mp
import nptable
import numpy as np
import pandas as pd
import pickle
import scipy.constants as const
import scipy.ndimage as ndimage
import scipy.optimize as opt
import scipy.signal as signal
import time

# Local libraries
# sys.path.append("./pyLi") # Adds pyLi directory to python modules path.
# import lithium as li
# import other as ot

# Loading constants
a0 = const.physical_constants["Bohr radius"][0]
c = const.physical_constants["speed of light in vacuum"][0]
h = const.physical_constants["Planck constant"] [0]
hbar = h/(2*np.pi)
kB = const.physical_constants["Boltzmann constant"][0]
m = 9.9883414 * 10**(-27) # Lithium mass [Kg]
muB = const.physical_constants["Bohr magneton"][0]


# Setting inline plots in notebook
%load_ext autoreload
%autoreload 2
%matplotlib inline
if os.name == 'posix':
    %config InlineBackend.figure_format = 'retina' #retina display settings
    
# Set and create an os independent savepath
save_path = os.sep.join((os.path.expanduser("~"), "Desktop", "jupyter"))
os.makedirs(save_path, exist_ok=True)
mplc = plt.rcParams['axes.prop_cycle'].by_key()['color']
{% endhighlight %}