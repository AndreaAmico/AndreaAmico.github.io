---
layout: post
title:  "Jupyter import"
date:   2016-09-02 18:30:00 +0100
categories: fast_copy_paste
---

{% highlight python %}
# Public libraries
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.constants as const
import pandas as pd
import lmfit

# Home made libraries
# import pyLi.loadImages as loadimages


# Loading constants
c = const.physical_constants["speed of light in vacuum"][0]
h = const.physical_constants["Planck constant"] [0]
hbar = h/(2*np.pi)
kB = const.physical_constants["Boltzmann constant"][0]
muB = const.physical_constants["Bohr magneton"][0]
m = 9.9883414 * 10**(-27) #kg

# Setting inline plots in notebook
%matplotlib inline
{% endhighlight %}