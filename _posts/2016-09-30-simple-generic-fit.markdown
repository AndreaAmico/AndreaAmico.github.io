---
layout: post
title:  "Generic function fit"
date:   2016-09-30 13:30:00 +0100
categories: data_analysis
---

Fitter prototype using [lmfit](https://lmfit.github.io/lmfit-py/ "https://lmfit.github.io/lmfit-py/") module.

--TODO--
Copy paste use
{% highlight python %}
import matplotlib.pyplot as plt
import numpy as np
import lmfit

params = lmfit.Parameters()
params.add('aa', value=1, vary=True, min=-np.inf, max=np.inf)
params.add('bb', value=-5, vary=True, min=-np.inf, max=np.inf)
params.add('cc', value=1, vary=True, min=-np.inf, max=np.inf)

def fit_function(params, x):
    aa = params['aa'].value
    bb = params['bb'].value
    cc = params['cc'].value
    
    return aa + bb*x + cc*x**2

data_x = np.array([1,3,4,5,7,8,9,11])
data_y = np.array([1,9,15,27,40,66,100,150])

##--------------------------------------------------------------------##
try: data_x.shape
except: data_x = np.arange(data_y.shape[0])
def residual(params, x, data): return (data - fit_function(params, x))
out = lmfit.minimize(residual, params, args=(data_x, data_y))
x_plot = np.linspace(np.min(data_x), np.max(data_x), 100)
plt.scatter(data_x, data_y)
plt.plot(x_plot, fit_function(params, x_plot), c="red") # init param
plt.plot(x_plot, fit_function(out.params, x_plot), c="green")
plt.xlabel("data x")
plt.ylabel("data y")
# plt.savefig("out.svg") # save figure
plt.show()
lmfit.report_fit(out, show_correl=False)
##--------------------------------------------------------------------##
{% endhighlight %}

Figure output:
{% include _images/{{ page.date | date: '%Y-%m-%d'}}/fit_prototype.svg%}

Text output:
{% highlight python %}
[[Fit Statistics]]
    # function evals   = 10
    # data points      = 8
    # variables        = 3
    chi-square         = 232.924
    reduced chi-square = 46.585
    Akaike info crit   = 36.730
    Bayesian info crit = 36.969
[[Variables]]
    aa:   8.61092408 +/- 8.611083 (100.00%) (init= 1)
    bb:  -5.93825394 +/- 3.222284 (54.26%) (init=-5)
    cc:   1.70532460 +/- 0.260686 (15.29%) (init= 1)
{% endhighlight %}
