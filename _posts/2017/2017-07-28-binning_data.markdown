---
layout: post
title:  "Binning data"
date:   2017-07-27 22:00:00 +0100
categories: data_analysis
---

Possible way to bin data using numpy digitize function:



{% highlight python %}
def bin_data(data_x, data_y, bins):
    digitized = np.digitize(data_x, bins)
    x = np.array([data_x[digitized == i].mean() for i in range(1, len(bins)) if i in digitized])
    x_err = np.array([data_x[digitized == i].std() for i in range(1, len(bins)) if i in digitized])
    y = [data_y[digitized == i].mean() for i in range(1, len(bins)) if i in digitized]
    y_err = np.array([data_y[digitized == i].std() for i in range(1, len(bins)) if i in digitized])
    return x, y, x_err, y_err
{% endhighlight %}

Example:
{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt

x = np.random.random(size=500)
y = np.random.normal(size=500)
bins = np.linspace(0, 1, 10)

plt.scatter(x, y, color='grey', s=5)

x_bin, y_bin, x_err, y_err = bin_data(x, y, bins)
plt.errorbar(x_bin, y_bin, yerr= y_err, xerr=x_err, fmt='H', ms=10, color='black')
{% endhighlight %}

{% include _images/{{ page.date | date: '%Y-%m-%d'}}/binning.svg%}
