---
layout: post
title:  "fitwrap module"
date:   2018-03-03 11:00:00 +0100
categories: data_analysis
---

The fitwrap module is available on [github](https://github.com/AndreaAmico/fitwrap "https://github.com/AndreaAmico/fitwrap") and can be installed via pip:
{% highlight console %}
pip install fitwrap
{% endhighlight %}

This module provides a wrapper for the [scipy](https://www.scipy.org/ "https://www.scipy.org/") function `optimize.curve_fit()`. The full fitting procedure is reduced in one-line command, moreover the initial fitting parameters and boudaries are set by the keyword arguments of the model fitting function. The package comes with the following functions:
- `fit`: non linear 1D fit
- `fit2d`: non linear 2D fit
- `fit_sin`: sinusoidal fit with automatic fit funnction and initial guess.

Here are some usage example:
{% highlight python %}
import fitwrap as fw
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def model_function(x, off=4, m=(1.2, 1, 2), b=3.3, fixed_args=['off']):
    return off + m*x + b*x**2

np.random.seed(42)
x = np.random.random(50) - 0.5
y = model_function(x, off=3.3, m=3, b=15) + np.random.random(x.shape[0]) - 0.5

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 4))
fw.fit(model_function, x, y, fig_ax=[fig, ax1])

def model_function2(x, off=1, m=1, b=1):
    return off + m*x + b*x**2
fw.fit(model_function2, x, y, fig_ax=[fig, ax2], print_results=False);
{% endhighlight %}
off:  4       Fixed<br>
  m:  2.0     +/- 0.305     (15.2%)  initial:(1.2, 1, 2)<br>
  b:  9.404   +/- 0.789      (8.4%)  initial:3.3
{% include _images/{{ page.date | date: '%Y-%m-%d'}}/parabola.svg%}
------------
{% highlight python %}
import fitwrap as fw
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

def my_signal(t, off, amp, freq, phase):
    return  off + amp * np.sin(2*np.pi*freq*t + phase)

NOISE_LEVEL = 2
np.random.seed(42)
times = np.random.random(100)
signals = my_signal(times, 0.5, 2, 4, np.pi/4) + (np.random.random(times.shape)-0.5)*NOISE_LEVEL
errors = np.abs(my_signal(times, 0, 1, 4, np.pi/2))/2+0.01

fw.fit_sin(times, signals, sigma=errors)

{% endhighlight %}
Fitting function model: y = off + amp * sin(2 * pi * freq * x + phase)<br>
  off:  0.5493  +/- 0.0661    (12.0%)  initial:0.6485271451564831<br>
  amp:  1.969   +/- 0.222     (11.3%)  initial:4.236288465011205<br>
 freq:  3.9981  +/- 0.026      (0.7%)  initial:4.043991443019413<br>
phase:  0.73    +/- 0.149     (20.4%)  initial:0.39269908169872414
{% include _images/{{ page.date | date: '%Y-%m-%d'}}/sine.svg%}

-----------

{% highlight python %}
import fitwrap as fw
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

def g2(yx, x0=21, y0=32, sx=8, sy=5):
    return np.exp(-(yx[1]-x0)**2/(2*sx**2) -(yx[0]-y0)**2/(2*sy**2))
    
yx = np.mgrid[:100,:40]
gg = g2(yx, 20, 30, 5, 5)+(0.5-np.random.random(yx[0].shape))*0.2

fw.fit2d(g2, yx, gg, x_rescale=2, y_rescale=0.5)
{% endhighlight %}
x0:  20.0004 +/- 0.0459     (0.2%)  initial:21<br>
y0:  30.0435 +/- 0.0455     (0.2%)  initial:32<br>
sx:  5.0078  +/- 0.0398     (0.8%)  initial:8<br>
sy:  4.9602  +/- 0.0394     (0.8%)  initial:5
{% include _images/{{ page.date | date: '%Y-%m-%d'}}/gauss2d.svg%}

---------------

{% highlight python %}
import fitwrap as fw
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

def chirped(t, off, amp, freq, phase):
    return  off + amp * np.sin(2*np.pi*(freq + t*4)*t + phase)

def chirped2(t, off, amp, freq, phase):
    return  off + amp * np.sin(2*np.pi*(freq)*t + phase)

NOISE_LEVEL = 0.2
xx = np.random.random(3000)
yy = chirped(xx, 0.5, 2, 6, np.pi*0) + (np.random.random(xx.shape)-0.5)*NOISE_LEVEL
yy = yy + chirped2(xx, 0.5, 2, 20, 2)


xmin = np.min(xx)
xmax = np.max(xx)
span = 0.2
n_bins = 40
x_bins = np.linspace(xmin+span, xmax-span, n_bins)

freq_1d, lomb_spectrum_1d = fw.lomb_spectrum(xx, yy, frequency_span=[1,30], grid_size=1000)

spectrogram = np.zeros([1000, n_bins])

for index, x_bin in enumerate(x_bins):
    mask = np.logical_and((x_bin-span)<=xx, (x_bin+span)>=xx) 
    frequency_grid, lombscargle_spectrum = fw.lomb_spectrum(xx[mask], yy[mask],
                                                            frequency_span=[1,30], grid_size=1000)
    spectrogram[:, index] = lombscargle_spectrum


fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(12,3))
ax1.plot(*zip(*sorted(zip(xx, yy), key=lambda x: x[0])))
ax2.plot(freq_1d, lomb_spectrum_1d)
ax3.imshow(spectrogram, aspect='auto', extent=[x_bins[0],x_bins[-1],
            frequency_grid[0],frequency_grid[-1]], origin='lower') 
{% endhighlight %}
{% include _images/{{ page.date | date: '%Y-%m-%d'}}/lomb.svg%}











