---
layout: post
title:  "fitwrap module"
date:   2018-03-03 11:00:00 +0100
categories: data-analysis
---


### How to install

The fitwrap module is available on [github](https://github.com/AndreaAmico/fitwrap "https://github.com/AndreaAmico/fitwrap") and can be installed via pip:
{% highlight console %}
pip install fitwrap
{% endhighlight %}

This module provides a wrapper for the [scipy](https://www.scipy.org/ "https://www.scipy.org/") function `optimize.curve_fit()`. The full fitting procedure is reduced in one-line command, moreover the initial fitting parameters and boundaries are set by the keyword arguments of the model fitting function. The package comes with the following functions:
- `fit`: non linear 1D fit
- `fit2d`: non linear 2D fit
- `fit_sin`: sinusoidal fit with automatic fit function and initial guess.

-------

## Usage examples
Import the dependencies we need:
```python
import matplotlib.pyplot as plt
import numpy as np
import fitwrap as fw
```

### Simple fit
Lets start with a simple fit of a parabolic function.

#### Generate data with black box function
```python
np.random.seed(42)
xx = np.linspace(-2,10, 30)
yy = 0.1*(xx-5)**2 + 0.5*(xx-5) + np.random.normal(size=xx.shape)*0.5

fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111, xlabel="x", ylabel="y")
ax.scatter(xx, yy)
#fig.savefig('img/simple_data.svg')
```
<p style="text-align:center;"><img src="/asset/images/fitwrap/simple_data.svg" alt="simple data" width="450"></p>


#### Define model function and fit
The first variable of the model function is the independent variable (in our case `x`). The following arguments will be used as fit parameters and will be initialized to 1 is not specified.

To fit the data with the model function it is sufficient to call the **fit** method: `fw.fit(model_function, xx, yy)`.

The function will print the fit results, specifying the best fitted value, the absolute error, the relative error and the initial guess for each of the fit parameters. It will plot the **input data** in green, the **best fit curve** and the **confidence interval** corresponding to the confidence probability of 0.95.

```python
def model_function(x, x0, a2, a1):
    return a2*(x-x0)**2 + a1*(x-x0)

fit_out = fw.fit(model_function, xx, yy)
```

```text
x0:  0.18    +/- 0.191    (106.1%)  initial:1
a2:  0.11061 +/- 0.00668    (6.0%)  initial:1
a1:  -0.5875 +/- 0.0555    (-9.4%)  initial:1
```
<p style="text-align:center;"><img src="/asset/images/fitwrap/simple_fit.svg" alt="simple fit" width="550"></p>


#### Initial guess for the fitting parameters
In order to set an initial guess for the initial parameters, pass the initial guess as default parameters in the model function. Keep in mind to place the initial guess to the **most right** arguments of the model function.
```python
# Good
def model_function(x, x0, a2=0.5, a1=0.2):
    return a2*(x-x0)**2 + a1*(x-x0)

# Syntax Error
def model_function(x, x0=3, a2=0.5, a1):
    return a2*(x-x0)**2 + a1*(x-x0)
```
```text
  File "<ipython-input-130-3f5a2423c6cd>", line 6
    def model_function(x, x0=3, a2=0.5, a1):
                      ^
SyntaxError: non-default argument follows default argument
```


## Fixed arguments
It is possible to fix an agument to exclude it from the fitting parameters using the special keyword **fixed_args**. Set it as the last parameter of the model function and pass the list of the name of the fitting parameters you want to fix.

#### Generate a new black box model
```python
np.random.seed(42)
t_decay = np.linspace(0,20, 30)
y_decay = 0.1 + np.exp(-t_decay/2) + np.random.normal(size=t_decay.shape)*0.02

fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111, xlabel="time", ylabel="y")
ax.scatter(t_decay, y_decay)
```
<p style="text-align:center;"><img src="/asset/images/fitwrap/simple_decay.svg" alt="simple decay" width="450"></p>


Lets define the model function by fixing the offset of the exponential decay to 0. We expect the fit to be bad since our black box model has an offset of `0.1`.

```python
def model_function(t, off=0, tau=5, amp=1, fixed_args=['off']):
    return off + amp*np.exp(-t/tau)

fit_out_fixed = fw.fit(model_function, t_decay, y_decay)
```
```text
off:  0       Fixed
tau:  3.055   +/- 0.247      (8.1%)  initial:5
amp:  1.0271  +/- 0.0527     (5.1%)  initial:1
```
<p style="text-align:center;"><img src="/asset/images/fitwrap/fixed_offset.svg" alt="fixed offset" width="550"></p>


## Bounded parameters

To fix the boundaries of a fitting parameter one can set its default value in the model definition as the tuple:
**(initial guess, lower bound, upper bound)**

Lets set the offset to be bounded between `0.2` and `0.5`. We expect again the fit function to fail.
```python
def model_function(t, off=(0.4, 0.2, 0.5), tau=5, amp=1):
    return off + amp*np.exp(-t/tau)

fit_out_bound = fw.fit(model_function, t_decay, y_decay)
```
```text
off:  0.2     +/- 0.0207    (10.3%)  initial:(0.4, 0.2, 0.5)
tau:  1.505   +/- 0.27      (17.9%)  initial:5
amp:  0.9371  +/- 0.0882     (9.4%)  initial:1
```
<p style="text-align:center;"><img src="/asset/images/fitwrap/bounded_offset.svg" alt="bounded offset" width="550"></p>



## Consider errors on data points
It is possible to consider errors on the `y` variable by passing an array of sigmas to the fit function.

#### Generate a new black box model
Let us now consider two different types of errors: amplitude error (left plot) and offset error (right plot).
```python
np.random.seed(42)
t_decay = np.linspace(0,10, 30)
y_decay = 0.1 + np.exp(-t_decay/2) + np.random.normal(size=t_decay.shape)*0.03
y_decay_err_1 =  (0.01 + np.exp(-t_decay/2))/2
y_decay_err_2 =  1/(y_decay_err_1*1e3)


fig = plt.figure(figsize=(10,3))
ax1 = fig.add_subplot(121, xlabel="time", ylabel="y")
ax1.errorbar(t_decay, y_decay, y_decay_err_1, fmt='o')

ax2 = fig.add_subplot(122, xlabel="time", ylabel="y")
ax2.errorbar(t_decay, y_decay, y_decay_err_2, fmt='o')
```
<p style="text-align:center;"><img src="/asset/images/fitwrap/simple_decay_err.svg" alt="decay err data" width="1000"></p>



Lets fit the two datasets including the different errors.

```python
def model_function(t, off=0.1, tau=5, amp=1):
    return off + amp*np.exp(-t/tau)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))

print('Error on amplitude:')
fit_out_err_1 = fw.fit(model_function, t_decay, y_decay, 
         sigma=y_decay_err_1, fig_ax=(fig, ax1))

print('\nError on offset:')
fit_out_err_2 = fw.fit(model_function, t_decay, y_decay,
        sigma=y_decay_err_2, fig_ax=(fig, ax2))
```
<p style="text-align:center;"><img src="/asset/images/fitwrap/decay_errors.svg" alt="decay err" width="1000"></p>



------------



## Example of automatic sinusoidal fit

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

## Example of 2D fit

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

## Example of lomb spectrum

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











