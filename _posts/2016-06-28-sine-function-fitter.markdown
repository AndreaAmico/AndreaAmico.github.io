---
layout: post
title:  "Sine function fitter"
date:   2016-06-28 17:00:21 +0100
categories: data_analysis
---

Sine function fitter using the [lmfit](https://lmfit.github.io/lmfit-py/ "https://lmfit.github.io/lmfit-py/") module.
The initial parameters guess is estimated in the following way:

**Offset and amplitude:** Calculate the mean of the higher and lower *max_min_estimation_size (default=5)* data points. Half of the diference 
gives an estimation for the amplitude and the mean gives an estimation for the offset.

**Frequency:** An estimation for the frequency is obtained using the [lombscargle algorithm](https://en.wikipedia.org/wiki/Least-squares_spectral_analysis "https://en.wikipedia.org/wiki/Least-squares_spectral_analysis") with a *frequency_grid_size (default=100)* comb of frequency. The minim frequecy considered is given by the inverse of the total time *np.max(x_data) - np.min(x_data)*, while the maximum frequency is equal to the minimum frequency times the number of *x* data point.


{% highlight python %}
def fit_sine(x_data, y_data, frequency_grid_size=100, max_min_estimation_size=5, report=True, plot=True):
    """ NB: Requires numpy, lmfit, scipy and matplotlib if plot=True
    Fit datapoint with a sine function of the form off+amp*sin(freq*x+phase)
    The initial guess of the frequency is obtained using the lombscargle algorithm
    Returns the output of lmfit.minimize
    """

    import scipy.signal as signal
    import numpy as np
    import lmfit
    
    y_data_sorted = np.sort(y_data)
    y_data_min = np.mean(y_data_sorted[:max_min_estimation_size])
    y_data_max = np.mean(y_data_sorted[-max_min_estimation_size:])
    y_data_offset = (y_data_min + y_data_max)/2

    y_data_centred = y_data - y_data_offset

    total_time = np.max(x_data) - np.min(x_data)
    min_frequency = 1/total_time
    normval = x_data.shape[0]
    max_frequency = min_frequency*normval

    frequency_grid = np.linspace(min_frequency, max_frequency, frequency_grid_size)
        
    pgram = signal.lombscargle(x_data, y_data_centred, frequency_grid)

    lombscargle_spectrum = np.sqrt(4*(pgram/normval))
    maximum_indices = np.where(lombscargle_spectrum==np.max(lombscargle_spectrum))    
    central_frequency = frequency_grid[maximum_indices[0]]

    def sin_function(params, x):
        amp = params['amp'].value
        pshift = params['phase'].value
        freq = params['frequency'].value
        off = params['offset'].value
        
        return amp * np.sin(x * freq  + pshift) + off

    def residual(params, x, data):
        model = sin_function(params, x)
        return (data - model)

    params = lmfit.Parameters()
    params.add('amp', value=(y_data_max-y_data_min)/2.)
    params.add('offset', value=y_data_offset)
    params.add('phase', value=0.0, min=-np.pi/2., max=np.pi/2.)
    params.add('frequency', value=central_frequency, min=0)

    out = lmfit.minimize(residual, params, args=(x_data, y_data))
    
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,2, figsize=(12, 4))
        ax[0].scatter(frequency_grid, lombscargle_spectrum)
        ax[0].axvline(central_frequency, c="green") 
        ax[0].set_xlabel("x_data")
        ax[0].set_ylabel("Lomb Scargle spectrum")    

        x_plot = np.linspace(np.min(x_data), np.max(x_data), 100)
        ax[1].scatter(x_data, y_data)
        ax[1].plot(x_plot, sin_function(out.params, x_plot), c="green")
        ax[1].set_xlabel("x_data")
        ax[1].set_ylabel("y_data")       
        
        # fig.savefig("out.svg")
        plt.show()   

    if report: lmfit.report_fit(out, show_correl=False)

    return out
{% endhighlight %}

Example
----------------------------

{% highlight python %}
off = 5
freq = 3
amp = 2
phi = 1

x = np.linspace(0,6,50)
noise = (np.random.random(x.shape[0])-0.5)
data = np.cos(freq*x* + phi)*amp + off + noise*3

out = fit_sine(x, data)
{% endhighlight %}
Figure output:

{% include _images/{{ page.date | date: '%Y-%m-%d'}}/sine-fitter-output.svg%}

Text output:
{% highlight python %}
[[Fit Statistics]]
    # function evals   = 56
    # data points      = 50
    # variables        = 4
    chi-square         = 38.981
    reduced chi-square = 0.847
    Akaike info crit   = -0.279
    Bayesian info crit = 7.369
[[Variables]]
    amp:         1.94390338 +/- 0.186334 (9.59%) (init= 2.721456)
    offset:      5.17910117 +/- 0.134196 (2.59%) (init= 5.24784)
    phase:       1.57079593 +/- 0.193860 (12.34%) (init= 0)
{% endhighlight %}
