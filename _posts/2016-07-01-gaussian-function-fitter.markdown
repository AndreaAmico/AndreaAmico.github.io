---
layout: post
title:  "Gaussian function fitter"
date:   2016-07-01 12:30:20 +0100
categories: python
---

Gaussian function fitter using the [lmfit](https://lmfit.github.io/lmfit-py/ "https://lmfit.github.io/lmfit-py/") module.
The initial parameters guess is estimated in the following way:

**Amplitude:** the estimated value for the absolute value of the amplitude is given by the absolute value
 of the difference between the maximum and the minimun Y data point.

 **Sign of the amplitude:** in order to guess the sign of the amplitude, the absolute value of the difference between the mean
 of the Y value of the most left and most right data point, and the minimum Y value is compared to the same difference but with the maximum Y value instead `(np.abs((min_x[1] + max_x[1])/2 - min_y[1]) < np.abs((min_x[1] + max_x[1])/2 - max_y[1]))`. If the second term is bigger, than probably the amplitude sign is positive, if not than it is probably negative.

 **Peak position and offset:** If the amplitude sign is positive the X position of the biggest Y data point estimate the position of the peak and the Y value of the smallest Y data point estimates the offset. If the amplitude sign is negative it is the opposite.

 **Sigma:** the initial guess for the sigma is obtained estimating the Full Width Half Maximum first. In order to do this the data points are splitted in two, the ones on the left and the ones on the right with respect to the peak position. Then one finds the X position of the point that has the Y value closer to the half amplitude value, both for the left and the right side. The X position difference between those two points estimates the FWHM. To obtain the sigma one divedes this value by `(2*np.sqrt(2 * np.log(2))`.
 


{% highlight python %}
def fit_gaussian(x_data, y_data, report=True, plot=True):
    """ NB: Requires numpy, lmfit,and matplotlib if plot=True
        function form: off + amp * np.exp(-(x-x0)**2/(2*sigma**2))
        variable name: ['amplitude', 'offset', 'peak_position', 'sigma']
        to extract parameters from the output use: out.params["amplitude"].value
    """
    import numpy as np
    import lmfit

    data_array = np.array([x_data, y_data]).T
    x_sorted_data = data_array[np.argsort(data_array[:,0])]
    y_sorted_data = data_array[np.argsort(data_array[:,1])]

    min_y = y_sorted_data[0, :]
    max_y = y_sorted_data[-1, :]

    min_x = x_sorted_data[0, :]
    max_x = x_sorted_data[-1, :]


    if np.abs((min_x[1] + max_x[1])/2 - min_y[1]) < np.abs((min_x[1] + max_x[1])/2 - max_y[1]):
        amp_sign = 1
        peak_position = max_y[0]
        offset = min_y[1]
    else:
        amp_sign = -1
        peak_position = min_y[0]
        offset = max_y[1]

    amplitude = amp_sign*np.abs(max_y[1] - min_y[1])

    right_side = x_sorted_data[x_sorted_data[:, 0]>peak_position, :]
    left_side = x_sorted_data[x_sorted_data[:, 0]<peak_position, :]

    right_side[:, 1] = np.abs(right_side[:, 1] - offset - amplitude/2)
    right_fwhm = right_side[np.where(right_side[:, 1]==np.min(right_side[:, 1]))[0],0]

    left_side[:, 1] = np.abs(left_side[:, 1] - offset - amplitude/2)
    left_fwhm = left_side[np.where(left_side[:, 1]==np.min(left_side[:, 1]))[0],0]

    sigma = ((right_fwhm - left_fwhm) / (2*np.sqrt(2 * np.log(2))))[0]
    
    def gauss_function(params, x):
        amp = params['amplitude'].value
        sigma = params['sigma'].value
        x0 = params['peak_position'].value
        off = params['offset'].value
        return off + amp * np.exp(-(x-x0)**2/(2*sigma**2))

    def residual(params, x, data):
        model = gauss_function(params, x)
        return (data - model)

    params = lmfit.Parameters()
    params.add('amplitude', value=amplitude)
    params.add('offset', value=offset)
    params.add('peak_position', value=peak_position)
    params.add('sigma', value=sigma)

    out = lmfit.minimize(residual, params, args=(data_array[:, 0], data_array[:, 1]))

    if plot:
        import matplotlib.pyplot as plt
        x_plot = np.linspace(np.min(x_data), np.max(x_data), 100)
        plt.scatter(x_data, y_data)
        plt.plot(x_plot, gauss_function(out.params, x_plot), c="green")
        plt.xlabel("x_data")
        plt.ylabel("y_data")       

#         plt.savefig("out.svg")
        plt.show()

    if report: lmfit.report_fit(out, show_correl=False)
        
    return out

{% endhighlight %}

Example
----------------------------

{% highlight python %}
off = 1
amp = -2
x0 = 2
sigma = 1

noise = np.random.random(100)-0.5
x_data = (np.random.random(100)-0.5)*10
y_data = off + amp*np.exp(-(x_data-x0)**2/(2*sigma**2)) + noise

fit_gaussian(x_data, y_data)
{% endhighlight %}

Figure output:

{% include _images/{{ page.date | date: '%Y-%m-%d'}}/gauss-fitter-output.svg%}

Text output:
{% highlight python %}
[[Fit Statistics]]
    # function evals   = 33
    # data points      = 100
    # variables        = 4
    chi-square         = 32.849
    reduced chi-square = 0.342
    Akaike info crit   = -99.243
    Bayesian info crit = -88.822
[[Variables]]
    amplitude:      -2.43280018 +/- 0.198035 (8.14%) (init=-3.902941)
    offset:          0.97912407 +/- 0.084108 (8.59%) (init= 1.957917)
    peak_position:   2.00695831 +/- 0.072144 (3.59%) (init= 2.21673)
    sigma:           0.87205535 +/- 0.088285 (10.12%) (init= 1.938921)
{% endhighlight %}

