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
        s = params['sigma'].value
        x0 = params['peak_position']
        off = params['offset']    
        return off + amp * np.exp(np.float64(-(x-x0)**2/(2*sigma**2)))

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

        # fig.savefig("out.svg")
        plt.show()   

    if report: lmfit.report_fit(out, show_correl=False)
        
    return out
{% endhighlight %}

Example
----------------------------


