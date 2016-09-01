---
layout: post
title:  "2D gaussian function fitter"
date:   2016-09-01 9:30:20 +0100
categories: data_analysis
---

Gaussian function fitter using the [lmfit](https://lmfit.github.io/lmfit-py/ "https://lmfit.github.io/lmfit-py/") module.
The initial parameters guess is estimated in the following way:



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


def fit_gaussian_2d(data, report=True, plot=True):
    """ NB: Requires numpy, lmfit,and matplotlib if plot=Truie
    function form: off + amp * np.exp(-(xy[0]-x0)**2/(2*sx**2))*np.exp(-(xy[1]-y0)**2/(2*sy**2))
    variable name: ['amplitude', 'offset', 'peak_position', 'sigma']
    to extract parameters from the output use: out.params["amplitude"].value        
    """
    
    import numpy as np
    import lmfit
    
    out_x = fit_gaussian(np.arange(data.shape[1]), np.mean(data, 0), report=False, plot=False)
    out_y = fit_gaussian(np.arange(data.shape[0]), np.mean(data, 1), report=False, plot=False)

    x0 = out_x.params["peak_position"].value
    y0 = out_y.params["peak_position"].value

    sigma_x = out_x.params["sigma"].value
    sigma_y = out_y.params["sigma"].value

    offset = (out_x.params["offset"].value + out_y.params["offset"].value)/2
    amplitude = (out_x.params["amplitude"].value + out_y.params["amplitude"].value)/2


    def gauss_function_2d(params, xy):
        amp = params['amplitude'].value
        off = params['offset'].value
        sx = params['sigma_x'].value
        sy = params['sigma_y'].value
        x0 = params['x0'].value
        y0 = params['y0'].value

        return off + amp * np.exp(-(xy[0]-x0)**2/(2*sx**2))*np.exp(-(xy[1]-y0)**2/(2*sy**2))

    def residual(params, xy, data):
        model = gauss_function_2d(params, xy)
        return (data - model)

    xy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

    params = lmfit.Parameters()
    params.add('amplitude', value=amplitude)
    params.add('offset', value=offset)
    params.add('x0', value=x0)
    params.add('y0', value=y0)
    params.add('sigma_x', value=sigma_x)
    params.add('sigma_y', value=sigma_y)

    out = lmfit.minimize(residual, params, args=(xy, data))
    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits import axes_grid1

        def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
            """Add a vertical color bar to an image plot."""
            divider = axes_grid1.make_axes_locatable(im.axes)
            width = axes_grid1.axes_size.AxesY(im.axes, aspect=1/aspect)
            pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
            current_ax = plt.gca()
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.sca(current_ax)
            return im.axes.figure.colorbar(im, cax=cax, **kwargs)

        fig = plt.figure(figsize = (10,5))
        im = plt.imshow(data, cmap="viridis")
        add_colorbar(im)
        plt.contour(xy[0], xy[1], gauss_function_2d(out.params, xy), 10, colors='w', alpha=0.4)

        #plt.savefig("out.svg")
        plt.show()   
    if report: lmfit.report_fit(out, show_correl=False)
{% endhighlight %}

Example
----------------------------

{% highlight python %}
import numpy as np

xy = np.meshgrid(np.arange(100), np.arange(50))
off = 2
amp = 3
x0 = 30
y0 = 20
sx = 15
sy = 7
noise = np.random.random(xy[0].shape)
data =  off + amp * np.exp(-(xy[0]-x0)**2/(2*sx**2))*np.exp(-(xy[1]-y0)**2/(2*sy**2)) + noise

fit_gaussian_2d(data)
{% endhighlight %}

Figure output:

{% include _images/{{ page.date | date: '%Y-%m-%d'}}/gauss_2d_fitter_output.svg%}

Text output:
{% highlight python %}
[[Fit Statistics]]
    # function evals   = 31
    # data points      = 5000
    # variables        = 6
    chi-square         = 412.380
    reduced chi-square = 0.083
    Akaike info crit   = -12458.235
    Bayesian info crit = -12419.132
[[Variables]]
    amplitude:   2.98708598 +/- 0.022672 (0.76%) (init= 1.063371)
    offset:      2.51082804 +/- 0.005526 (0.22%) (init= 2.510599)
    x0:          30.0533144 +/- 0.113046 (0.38%) (init= 30.10914)
    y0:          19.9996817 +/- 0.052732 (0.26%) (init= 19.9722)
    sigma_x:     14.8597938 +/- 0.126158 (0.85%) (init= 14.94474)
    sigma_y:     6.99834827 +/- 0.058263 (0.83%) (init= 7.036165)
{% endhighlight %}

