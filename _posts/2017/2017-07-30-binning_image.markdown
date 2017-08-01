---
layout: post
title:  "Binning image"
date:   2017-07-30 22:00:00 +0100
categories: data_analysis
---


{% highlight python %}
def bin_image(image, binsize_x=1, binsize_y=None, aggregation_function=np.sum):
    sy, sx = image.shape
    if not binsize_y:
        binsize_y = binsize_x
        
    y_bins, x_bins = sy // binsize_y, sx // binsize_x
    crop = np.ogrid[(sy % binsize_y)//2: sy-((sy % binsize_y)//2+(sy%binsize_y)%2),
                   (sx % binsize_x)//2: sx-((sx % binsize_x)//2+(sx%binsize_x)%2)]
    cropped = image[crop]
    x_agg = aggregation_function(cropped.reshape(cropped.shape[0], x_bins, binsize_x), axis=2)
    return aggregation_function(x_agg.reshape(y_bins, binsize_y, x_agg.shape[1]), axis=1)
{% endhighlight %}

Example:
{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

yx = np.mgrid[:124, :300]
def f(yx):
    return np.exp(-(yx[0]-60)**2/300)*np.sin(-(yx[1]/20))+np.random.random(yx[0].shape)*0.2

image = f(yx)

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].imshow(image)
ax[1].imshow(bin_image(image, binsize_x=10, binsize_y=7))
plt.show()
{% endhighlight %}

{% include _images/{{ page.date | date: '%Y-%m-%d'}}/binning_image.svg%}
