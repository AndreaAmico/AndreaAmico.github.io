---
layout: post
title:  "Matplotlib plot bucket"
date:   2017-01-22 14:00:00 +0100
categories: data_analysis
---

A bunch of random example to customise a [matplotlib](http://matplotlib.org/ "http://matplotlib.org/") scatter plot with errorbars.



{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker

# %matplotlib inline # Use it in Jupyter

################# DATA ####################
x1 = np.random.random(10)*2
y1 = x1**2
err1 = y1/5

x2 = np.linspace(0,3*np.pi,30)
y2 = np.sin(x2 - np.pi/2) + 1
err2_x = np.random.random(30)
err2_y = np.random.random(30)
###########################################

def sort_by_first(x, y):
    xy_T = np.array([x, y]).T
    xy_sorted = xy_T[np.argsort(xy_T[:,0])].T
    return xy_sorted[0], xy_sorted[1]

def sort_by_column(data_array, column):
    data_array_T = np.array(data_array).T
    return data_array_T[np.argsort(data_array_T[:,column])].T

title_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal','verticalalignment':'bottom'}
axis_font = {'fontname':'Arial', 'size':'16', 'color':(0.2,0.2,0.2,1)}
font_prop = font_manager.FontProperties(fname='C:\Windows\Fonts\Arial.ttf', size=14)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(14)

x1, y1, err1 = sort_by_column([x1,y1,err1],0)

ax.errorbar(x1, y1, err1, fmt='h', markerfacecolor=(0.5,0.5,1,0.7), markeredgecolor="midnightblue",
            markersize=10, label = 'Parabola',color=(0.5,0.5,1,0.7), fillstyle='full',
            markeredgewidth=1.0, linewidth=2,capsize=0)

ax.plot(x2, y2, color="green", linestyle=':', linewidth=2 )
ax.errorbar(x2, y2, err2_x, err2_y, fmt='H', color='green', alpha=0.5, markersize=10, label = 'Sin', capsize=0)
ax.grid(True)
ax.set_title('Remember to set a good title', **title_font)
ax.set_xlabel('More ore less the x axis', **axis_font)

ax.xaxis.set_major_locator(ticker.MultipleLocator(np.pi))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(np.pi/4))

ax.set_ylabel('The mighty Y axis', **axis_font)
ax.set_yscale('log')

ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
plt.legend(loc='lower right', prop=font_prop, numpoints=1);
{% endhighlight %}


{% include _images/{{ page.date | date: '%Y-%m-%d'}}/bucket.svg%}
