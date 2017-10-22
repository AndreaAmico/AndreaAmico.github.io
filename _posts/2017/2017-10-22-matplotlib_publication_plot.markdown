---
layout: post
title:  "Publication plot"
date:   2017-10-22 11:00:00 +0100
categories: data_analysis
---


{% highlight python %}
single_column = {
    'axes.labelsize': 5,
    'axes.titlesize':6,
    'font.size': 5,
    'legend.fontsize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'lines.linewidth' : 0.5,
    'lines.markeredgewidth' : 0.5,
    'lines.markersize' : 3,
    'font.family' : 'sans-serif',
    'font.sans-serif' : 'Arial',
    'pdf.fonttype' : 42,
    'mathtext.fontset' : 'stix',

    'axes.linewidth' :  0.5,     # edge linewidth
    'axes.grid' : True,   # display grid or not

    ### TICKS
    'xtick.top' : True,   # draw ticks on the top side
    'xtick.bottom' : True,   # draw ticks on the bottom side
    'xtick.major.size' : 1.4,      # major tick size in points
    'xtick.minor.size' : 0.8,      # minor tick size in points
    'xtick.major.width' : 0.5,    # major tick width in points
    'xtick.minor.width' : 0.3,    # minor tick width in points

    'xtick.direction' : 'in',    # direction: in, out, or inout
    'xtick.minor.visible' : False,  # visibility of minor ticks on x-axis
    'xtick.major.top' : True,   # draw x axis top major ticks
    'xtick.major.bottom' : True,   # draw x axis bottom major ticks
    'xtick.minor.top' : False,   # draw x axis top minor ticks
    'xtick.minor.bottom' : False,   # draw x axis bottom minor ticks

    'ytick.left' : True,   # draw ticks on the left side
    'ytick.right' : True,  # draw ticks on the right side
    'ytick.major.size' : 1.4,      # major tick size in points
    'ytick.minor.size' : 0.8,      # minor tick size in points
    'ytick.major.width' : 0.5,    # major tick width in points
    'ytick.minor.width' : 0.3,    # minor tick width in points
    'ytick.direction' : 'in',    # direction: in, out, or inout
    'ytick.minor.visible' : False,  # visibility of minor ticks on y-axis
    'ytick.major.left' : True,   # draw y axis left major ticks
    'ytick.major.right' : True,   # draw y axis right major ticks
    'ytick.minor.left' : False,   # draw y axis left minor ticks
    'ytick.minor.right' : False,   # draw y axis right minor ticks

    'figure.facecolor' : 'None',   # figure facecolor; 0.75 is scalar gray
    'figure.edgecolor' : 'None',   # figure edgecolor
    'axes.facecolor' : 'None',
    'grid.linewidth' : 0.5,       # in points
    'grid.alpha' : 0.4,       # transparency, between 0.0 and 1.0
    'backend' : 'pdf'
    }


import matplotlib.pyplot as plt
import numpy as np

with plt.rc_context(single_column):
    fig = plt.figure(figsize=[s/25.5 for s in (32,28)], dpi=141)  # pdi of the screen to show
                                                                  # the real size of the plot
    ax = fig.add_axes([0,0,1,1]) # set the axes to be the full size of the figure
    
    ax.plot(np.sin(np.linspace(0,2*np.pi,100)))
    ax.scatter(np.sin(np.linspace(0,np.pi,30))*100, np.sin(np.linspace(0,2*np.pi,30)))

    ax.set_xlabel('x Label')
    ax.set_ylabel('y label')
    ax.set_title('Such a good title')

    fig.savefig(r'test.pdf', bbox_inches="tight")

{% endhighlight %}

{% include _images/{{ page.date | date: '%Y-%m-%d'}}/test.svg%}
