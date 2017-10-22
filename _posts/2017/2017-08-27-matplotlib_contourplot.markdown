---
layout: post
title:  "Tkinter mouse events"
date:   2017-07-12 08:30:00 +0100
categories: python-GUI
---

{% highlight python %}

fig, ax = plt.subplots(1, figsize=(6,4))
ratio = (zi_int/zi_non_int)#/zi_non_int
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    ratio[ratio<1] = 1


CS = plt.contour(xi,yi,ratio,10,linewidths=0.5,colors='white', alpha=0.3)
CF = plt.contourf(xi,yi,ratio,10,cmap='magma')#, vmin=0, vmax=3)#'bwr'

cbar = plt.colorbar()
# plt.scatter(np.array(d['mean_N']).flatten(),np.array(d['time']).flatten(),marker='H',c='white',s=2)

plt.xlim(123,20,6)
plt.ylim(100,2850)
plt.xlabel('Atom number')
plt.ylabel('Interaction time ($\mu s$)')
cbar.set_label('Var(N)$_{1.4}$ / Var(N)$_{0.4}$')
plt.yscale('log')
plt.show()
{% endhighlight %}