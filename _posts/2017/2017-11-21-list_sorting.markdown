---
layout: post
title:  "Sorting two lists"
date:   2017-11-20 11:00:00 +0100
categories: fast_copy_paste
---

Zip plus sorted allows to order two list at once:

{% highlight python %}
np.random.seed(4)
a, b = list(np.random.randint(5, size=(2, 8)))

plt.scatter(a, b, zorder=10)
a, b = zip(*sorted(zip(a, b), key=lambda x: x[1]))
plt.plot(a, b, alpha=0.5, zorder=5)

a, b = zip(*sorted(zip(a, b), key=lambda x: x[0]))
plt.plot(a, b, alpha=0.8, zorder=4)

# if a and b need to be lists then
a, b = (list(t) for t in zip(*sorted(zip(a, b), key=lambda x: x[0])))
{% endhighlight %}

{% include _images/{{ page.date | date: '%Y-%m-%d'}}/zorder.svg%}
