---
layout: post
title:  "Numpy array intersection"
date:   2017-03-07 23:00:00 +0100
categories: python
---

{% highlight python %}
def merge_arrays(x_np_array, y_np_array):
    from functools import reduce
    intersection = reduce(np.intersect1d, x_np_array)
    return_list = [intersection]
    for x, y in zip(x_np_array, y_np_array):
        return_list.append(np.array([y[np.where(i==x)[0]][0] for i in intersection]))
    return return_list
{% endhighlight %}

Example:
{% highlight python %}
a = np.array([1, 2, 3])
aa = np.array([11, 22, 33])

b = np.array([1, 3, 7, 5])
bb = np.array([2, 6, 14, 10])

c = np.array([3, 9, 8, 7, 1])
cc = np.array([3, 1, 1, 1, 1])

merge_arrays([a, b, c], [aa, bb, cc])
{% endhighlight %}
[array([1, 3]), array([11, 33]), array([2, 6]), array([1, 3])]
