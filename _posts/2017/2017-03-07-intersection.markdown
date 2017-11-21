---
layout: post
title:  "List and numpy array intersection"
date:   2017-03-07 23:00:00 +0100
categories: python
---

Merging numpy arrays:

{% highlight python %}
def intersect_arrays(x_np_array, y_np_array):
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

intersect_arrays([a, b, c], [aa, bb, cc])
{% endhighlight %}
[array([1, 3]), array([11, 33]), array([2, 6]), array([1, 3])]


-----------------

Merging lists:

{% highlight python %}
def intersect_list(list_x1, list_y1, list_x2, list_y2):
    try: new_list_x = sorted(set(list_x1) & set(list_x2))
    except: new_list_x = list(set(list_x1) & set(list_x2))
    new_list_y1 = []
    new_list_y2 = []
    for x in new_list_x:
        new_list_y1.append(list_y1[list_x1.index(x)])
        new_list_y2.append(list_y2[list_x2.index(x)])
    return new_list_x, new_list_y1, new_list_y2
{% endhighlight %}