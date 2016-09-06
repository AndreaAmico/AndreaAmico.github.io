---
layout: post
title:  "Text progressbar"
date:   2016-09-06 16:30:00 +0100
categories: python
---

{% highlight python %}
def progress_bar(current_value, max_value):
    progress = ((current_value + 1)/max_value)*100
    print('\r[{0}{1}] {2:.1f}%'.format('#'*int(progress/2), ' '*(50-int(progress/2)), progress), end='')

for index, item in enumerate(items):
    # DO YOUR THINGS
    progress_bar(index, len(items))
{% endhighlight %}

