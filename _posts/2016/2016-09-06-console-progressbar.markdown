---
layout: post
title:  "Text progressbar"
date:   2016-09-06 16:30:00 +0100
categories: python
---

{% highlight python %}
def progress_bar(current_value, max_value, size=50):
    prog = (current_value+1)/max_value
    left = '#'*int(prog * size + 0.5) # 0.5 to round up when casting to int
    right = '-'*(size-len(left))
    print('\r[{}{}] {:.1f}%'.format(left, right, prog*100), end='')

{% endhighlight %}


{% highlight python %}  
for index, item in enumerate(items):
    # DO YOUR THINGS
    progress_bar(index, len(items))
{% endhighlight %}

