---
layout: post
title:  "String format"
date:   2016-06-29 11:10:00 +0100
categories: python
---


Recap of `.format()` method ([here](https://pyformat.info/ "https://pyformat.info/") for a more detailed description).
*NB: Multiple spaces are represented with underscores.*

-----------------------------
{% highlight python %}
'{:4d}'.format(42)
{% endhighlight %}
"__42"

-----------------------------
{% highlight python %}
'{:06.2f}'.format(3.141592653589793)
{% endhighlight %}
"003.14"

-----------------------------
{% highlight python %}
'{:.2e}'.format(100000)
{% endhighlight %}
"1.00e+05"

-----------------------------
{% highlight python %}
'{}, {}'.format('one', 'two')
{% endhighlight %}
"one, two"

-----------------------------
{% highlight python %}
'{1}, {0}'.format('one', 'two')
{% endhighlight %}
"two, one"

-----------------------------
{% highlight python %}
'{:>10}'.format('test')
{% endhighlight %}
"______test"

-----------------------------
{% highlight python %}
'{:10}'.format('test')
{% endhighlight %}
"test______"

-----------------------------
{% highlight python %}
'{:<{}s}'.format('test', 8)
{% endhighlight %}
"test____"

-----------------------------
{% highlight python %}
'{:^10}'.format('test')
{% endhighlight %}
"__ test __"

