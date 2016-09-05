---
layout: post
title:  "File operations"
date:   2016-09-05 10:30:00 +0100
categories: python
---

{% highlight python %}
import os

my_dir = os.path.join(os.path.curdir, 'next', 'myfiles')
if not os.path.isdir(my_dir): os.makedirs(my_dir)

my_path = os.path.join(my_dir, 'file.txt')
os.path.split(my_path)
{% endhighlight %}
('./next/myfiles', 'file.txt')

{% highlight python %}
my_file = open(my_path, 'w')
my_file.write('I am the first line')
my_file.write(', so am I.\n')
my_file.write('I am the second.\n')
my_file.close()
my_file = open(my_path, 'a')
my_file.write('Sry I was late.')
my_file.close()

my_file = open(my_path)
content = my_file.read()
my_file.close()
print(content)
{% endhighlight %}
I am the first line, so am I.
I am the second.
Sry I was late.

{% highlight python %}
my_file = open(my_path)
line_1 = my_file.readline()
line_2 = my_file.readline()
my_file.close()
print(line_1)
print(line_2)
{% endhighlight %}
I am the first line, so am I.

I am the second.

{% highlight python %}
my_file = open(my_path)
lines = my_file.readlines()
my_file.close()
print(lines)
{% endhighlight %}
['I am the first line, so am I.\n', 'I am the second.\n', 'Sry I was late.']

{% highlight python %}
import glob
glob.glob(my_dir+"/*.txt")
{% endhighlight %}
['./next/myfiles/file.txt']
