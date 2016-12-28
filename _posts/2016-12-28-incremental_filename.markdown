---
layout: post
title:  "Incremental filename"
date:   2016-12-28 9:30:00 +0100
categories: python
---


{% highlight python %}
from glob import glob
import os
import re

def increment_filename(path, filename, starting_number=0):
    files = glob(os.path.join(path, filename))
    if files:
        files.sort()
        new_count = int(re.findall('(?<=\()(\d+)(?=\)\.\S+$)', os.path.basename(files[-1]))[-1]) + 1
        new_complete_path = re.sub('(?<=\()(\d+)(?=\)\.\S+$)', str(new_count), files[-1])
    else:     
        new_complete_path = os.path.join(path, re.sub('(?<=\()(\*)(?=\)\.\S+$)', str(starting_number), filename))
    return new_complete_path
{% endhighlight %}

{% highlight python %}
new_path = increment_filename(path="my_path", filename="my_file(*).txt")

f = open(new_path, 'w')
f.writelines("I am a really nice file!!!")
f.close()
{% endhighlight %}
