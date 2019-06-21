---
layout: post
title:  "Incremental filename"
date:   2016-12-28 9:30:00 +0100
categories: python
---

`increment_filename` looks in `path` for the file *filename.ext*  or *filename(#any number).ext* and return:

- *filename.ext*: if nothing is found
- *filename(2).ext*: if only *filename.ext* is found
- *filename(**n+1**).ext*: if *filename(**n**).ext* is found (with **n** being the maximum in the folder)

{% highlight python %}
from glob import glob
import os
import re

def increment_filename(path, filename, ext):
    regex_pattern = '(?<=\()(\d+)(?=\)\.'+ext+'$)'
    files = glob(os.path.join(path, filename+"(*)."+ext))
    if files:
        files.sort()
        new_count = int(re.findall(regex_pattern, os.path.basename(files[-1]))[-1])+1
        new_complete_path = re.sub(regex_pattern, str(new_count), files[-1])
    else:
        single = glob(os.path.join(path, filename+'.'+ext))
        if single:
            new_complete_path = os.path.join(path, filename+'(2).'+ext)
        else:
            new_complete_path = os.path.join(path, filename+'.'+ext)
    return new_complete_path
{% endhighlight %}

{% highlight python %}
new_path = increment_filename(path="my_path", filename="my_file", ext="txt")

f = open(new_path, 'w')
f.writelines("I am a really nice file!!!")
f.close()
{% endhighlight %}
