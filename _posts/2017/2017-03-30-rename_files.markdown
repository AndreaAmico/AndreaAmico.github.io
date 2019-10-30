---
layout: post
title:  "Rename files"
date:   2017-04-19 22:00:00 +0100
categories: fast-copy-paste
---

In the folder *".\my_folder\subfolder\subsubfolder\"* there are files named:
- fileA_1.txt
- fileA_11.txt
- fileA_13.txt

I want to rename them changing **A** to **B**:

{% highlight python %}
import os, glob
import re
file_path = os.path.abspath('my_folder//subfolder\subsubfolder/fileA_*.txt')
prev = 'file'
after = '_\d+\.txt$'
old = 'A'
new = 'B'

mach_string = '(?<={}){}(?={})'.format(prev, old, after)
for single_file in glob.glob(file_path):
    os.rename(single_file, re.sub(mach_string, new, single_file))
{% endhighlight %}


`os.path.abspath` is used to normalize the path (independently of the OS).

`glob.glob` returns the list af all file paths matching the input string.

`A(?=_\d+\.txt$)` looks for a **A** that preceeds a string of the type *"_##ANY_NUMBER##.txt"*.

`(?<=file)A` checks that **A** is preceeded by the string *"file"*.

