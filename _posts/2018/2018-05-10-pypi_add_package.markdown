---
layout: post
title:  "Add package to pypi"
date:   2018-05-10 22:00:00 +0100
categories: other
---


1. Register on [test.pypi.org](https://test.pypi.org) and [pypi.org](https://pypi.org)

2. Create .pypirc file and put it in ~/.pypirc (or windows10 C:\User\$username$)
(under unix set the right permission "chmod 600 ~/.pypirc")
__.pypirc__:
{% highlight Plain Text %}
[distutils]
index-servers =
  pypi
  pypitest

[pypi]
username=your_username
password=your_password

[pypitest]
repository=https://test.pypi.org/legacy/
username=your_username
password=your_password
{% endhighlight %}


3. Upload your package to github with the following files:

__setup.py__:
{% highlight python %}
from distutils.core import setup
setup(
  name = 'my-package-name',
  packages = ['my-package-name'],
  version = '0.1',
  description = 'package description',
  author = 'Name LastName',
  author_email = 'gihub-username@gmail.com',
  url = 'https://github.com/gihub-username/my-package-name',
  download_url = 'https://github.com/gihub-username/my-package-name/archive/0.1.tar.gz',
  keywords = ['whatever', 'you', 'want'],
  classifiers = [],
)
{% endhighlight %}


__setup.cfg__:
{% highlight Plain Text %}
[metadata]
description-file = README.md
{% endhighlight %}


__README.md__:
{% highlight Markdown %}
## to do
{% endhighlight %}


__gitignore.txt__:
{% highlight Plain Text %}
*.pyc
*.DS_Store
*.log
dist
MANIFEST
{% endhighlight %}

__LICENSE.txt__: (MIT license example - change year and name in the first line)
{% highlight Plain Text %}
Copyright (c) year Name LastName

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
{% endhighlight %}

a folder with the package name with a __\_\_init\_\_.py__ file inside.

5. Upload everything on github (with a tag):
{% highlight Plain Text %}
git tag 0.0.1 -m "Here are my updates"
{% endhighlight %}

{% highlight Plain Text %}
git push --tags origin master
{% endhighlight %}


6. run: python setup.py sdist upload -r pypitest
7. run: python setup.py sdist upload -r pypi