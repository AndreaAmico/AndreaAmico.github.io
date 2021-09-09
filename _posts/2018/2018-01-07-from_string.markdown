---
layout: post
title:  "String to np.array"
date:   2018-01-20 22:00:00 +0100
categories: data-analysis
---

Take a string in the format:

data = '''

1   4

2   8.2

3 12

4     15,16

'''

and return the following list of numpy arrays:
[np,array([1, 2, 3, 4]),
np.array(4, 8.2, 12, 15.16)]



{% highlight python %}
def from_string(data):
    import re
    data = re.sub(',','.', data)
    data = re.sub(r'\t',' ', data)
    data = re.sub(' +',' ', data)
    columns = []
    for line in data.splitlines():
        try:
            columns.append([float(x) for x in line.split(' ')])         
        except ValueError as e:
            pass
    return [np.array(raw) for raw in zip(*columns)]
{% endhighlight %}



Example:
{% highlight python %}
data = '''

1  1
2  4.02
3  9
4  16.3
5 25

'''

x, y = from_string(data)
print(x - np.sqrt(y))
{% endhighlight %}

[ 0.         -0.00499377  0.         -0.03732585  0.        ]


## Using pandas
If we need a pandas dataframe instead, we can exploit the `StringIO` method from the standard `io` library:
{% highlight python %}
import pandas as pd
from io import StringIO

data = '''
x,y
1,1
2,4.02
3,9
4,6.3
5,25
'''

df = pd.read_csv(StringIO(data), sep=",")
{% endhighlight %}
