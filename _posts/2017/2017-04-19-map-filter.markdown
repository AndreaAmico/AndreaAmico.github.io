---
layout: post
title:  "Map & filter vs list comprehensions"
date:   2017-04-19 23:00:00 +0100
categories: python
---

`Map` applies a given function to a list. The same effect can be achieved with a list comprehensions. 

NB: it returns a map object that can be converted back to a list, or used as a generator.

```python
my_list = [0, 0, 7]

def foo(x):
    return (x+1)**2

mapped_list = list(map(foo, my_list))

# or with a lambda function:
mapped_list = list(map(lambda x:(x+1)**2, my_list))

# or with a list comprehensions:
mapped_list = [foo(x) for x in my_list]

print(mapped_list)
```
Output:
*[1, 1, 64]*

--------------------------------------

`Filter` takes a function and a list as parameters, and returns a filter object (that can be converted into a list or used as a generator) that contains the values of the input list for which the input function returns `True`.
```python
my_list = [1, 2, 90890, 5, 8, 234253, 9]

filtered_list = filter(lambda x:x<100, my_list)

# or with a list comprehensions:
filtered_list = [x for x in my_list if x<100]


print(*filtered_list, sep=' - ')
```
Output:
*1 - 2 - 5 - 8 - 9*