---
layout: post
title:  "File operations"
date:   2016-09-05 10:30:00 +0100
categories: python
---

```python
import os

my_dir = os.path.join(os.path.curdir, 'next', 'myfiles')
if not os.path.isdir(my_dir): os.makedirs(my_dir)

my_path = os.path.join(my_dir, 'file.txt')
os.path.split(my_path)
```
('./next/myfiles', 'file.txt')

```python
with open(my_path, 'w') as my_file: # a for append
	my_file.write('I am the first line')
	my_file.write(', so am I.\n')
	my_file.write('I am the second.\n')

with open(my_path, 'r') as my_file:
	content = my_file.read()

print(content)
```
I am the first line, so am I.
I am the second.
Sry I was late.

```python
with open(my_path, 'r') as my_file:
	line_1 = my_file.readline()
	line_2 = my_file.readline()

print(line_1)
print(line_2)
```
I am the first line, so am I.

I am the second.

```python
with open(my_path, 'r') as my_file:
	lines = my_file.readlines()

print(lines)
```
['I am the first line, so am I.\n', 'I am the second.\n', 'Sry I was late.']

```python
import glob
glob.glob(my_dir+"/*.txt")
```
['./next/myfiles/file.txt']
