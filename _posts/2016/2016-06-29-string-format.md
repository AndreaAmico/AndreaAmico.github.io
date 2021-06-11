---
layout: post
title:  "String format"
date:   2016-06-29 11:10:00 +0100
categories: python
---


Recap of `.format()` method ([here](https://pyformat.info/ "https://pyformat.info/") for a more detailed description).
**NB: Multiple spaces are represented with underscores.**

-----------------------------
```python
f'{42:4d}'
```

"__42"
-----------------------------
```python
f'{3.141592653589793:06.2f}'
```
"003.14"

-----------------------------
```python
f'{100000:.2e}'
```
"1.00e+05"

-----------------------------

```python
f'{'test':>10}'
```
"______test"

-----------------------------
```python
f'{'test':10}'
```
"test______"

-----------------------------
```python
f'{test:^10}'
```
"__ test __"
