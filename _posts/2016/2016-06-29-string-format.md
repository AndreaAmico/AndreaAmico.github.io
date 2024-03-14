---
layout: post
title:  "String format"
date:   2016-06-29 11:10:00 +0100
categories: python
---


Recap of `.format()` method ([here](https://pyformat.info/ "https://pyformat.info/") for a more detailed description).


-----------------------------
```python
f'{1000000000:_}'  # {:,} gives similar results with comma
```

"1_000_000_000"
-----------------------------

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
f'{"test":>10}'  # ^ to center align, < to left align
```
"______test"

-----------------------------

```python
f'{"test":*>10}'
```
"******test"

-----------------------------
```python
from datetime import datetime

now: datetime = datetime.now()
f'{now:%d-%m-%y (%H.%M.%S)}'
```
"14-03-24 (08.08.10)"

-----------------------------

```python
a: int = 2
f'{a + a = }'
```
"a + a = 4"

