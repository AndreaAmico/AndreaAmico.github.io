---
layout: post
title:  "Pandas merge"
date:   2019-12-10 22:00:00 +0100
categories: data-analysis
---


```python
df_left = pd.DataFrame(dict(x = ['A', 'B', 'C']), index=[0,1,2])
df_right = pd.DataFrame(dict(y = ['a', 'b', 'c']), index=[2,3,4])
```

### Merging on indexes
```python
pd.merge(left=df_left, right=df_right, left_index=True, right_index=True)
```

<p style="text-align:center;"><img src="/asset/images/pandas/merge/index.svg" alt="index" width="800"></p>

### Inner, outer, left, right
```python
pd.merge(left=df_left, right=df_right, left_index=True, right_index=True, how='inner')
```
<p style="text-align:center;"><img src="/asset/images/pandas/merge/index.svg" alt="index" width="800"></p>

```python
pd.merge(left=df_left, right=df_right, left_index=True, right_index=True, how='outer')
```
<p style="text-align:center;"><img src="/asset/images/pandas/merge/outer.svg" alt="index" width="800"></p>

```python
pd.merge(left=df_left, right=df_right, left_index=True, right_index=True, how='left')
```
<p style="text-align:center;"><img src="/asset/images/pandas/merge/left.svg" alt="index" width="800"></p>

```python
pd.merge(left=df_left, right=df_right, left_index=True, right_index=True, how='right')
```
<p style="text-align:center;"><img src="/asset/images/pandas/merge/right.svg" alt="index" width="800"></p>



--------------------------------------------


```python
df_left = pd.DataFrame(dict(x=[0,1,2], y=['a','b','c'], z=[0,10,20]), index=[4,5,6])
df_right = pd.DataFrame(dict(x=[2,1,5], y=['C','B','A']), index=[7,8,9])
```

### Merging on columns

```python
pd.merge(left=df_left,right=df_right, left_on='x', right_on='x', suffixes=['_L','_R'])
```
<p style="text-align:center;"><img src="/asset/images/pandas/merge/suffix.svg" alt="index" width="800"></p>


-------------------------


### Indicator
When merging with `how` option others then 'inner' one can use `indicator=True` flag to add a new column to the data frame describing for each row if it was merged using `both` left and right data frames, `left_only` or `right_only`.



------------------------------


### Validate
`validate` parameter can be used to *assert* the type of relation between the two merged tables:
 - “one_to_one” or “1:1”: assert if merge keys are unique in both left and right datasets.
 - “one_to_many” or “1:m”: assert if merge keys are unique in the left dataset.
 - “many_to_one” or “m:1”: assert if merge keys are unique in the right dataset.
 - “many_to_many” or “m:m”: allowed, but does not result in checks.


```python
df_left = pd.DataFrame(dict(x=[0,1,2], y=['a', 'b', 'c']), index=[4,5,6])
df_right = pd.DataFrame(dict(x=[2,1,1], y=['C', 'B', 'A']), index=[7,8,9])
pd.merge(left=df_left, right=df_right, left_on='x', right_on='x', validate='1:1')
```

would result in a **MergeError**.
