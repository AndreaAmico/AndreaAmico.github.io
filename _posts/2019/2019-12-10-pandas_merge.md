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

### Merging by index
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



