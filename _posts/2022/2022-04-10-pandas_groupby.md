---
layout: post
title:  "Pandas groupby"
date:   2022-04-18 20:00:00 +0100
categories: data-analysis
---

## Create a dummy dataset
```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame(np.random.normal(0, 2, size=(200,4)), columns=['A', 'B', 'C', 'D'])

df = (df
        .assign(A_cat = lambda df: np.round(df.A).astype('str')).drop('A', axis=1)
        .assign(B_cat = lambda df: np.round(df.B)).drop('B', axis=1)
)

df.head()
```
<p style="text-align:center;"><img src="/asset/images/pandas/groupby_dummy.PNG" alt="dummy dataset" hight="300"></p>

## Filtering
Select data such that grouping by `A_cat` will result in groups of 20 or more elements:
```python
df = df.groupby('A_cat').filter(lambda df:len(df)>=20)
```

Select data such that grouping by `B_cat` will result in groups having low standard dev on variable `C`:
```python
df = df.groupby('B_cat').filter(lambda df:df.C.std() < 2)
```

## Transformations
Create column `C_std` storing the standard dev of variable `C` withing the `B_cat` groups:
```python
df = df.assign(C_std = lambda df:df.groupby('B_cat').C.transform('std'))
```


## Aggregations
Aggregate `A_cat` groups by calculating the mean of the `C` columns and the standard error of the mean of the `D` column:
```python
df.groupby('A_cat').agg(
    C_mean = ('C', np.mean),
    D_sem = ('D', 'sem')
)
```
<p style="text-align:center;"><img src="/asset/images/pandas/groupby_agg.PNG" alt="agg" hight="300"></p>

For custom aggregations, which need mixing more columns we can use the `apply` function:
```python
def custom_agg(df):
    C_mean = df.C.mean()
    D_std = df.D.std()
    return pd.Series(dict(
        C_std_D = C_mean/D_std,
        C_mean = C_mean
    ))

df.groupby('A_cat').apply(custom_agg)
```
<p style="text-align:center;"><img src="/asset/images/pandas/groupby_apply.PNG" alt="apply" hight="300"></p>

## Utilities
```python
df.groupby('A_cat').ngroups           # get number of groups
df.groupby('A_cat').groups            # get dictionary of data grouped
df.groupby('A_cat').size()            # get the size of each group
df.groupby('A_cat').groups.keys()     # get the groups name
df.groupby('A_cat').get_group('1.0')  # get a single group by name
```

## Multiple grouping
```python
df.groupby(['A_cat', 'B_cat']).apply(custom_agg).head()
```

<p style="text-align:center;"><img src="/asset/images/pandas/groupby_double.PNG" alt="apply" hight="400"></p>

```python
df.groupby(['A_cat', 'B_cat']).apply(custom_agg).unstack().head()
```

<p style="text-align:center;"><img src="/asset/images/pandas/groupby_unstack.PNG" alt="apply" width="800"></p>

```python
df.groupby(['A_cat', 'B_cat']).apply(custom_agg).swaplevel(axis='index').unstack().head()
```

<p style="text-align:center;"><img src="/asset/images/pandas/groupby_swap_unstack.PNG" alt="apply" width="800"></p>
