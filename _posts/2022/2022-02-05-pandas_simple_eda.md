---
layout: post
title:  "Pandas simple EDA"
date:   2022-02-05 21:30:00 +0100
categories: data-analysis
---

The following is a simple script to perform a quick EDA of a Pandas dataframe. For a more complete analysis one can use one of the following libraries:

- [dataprep](https://docs.dataprep.ai/index.html#): collect, explore and clean all in one tool.
- [pandasprofiling](https://github.com/ydataai/pandas-profiling): perfect for html reports.
- [sweetviz](https://pypi.org/project/sweetviz/): best for dataset comparison.

--------------------

In the following code we `df` is the pandas dataframe to be analyzed and `df_out` is the output eda dataframe.

```python
import numpy as np
import pandas as pd

## df is the dataframe to analyze

df_c = df.convert_dtypes()
fullest_row = df_c.isna().mean(axis=1).argmin()
cat_threshold = 50
info_list = []

for column_name, s in df_c.items():
    info_dict = {}

    n_unique = s.nunique()
    is_cat = n_unique < cat_threshold # check if is categorical data
    is_num = pd.api.types.is_numeric_dtype(s.dtype)
    is_int = pd.api.types.is_integer_dtype(s.dtype)
    is_str = pd.api.types.is_string_dtype(s.dtype)
    is_date = pd.api.types.is_datetime64_any_dtype(s.dtype)

    info_dict['name'] = column_name
    info_dict['nan percentage'] = s.isna().mean()
    info_dict['value example'] = s[fullest_row]
    info_dict['number of unique values'] = n_unique

    if is_cat:
        info_dict['data type'] = 'categorical'
        if n_unique > 0:
            info_dict['1st frequent'] = s.value_counts().index[0]
        else:
            info_dict['1st frequent'] = None
        if n_unique > 1:
            info_dict['2nd frequent'] = s.value_counts().index[1]

    if is_num and not is_cat:
        info_dict['data type'] = 'numerical'
        info_dict['mean value'] = np.mean(s)
        info_dict['10 percentile'] = s.quantile(0.1)
        info_dict['50 percentile'] = s.quantile(0.5)
        info_dict['90 percentile'] = s.quantile(0.9)

    if is_str and not is_cat:
        info_dict['data type'] = 'string'
        info_dict['average number of character'] = np.mean(s.str.len())
    
    if is_date and not is_cat:
        info_dict['data type'] = 'date'
        info_dict['min_date'] = s.min()
        info_dict['max_date'] = s.max()
        info_dict['tot_time'] = s.max()-s.min()
        info_dict['avg_day_entry'] = len(s)/(s.max()-s.min()).days

    info_list.append(info_dict)

df_out = pd.DataFrame(info_list)
```