---
layout: post
title:  "Data exploration"
date:   2019-05-20 23:00:00 +0100
categories: data_analysis
---
### Correlation matrix

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df = sns.load_dataset('titanic')

corr = df.corr()
corr.values[np.triu_indices_from(corr)] = np.abs(corr.values[np.triu_indices_from(corr)] )

fig = plt.figure(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, cmap=sns.diverging_palette(200, 15, as_cmap=True), vmin=-1, vmax=1)
```
<p style="text-align:center;"><img src="/asset/images/data-exploration/corr.svg" alt="correlation plot" width="500"></p>

### Crosstab

```python
df = sns.load_dataset('titanic')
survived_count = pd.crosstab(index=[df.sex], columns=[df.survived])

fig, ax = plt.subplots(1, figsize=(6,4))
survived_count.plot.bar(stacked=False, ax=ax)
```
<p style="text-align:center;"><img src="/asset/images/data-exploration/crosstab.svg" alt="correlation plot" width="400"></p>