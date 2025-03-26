---
layout: post
title:  "Visual Exploration"
date:   2025-03-25 22:00:00 +0100
categories: data-visualization
---

# Explore filling of a dataframe quantity with respect to two keys
### Function definition
```python
import pandas as pd
import plotly.express as px
import numpy as np

def plot_quantity_check(df, x, y, z, zmin=0, zmax=0.5, normalization='median',
                        template="plotly_dark", width=800, height=600,z_var_name=None):
    df = df.copy()
    norm_mapping = dict(
        median = np.nanmedian,
        none = np.ones_like
    )
    z_var_name = z_var_name if z_var_name else f'filling({z})'
    df[z_var_name] = df.groupby(y)[z].transform(lambda x: x/norm_mapping[normalization](x))
    df_plot = df.pivot(index='item_id', values=z_var_name, columns=x)
    fig = px.imshow(df_plot, labels=dict(x=x, y=y, color=z_var_name),
                    color_continuous_scale='Emrld',template=template, zmin=zmin, zmax=zmax)
    fig.update_layout(width=width, height=height)
    return fig
```

### Example dataset preparation:
```python
dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='ME')
df = pd.concat([
    pd.DataFrame({
    'item_id': [chr(item_id)]*len(dates),
    'date': dates,
    'quantity': np.random.randint(1, 15, len(dates))
}) for item_id in range(ord('a'), ord('a') + 10)])

df.sample(3)
```
```
| item_id | date       | quantity |
|---------|------------|----------|
| a       | 2025-07-31 |          |
| g       | 2025-11-30 | 10.0     |
| d       | 2025-06-30 | 1.0      |
| g       | 2025-04-30 |          |

```

### Generate plot
Visualize plot if in notebook, and save it as .html file:
```python
dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='ME')
df = pd.concat([
    pd.DataFrame({
    'item_id': [chr(item_id)]*len(dates),
    'date': dates,
    'quantity': np.random.randint(1, 15, len(dates))
}) for item_id in range(ord('a'), ord('a') + 10)])

df['quantity'] = df['quantity'] * np.random.choice([np.nan,1], p=[0.7, 0.3], size=len(df))

fig = plot_quantity_check(df, x='date', y='item_id', z='quantity', zmax=4, normalization='none',
                           width=600, height=400, z_var_name='Quantity',template="plotly")

fig.write_html('./myplot.html')
fig
```

{% include asset/images/data-exploration/fillingExploration.html %}