---
layout: post
title:  "Data scaling"
date:   2019-07-25 22:00:00 +0100
categories: machine_learning
---

### Normalization
Scale all the values between two extremes
```python
X = np.random.uniform(low=30, high=50, size=[10000, 1])

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
```
<p style="text-align:center;"><img src="/asset/images/preprocessing/normalization.png" alt="normalization" width="700"></p>


### Standardization
Scale all the values to have zero means and unitary standard deviation
```python
X = np.random.normal(loc=30, scale=50, size=[10000, 1])

from sklearn import preprocessing
scaler = skl.preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
```
<p style="text-align:center;"><img src="/asset/images/preprocessing/standardization.png" alt="standardization" width="700"></p>


### Box-cox transformation
*Gaussianizes* a distribution and standardizes it. It works only with positive values.

```python
X = np.random.exponential(scale=100, size=[10000, 1])

from sklearn import preprocessing
scaler_box = preprocessing.PowerTransformer(method='box-cox', standardize=True)
scaler_box.fit(X)
X_scaled_box = scaler_box.transform(X)
```
<p style="text-align:center;"><img src="/asset/images/preprocessing/box-cox.png" alt="box-cox" width="700"></p>

### Yeo-Johnson transformation
Similar effect to the *Box-cox transformation* but can be used for negative values as well.

```python
SHIFT = 5
X = np.random.lognormal(mean=2, sigma=0.6, size=[10000, 1]) - SHIFT

from sklearn import preprocessing
scaler = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
scaler.fit(X)
X_scaled = scaler.transform(X)
```
<p style="text-align:center;"><img src="/asset/images/preprocessing/yeo-johnson.png" alt="yeo-johnson" width="700"></p>


## Code for plots
```python
import matplotlib.pyplot as plt
from sklearn import preprocessing

X = np.random.normal(loc=30, scale=50, size=[10000, 1])
scaler = skl.preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,3))
for ax in (ax1, ax2):
    ax.set_yticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax1.text(30, 0, 'BEFORE', size=30, color=color(8), horizontalalignment='center', verticalalignment='bottom')
ax2.text(0, 0, 'AFTER', size=30, color=color(8), horizontalalignment='center', verticalalignment='bottom')
ax1.hist(X, bins=30, density=True, color=color(0))
ax2.hist(X_scaled, bins=30, density=True, color=color(0));
# fig.savefig('./plots/standardization.png',  bbox_inches='tight')
```
