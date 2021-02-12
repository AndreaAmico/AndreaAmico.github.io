---
layout: post
title:  "Feature importance"
date:   2021-02-11 20:00:00 +0100
categories: ml-tools
---

In a classification scenario, feature importance estimation can be helpful to understand which variables in our dataset are more useful to predict the target class. This allows both to simplify our model and to avoid features that can only introduce noise in the classification task.

First, we will download some sample data from the [sklearn](https://scikit-learn.org/stable/) library, then we will show two different feature importance estimation methods: the first by exploiting the random forest algorithm, and the latter by using the permutation importance method and a ridge classifier.


### Load sample data
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data
y = data.target
y_names = np.array(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
 
### Random forest feature importance
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

sort_index = np.argsort(clf.feature_importances_)
fig, ax = plt.subplots(1, figsize=(10, 2), dpi=100)
plt.barh(y=np.array(y_names[sort_index]),
         width=clf.feature_importances_[sort_index])
ax.set(xticks=[], xlabel='Feature importance')
[ax.spines[pos].set_visible(False) for pos in ('right', 'bottom', 'top')];
```
<p style="text-align:center;"><img src="/asset/images/data-exploration/random_forest_features.png" alt="random forest features" width="800"></p>


### Permutation importance with ridge regression
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance

model = Ridge().fit(X_train, y_train)

r = permutation_importance(model, X_test, y_test)
importance_index = np.argsort(r.importances_mean)

fig, ax = plt.subplots(1, figsize=(10, 2), dpi=100)
plt.barh(y=np.array(y_names[importance_index]),
         width=r.importances_mean[importance_index],
         xerr=r.importances_std[importance_index])
ax.set(xticks=[], xlabel='Feature importance')
[ax.spines[pos].set_visible(False) for pos in ('right', 'bottom', 'top')];
```
<p style="text-align:center;"><img src="/asset/images/data-exploration/ridge_features.png" alt="random forest features" width="800"></p>


