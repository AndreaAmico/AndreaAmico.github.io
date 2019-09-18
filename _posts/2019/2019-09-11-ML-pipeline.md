---
layout: post
title:  "ML pipeline"
date:   2019-09-11 23:00:00 +0100
categories: machine_learning
---

### Shuffling and splitting
```python
from sklearn.utils import shuffle
x, y = shuffle(x, y, random_state=42)

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import numpy as np
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
```
