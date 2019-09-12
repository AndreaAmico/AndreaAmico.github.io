---
layout: post
title:  "ML sklearn"
date:   2019-09-11 22:00:00 +0100
categories: machine_learning
---

```python
import time
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

classifiers = [
    [KNeighborsClassifier(3), 'KNeighbors'],
    [SVC(kernel="linear", C=0.025), 'SVC_linear'],
    [SVC(gamma=2, C=1), 'SVC'],
    [DecisionTreeClassifier(max_depth=5), 'DecisionTree'],
    [RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
        'RandomForest'],
    [MLPClassifier(alpha=1, max_iter=1000), 'MPL'],
    [AdaBoostClassifier(), 'AdaBoost'],
    [GaussianNB(), 'GaussianNB'],
    [QuadraticDiscriminantAnalysis(), 'QuadraticDiscriminantAnalysis']]

for clf, clf_name in classifiers:
    t0 = time.time()
    clf.fit(X_train, y_train)
    metric = accuracy_score(y_test, clf.predict(X_test))*100
    print(f'{clf_name}: {metric:.1f}%  Completed in {time.time()-t0:.2f}s')
```
