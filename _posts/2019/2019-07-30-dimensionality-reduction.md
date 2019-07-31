---
layout: post
title:  "Dimensionality reduction"
date:   2019-07-30 22:00:00 +0100
categories: machine_learning
---

### Summary

```python
## Fit + transform available
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA

## Only fit_transform
from sklearn.manifold import MDS
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE

## Both X and y must be provided
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

## n_clusters instead of n_components
from sklearn.cluster import FeatureAgglomeration
```

in the following we plot some examples of dimensionality reduction to plot three different datasets in two dimensions. Moreover, we score the goodness of each algorithm for the given dataset using a simple SVM classifier with cross validation:
```python
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
def get_score(X_2D, y):
    clf = SVC(gamma=2, C=1)
    scores = cross_val_score(clf, X_2D, y, cv=5)
    return f'{scores.mean():.2f} (+/- {scores.std()*2:.2f})'
``` 
### Iris dataset (sklearn)
<p style="text-align:center;"><img src="/asset/images/dimensionality_reduction/reduction_iris_1.svg" alt="reduction_iris_1" width="800"></p>
<p style="text-align:center;"><img src="/asset/images/dimensionality_reduction/reduction_iris_2.svg" alt="reduction_iris_2" width="800"></p>
<p style="text-align:center;"><img src="/asset/images/dimensionality_reduction/reduction_iris_3.svg" alt="reduction_iris_3" width="800"></p>

### Wine dataset (sklearn)
<p style="text-align:center;"><img src="/asset/images/dimensionality_reduction/reduction_wine_1.svg" alt="reduction_wine_1" width="800"></p>
<p style="text-align:center;"><img src="/asset/images/dimensionality_reduction/reduction_wine_2.svg" alt="reduction_wine_2" width="800"></p>
<p style="text-align:center;"><img src="/asset/images/dimensionality_reduction/reduction_wine_3.svg" alt="reduction_wine_3" width="800"></p>

### Digits dataset (sklearn)
<p style="text-align:center;"><img src="/asset/images/dimensionality_reduction/reduction_digits_1.svg" alt="reduction_digits_1" width="800"></p>
<p style="text-align:center;"><img src="/asset/images/dimensionality_reduction/reduction_digits_2.svg" alt="reduction_digits_2" width="800"></p>
<p style="text-align:center;"><img src="/asset/images/dimensionality_reduction/reduction_digits_3.svg" alt="reduction_digits_3" width="800"></p>


## Principal Component Analysis (PCA) for dummies
Project all the points along one direction and measure the variance of the projections position. The direction of the first component is chosen by maximizing this variance. The same idea is followed to choose the remaining components, with the constraint to be orthogonal to the previous ones. Mathematically it can be done by solving the eigenvalues problem using the *Singular Value Decomposition* and selecting as the most important directions the eigenvectors corresponding to the largest eigenvalues:
```python
from sklearn import datasets
data = datasets.load_iris()
y = data.target
X = data.data
```
```python
# X = (U @ np.diag(S) @ V)
# S are the eigenvalues in descending order
U, S, V = np.linalg.svd(X, full_matrices=False)

n_components = 2
U = U[:, :n_components]
U = U * S[:n_components]
```

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X)
```

<p style="text-align:center;"><img src="/asset/images/dimensionality_reduction/simple_PCA.svg" alt="PCA" width="800"></p>
