---
layout: post
title:  "Bayesian optimization"
date:   2019-05-08 07:00:00 +0100
categories: machine_learning
---

## Why Bayesian optimization
Bayesian optimization can be used for the global optimization of black-box function with no need to compute derivatives. In particular it can be exploited to efficiently optimize the hyperparameters defining a neural network model and its training. Since the training of the NN can be very time consuming, it is important to explore the hyperparameters space wisely. Solution like [grid search](https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e) or [random search](https://en.wikipedia.org/wiki/Random_search) can be extremely costy and may not be feasible. [Bayesian optimization](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f) is often a much better option, since it takes into account all the previously tested configurations to select the optimal next point to test, choosing the one that maximize the expected information gain.

## Bayesian optimization in code
Lets give a look at how Bayesian optimization works in practice by using [this python library](https://github.com/fmfn/BayesianOptimization) by Fernando Nogueira.

We start by defining our black-box function, which will be completely unknow to the algorithm:

```python
import numpy as np
import matplotlib.pyplot as plt

def black(x, y):
    sine_2d = np.sin(x*y*0.002)
    gaussian_2d = np.exp(-(x-50)**2/2000 -(y-70)**2/2000)*2
    return sine_2d + gaussian_2d + np.random.random(x.shape)/3

yy, xx = np.mgrid[0:100, 0:150]
zz = black(xx, yy)

plt.imshow(zz, origin=[0,0])
```
<p style="text-align:center;"><img src="/asset/images/2019-05-08/black_box.png" alt="black box" height="250" width="350"></p>

Our goal is to find the maximum of this function (most yellow region) by probing the function the least number of times. Lets set up the optimizer:
```python
optimizer = BayesianOptimization(
    f=black,
    pbounds={'x': (0, 150), 'y': (0, 100)},
    verbose=2,
    random_state=30)

utility = UtilityFunction(kind="ei", xi=0.0, kappa=None)
```
The utility function defines the criteria to follow in order to pick the next point to probe. In this case we used `ei` which stands for Expected improvement. Lets run the optimization algorithm and see what happens:
```python
for index in range(100):
    next_point_to_probe = optimizer.suggest(utility)
    target = black(**next_point_to_probe)
    optimizer.register(params=next_point_to_probe, target=target)
    # plot the result of the optimizer ...
```
![expected improvement](/asset/images/2019-05-08/ei.gif)
The black cross shows the current point being tested by the algorithm, the white dots represent all the previously tested point. The red dot shows the best point tested so far. The goal of the algorithm is to place the red dot on top of the most yellow region as fast as possible, without getting stuck in local maxima.

On the top left panel we plot our black-box function, which represents the ground truth we aim for. On the top right panel we plot the current best prediction of the function given the previously probed points (white dots). As we can see, the more point are acquired, the more faithful to the real black-box function is our prediction.

On the bottom left panel we plot the variance, which quantifies the uncertainty of our model in the parameter space. In the blue regions the variance is low, meaning that we are more confident about the goodness of our model. On the other hand, yellow regions correspond to high variace, i.e. high uncertainty regions. 