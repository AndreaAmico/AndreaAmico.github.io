---
layout: post
title:  "Train generator"
date:   2020-12-09 18:00:00 +0100
categories: other
---


One of the most flexible ways to train machine learning models is by feeding the training data to the `fit` function via a python generator. This method has several advantages, it allows to preprocess the data in a customized way for every training loop (e.g. data augmentation), it allows to automatically deal with the batch size and the shuffling of the data for different epochs, and finally, it can be used to use gradually the training dataset if its size does not allow to load everything into the RAM.

## Create a data generator
This snippet can be used as a blueprint to create the most suitable data generator for a given project.

```python
import random
import numpy as np

def data_generator(batch_size, data_x, data_y, shuffle=True):

    data_size = len(data_x)
    index_list = [*range(data_size)]
    
    if shuffle:
        random.shuffle(index_list)
    
    index = 0
    
    while True:
        batch_x, batch_y = [], []

        for i in range(batch_size):
            if index >= data_size:
                index = 0
                if shuffle:
                    random.shuffle(index_list)
            
            batch_x.append(data_x[index_list[index]])
            batch_y.append(data_y[index_list[index]])
        
            index += 1
        
        yield((batch_x, batch_y))
```

### Example
To show how the generator works we can create a simple dataset where `X` and `Y` are ordered lists of alphabet letters, lowercase, and uppercase respectively.

```python
X = [chr(i) for i in np.arange(ord('a'), ord('a')+26)] # lowercase letters
Y = [chr(i) for i in np.arange(ord('A'), ord('A')+26)] # uppercase letter
```

We can create the data generator with a `batch_size` of `3`:
```python
my_data_gen = data_generator(batch_size=3, data_x=X, data_y=Y, shuffle=True)
```

The generator can be finally fed to a `fit` method to train a machine learning model. We can see how it works with a simple for loop:
```python
for i in range(5):
    x, y = next(my_data_gen)
    print(f'Batch number {i}:  x = {x}   y = {y}')
```
output:
```text
Batch number 0:  x = ['z', 'k', 'n']   y = ['Z', 'K', 'N']
Batch number 1:  x = ['p', 'e', 'j']   y = ['P', 'E', 'J']
Batch number 2:  x = ['x', 'i', 'g']   y = ['X', 'I', 'G']
Batch number 3:  x = ['d', 'v', 'f']   y = ['D', 'V', 'F']
Batch number 4:  x = ['m', 'a', 'q']   y = ['M', 'A', 'Q']
```
