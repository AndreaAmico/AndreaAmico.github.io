---
layout: post
title:  "Denoise with autoencoders"
date:   2019-06-24 22:00:00 +0100
categories: machine_learning
---


## Autoencoders

An autoencoder is a NN structure used to extract a determined number of important feature from data, using unsupervised learning. The idea is to learn an efficient representation of the data (i.e. encoding) which has a much smaller dimension when compared with the original data itself. The working principle is simple: one creates a symmetric neural network with a information bottleneck in the middle and train it using the same dataset both as the input and output of the network. This way, the model learns to recreate the full data using only the small amount of information defined in the bottleneck of the neural network. Moreover, the model can be trained to be robust to noise by artificially introducing noise in the input of the network.

The main uses of autoencoders include data compression, feature extraction, denoise, and creation of generative models.

Here's the autoencoder scheme from wiki:
<p style="text-align:center;"><img src="/asset/images/autoencoder/autoencoder_structure.png" alt="autoecoder structure" height="300"></p>

### Denoise application

Let's see how one can create a simple convolutional network to encode the information of the handwritten digit 

TODO ...



# Loading mnist digit dataset

```python
from keras.datasets import mnist
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
```





