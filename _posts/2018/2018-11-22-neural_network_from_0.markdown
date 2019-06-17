---
layout: post
title:  "Neural Network from 0"
date:   2018-11-22 22:00:00 +0100
categories: machine_learning
---

Lets build a simple neural network from scratch in pure python plus numpy and finally train it in a genetic feshion.

Lets start importing the libraries we need in a jupyter notebook:

```python
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from matplotlib.pyplot import cm 
import scipy.ndimage as ndimage

%matplotlib inline
if os.name == 'posix':
    %config InlineBackend.figure_format = 'retina' #retina display settings
```



Let us define the activation function. We choose here a sigmoid, but it can be easily changed to a tanh or to one of the many ReLu variants. 

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

<p style="text-align:center;"><img src="/asset/images/nn_from_0/sigmoid.svg" alt="sigmoid" height="200"></p>

Let us create our neural network class. In the `__init__` function we require the `network_structure`, which is a list of integers defining the size of each layer of the network. `network_structure[0]` is the size of the input layer, while `network_structure[-1]` the size of the output one. The number of degrees of freedom of the network can be obtained using the `genome_size` function. Using set `set_genome` one can load a set of genomes into the network. Its input (`genome`) is a 2D numpy array containing a sequence of different genomes. Its shape must be *(number_of_genomes, genome_size)*. 

Finally the `feedforward` function uses the network to map a collection of inputs to a collection of outputs. The input shape must be *(number_of_genomes, input_size)*, while the shape of the output is *(number_of_genomes, output_size)*.

```python
class NeuralNetwork(object):
    def __init__(self, network_structure):
        self.network_structure = network_structure
        self.structure = []

        i = 0
        for a, b in zip(self.network_structure[:-1], self.network_structure[1:]):
            self.structure.append([i, i+a*b, i+a*b+b, a, b])
            i = i + a*b + b
        self._genome_size = i
    
    
    def set_genome(self, genome):
        
        population = genome.shape[0]
        self.weigths = []
        self.biases = []
        
        for i in self.structure:
            self.weigths.append(genome[:, i[0]:i[1]].reshape(population, i[3], i[4]))
            self.biases.append(genome[:, i[1]:i[2]].reshape(population, i[4]))

    def feedforward(self, inputs):
        layer = inputs
        for weigth, bias in zip(self.weigths, self.biases):
            layer = sigmoid(np.einsum('ji,jik->jk', layer, weigth) + bias)
        return layer
    
    @property
    def genome_size(self):
        return self._genome_size
```




## Track racing

Lets use the the network we just created to solve a simple task: we generate an horizontal random racing trak and player wins if it reacher the end without running off the road. A random track can be generated as follow:
{% highlight python %}
xx = np.arange(0, 400)

freq = np.exp(-(xx/10-20)**2/100)
par = (xx/10-25)**2/100
top_barrier = np.sin(xx/10*freq)*0.5 + 0.1 + np.sin(xx/10) + par
top_barrier = top_barrier - top_barrier[0] + 0.5
bottom_barrier = top_barrier - 1
{% endhighlight %}
<p style="text-align:center;"><img src="/asset/images/nn_from_0/simple_track.svg" alt="sigmoid" width="800"></p>

## Player physics
Lets define the player physics: a dot with a given 2D position and speed. Every evolution step change the position vector by the speed vector, moreover, the speed vector is updated by the force vector (mass=1).

```python
class Player(object):
    def __init__(self, p0, v0):
        self.p = p0
        self.v = v0
        
    def step(self, force=0.1):
        self.p = self.p + self.v/2
        self.v = self.v + force
        self.p = self.p + self.v/2
```


## Running simulation

Lets define two helper functions. The former runs the simulation given a set of genomes, the track and the nueral network. The latter plot both the track together with the player paths for a given set of genomes.

```python
def run(genomes, top_barrier, bottom_barrier, network, get_path=False):
    player = Player(np.zeros(genomes.shape[0]), np.zeros(genomes.shape[0]))
    network.set_genome(genomes)
    pp = np.ones([genomes.shape[0], top_barrier.shape[0]])
    max_score = (top_barrier.shape[0] - CUT)
    scores = np.ones(genomes.shape[0])*max_score
    
    for index in range(max_score):
        pp[:, index] = player.p

        inputs = np.array([player.v,
                           player.p-top_barrier[index],
                           player.p-bottom_barrier[index],
                           player.p-top_barrier[index+5],
                           player.p-bottom_barrier[index+5],
                           player.p-top_barrier[index+9],
                           player.p-bottom_barrier[index+9]
                          ]).reshape(7, genomes.shape[0]).transpose()
        
        force = network.feedforward(inputs).flatten() - 0.5
        player.step(force=force)
        is_out = np.logical_or(player.p>top_barrier[index], player.p<bottom_barrier[index])
        current = (1-is_out)*max_score + (is_out)*index
        scores = np.min([scores, current], axis=0).astype(np.int)
        
    if get_path:
        return [pp, scores]
    results = [scores, genomes]
    return list(zip(*sorted(list(zip(*results)), key=lambda x: -x[0])))

def plot_genome(genomes, top_barrier, bottom_barrier, network, alpha=0.5):
    genomes = np.array(genomes)
    xx = np.arange(top_barrier.shape[0])
    
    genome_paths = run(genomes, top_barrier, bottom_barrier, network, get_path=True)
    fig, ax = plt.subplots(1, figsize=(16, 4))
    for y, x_max in zip(*genome_paths):
        ax.plot(y[:x_max], color=cm.viridis(1-(x_max/np.max(genome_paths[1])*0.9 + 0.05)), alpha=alpha)
        
    ax.fill_between(xx, top_barrier, y2=np.max(top_barrier), color='#dddddd')
    ax.fill_between(xx, np.min(bottom_barrier), y2=bottom_barrier, color='#dddddd')
    ax.set_xlim(0, np.max(xx))
    ax.set_ylim(np.min(bottom_barrier), np.max(top_barrier))
    return a
```


Lets initialize the neural network and set a population of 50, with a evolution CUT of 10:

```python
network = NeuralNetwork(network_structure=[7,8,4,1])
POPULATION = 50
CUT = 10
```


