---
layout: post
title:  "Neural Network from scratch"
date:   2018-11-22 22:00:00 +0100
categories: reinforcement_learning
---

Let's build a simple neural network from scratch in pure python plus numpy and finally train it in a genetic fashion.

Let's start importing the libraries we need in a Jupiter notebook:

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



Let us define the activation function. We choose here a sigmoid, but it can be easily changed to a tanh or one of the many ReLu variants. 

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

<p style="text-align:center;"><img src="/asset/images/nn_from_0/sigmoid.svg" alt="sigmoid" height="200"></p>

Let us create our neural network class. In the `__init__` function we require the `network_structure`, which is a list of integers defining the size of each layer of the network. `network_structure[0]` is the size of the input layer, while `network_structure[-1]` the size of the output one. The number of degrees of freedom of the network can be obtained using the `genome_size` function. Using set `set_genome` one can load a set of genomes into the network. Its input (`genome`) is a 2D numpy array containing a sequence of different genomes. Its shape must be *(number_of_genomes, genome_size)*. 

Finally, the `feedforward` function uses the network to map a collection of inputs to a collection of outputs. The input shape must be *(number_of_genomes, input_size)*, while the shape of the output is *(number_of_genomes, output_size)*.

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
        self.weights = []
        self.biases = []
        
        for i in self.structure:
            self.weights.append(genome[:, i[0]:i[1]].reshape(population, i[3], i[4]))
            self.biases.append(genome[:, i[1]:i[2]].reshape(population, i[4]))

    def feedforward(self, inputs):
        layer = inputs
        for weigh, bias in zip(self.weights, self.biases):
            layer = sigmoid(np.einsum('ji,jik->jk', layer, weigh) + bias)
        return layer
    
    @property
    def genome_size(self):
        return self._genome_size
```




## Track racing

Let's use the network we just created to solve a simple task: we generate a horizontal random racing track and player wins if it reacher the end without running off the road. A random track can be generated as follow:
```python
xx = np.arange(0, 400)

freq = np.exp(-(xx/10-20)**2/100)
par = (xx/10-25)**2/100
top_barrier = np.sin(xx/10*freq)*0.5 + 0.1 + np.sin(xx/10) + par
top_barrier = top_barrier - top_barrier[0] + 0.5
bottom_barrier = top_barrier - 1
```

<p style="text-align:center;"><img src="/asset/images/nn_from_0/simple_track.svg" alt="simple track" width="800"></p>

## Player physics
Let's define the player physics: a dot with a given 2D position and speed. Every evolution step changes the position vector by the speed vector, moreover, the speed vector is updated by the force vector (mass=1).

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

Let's define two helper functions. The former runs the simulation given a set of genomes, the track, and the neural network. The latter plot both the track together with the player paths for a given set of genomes.

The `run` function takes as input a set of genomes and create the players. The player variable has the shape *(number_of_genomes, track_length)*, where *track_length* corresponds to the horizontal size of the track. Moreover, the `sensor` list contains the position of three sensors owed by the player: each frame the player knows the position of the track boundaries few pixels ahead. We initialize the neural network with the genomes and we run the simulation for *track_length - last_sensor* steps. Finally, the function returns a list of scores and genomes, sorted by the score. If `get_path==True` the path of each player is returned.

```python
def run(genomes, top_barrier, bottom_barrier, network, get_path=False, sensors=[0, 5, 9]):
    player = Player(np.zeros(genomes.shape[0]), np.zeros(genomes.shape[0]))
    network.set_genome(genomes)
    players = np.ones([genomes.shape[0], top_barrier.shape[0]])
    max_score = (top_barrier.shape[0] - sensors[-1])
    scores = np.ones(genomes.shape[0])*max_score
    
    for index in range(max_score):
        players[:, index] = player.p

        inputs = np.array([player.v,
                           player.p-top_barrier[index + sensors[0]],
                           player.p-bottom_barrier[index + sensors[0]],
                           player.p-top_barrier[index + sensors[1]],
                           player.p-bottom_barrier[index + sensors[1]],
                           player.p-top_barrier[index + sensors[2]],
                           player.p-bottom_barrier[index + sensors[2]]
                          ]).reshape(7, genomes.shape[0]).transpose()
    
        force = network.feedforward(inputs).flatten() - 0.5
        player.step(force=force)
        is_out = np.logical_or(player.p>top_barrier[index], player.p<bottom_barrier[index])
        current = (1-is_out)*max_score + (is_out)*index
        scores = np.min([scores, current], axis=0).astype(np.int)
        
    if get_path:
        return [players, scores]
    results = [scores, genomes]
    return list(zip(*sorted(list(zip(*results)), key=lambda x: -x[0])))
```

The function `plot_genome` plots the track and the path chosen by the different genomes. The better is the result of the single genome, the darker is its path color.

```python
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


Let's initialize the neural network with the following structure: `[7,8,4,1]`. 7 inputs, two hidden layers with 8 and 4 neurons respectively, and one output. We set each population to have 50 different genomes, which are initialized at random using a uniform distribution between -1 and 1. The result of running the simulation with this first set of random genomes is plotted below.

```python
network = NeuralNetwork(network_structure=[7,8,4,1])
POPULATION = 50
np.random.seed(42)

genomes = np.random.uniform(-1, 1, size=(POPULATION, network.genome_size))
results =  run(genomes, top_barrier, bottom_barrier, network)
```
<p style="text-align:center;"><img src="/asset/images/nn_from_0/first_genomes.svg" alt="first genomes" width="800"></p>

As expected, the first population has very poor performance. Nevertheless, we can still see that some genomes perform much better than others. We will see that defining the next generation of genomes by variations of the best performers, will quickly lead to good drivers.

## Evolution and selection

We now need to define how the next generation of genomes is created starting by the previous one. We choose a simple approach here: to create a child we randomly choose a parent by sampling using an exponential distribution (from the list of parents, sorted by performance). We then mutate the genome by summing each of the genome weights by a random number, selected by a uniform distribution between `-mutation_rate` and `+mutation_rate`:

```python
def evolution_step(results, mutation_rate=0.1):
    population = len(results[0])
    new_generation = []
    for _ in range(population):
        genome_in_mutation = results[1][min([int(np.random.exponential(scale=5)), population-1])]
        genome_in_mutation = genome_in_mutation + np.random.uniform(-mutation_rate, mutation_rate, size=genome_in_mutation.shape[0])
        new_generation.append(genome_in_mutation)
    return np.array(new_generation)
```
 We can now run the simulations for a number of epochs and look at the performance of the model. Below, we report the evolution after 30, 60 and 120 epochs, plotting the first 20 performers. 

```python
scores = []
for index in range(NUMBER_OF_EPOCHS):
    genomes =  evolution_step(results, mutation_rate=0.1)
    results =  run(genomes, top_barrier, bottom_barrier, network)
    scores.append(np.mean(results[0][:-20]))
```

<p style="text-align:center;"><img src="/asset/images/nn_from_0/genomes30.svg" alt="genomes 30" width="800"></p>
<p style="text-align:center;"><img src="/asset/images/nn_from_0/genomes60.svg" alt="genomes 60" width="800"></p>
<p style="text-align:center;"><img src="/asset/images/nn_from_0/genomes120.svg" alt="genomes 120" width="800"></p>

By looking at the performance at each evolution step we can see that, between 50 and 80 epochs, the model learns how to stay on track. Beyond 100 epochs almost all the genomes complete the track with no problem. 

<p style="text-align:center;"><img src="/asset/images/nn_from_0/genomes_performance.svg" alt="genomes performance" width="400"></p>


## Testing the model

We now want to check whether if the model learned to drive or just memorized the track. To check that, we create a brand new, much harder track, and we test how the model performs in an unknown environment, with no training. Let's create the track: 

```python
test_xx = np.arange(0, 4000)

np.random.seed(0)
test_top_barrier = np.random.random(test_xx.shape[0])
test_top_barrier = ndimage.gaussian_filter1d(test_top_barrier, 10)*20

test_top_barrier = test_top_barrier - test_top_barrier[0] + 0.5
test_bottom_barrier = test_top_barrier - 1
```

<p style="text-align:center;"><img src="/asset/images/nn_from_0/test_track.svg" alt="test track" width="800"></p>

And run the simulation, showing again the best 20 performers.
```python
ax = plot_genome(results[1][:20], test_top_barrier, test_bottom_barrier, network)
```

<p style="text-align:center;"><img src="/asset/images/nn_from_0/final_test.svg" alt="final test" width="800"></p>

As we can see, all the 20 genomes reach the end of the track! The conclusion is that the neural network learned to stay on track, and its knowledge is general and not track specific. 


