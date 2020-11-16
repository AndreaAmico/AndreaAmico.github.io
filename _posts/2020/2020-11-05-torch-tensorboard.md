---
layout: post
title:  "Torch tensorboard"
date:   2020-11-05 22:00:00 +0100
categories: ML-tools
---

Collection of snippets of `tensorboard` usage with `pytorch` using the [tensorboardX](https://github.com/lanpa/tensorboardX) library.


## Requirements
```bash
pip install torch
pip install tensorflow
pip install tensorboard
```

## Run tensorboard
```bash
tensorboard --logdir runs
```
Open up the webapp in your browser, usually at `http://localhost:6006`.


## Training script
```python
import numpy as np
from torch.utils.tensorboard import SummaryWriter

## clean run folder
import shutil
import time
shutil.rmtree('runs', ignore_errors=True)


## run 1
writer = SummaryWriter(flush_secs=1)
for x in range(100):
    noise = np.random.normal(scale=1)
    y = (x/20)**2 + noise
    writer.add_scalar('noisy_parabola', y, x)
    time.sleep(0.1)
writer.close()


## run 2
writer = SummaryWriter(flush_secs=1)
for x in range(100):
    noise = np.random.normal(scale=6)
    y = (x/20)**2 + noise
    writer.add_scalar('noisy_parabola', y, x)
    time.sleep(0.1)
writer.close()
```
The tensorboard plot will be updated in real time. The final output will be something similar to this:
<p style="text-align:center;"><img src="/asset/images/tensorboard/tensorboard.png" alt="tensorboard" width="500"></p>

