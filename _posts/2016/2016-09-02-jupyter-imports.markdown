---
layout: post
title:  "Jupyter import"
date:   2016-09-02 18:30:00 +0100
categories: fast-copy-paste
---

### ML project imports
```python
import os, sys, glob, time, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import IPython #IPython.display.clear_output()

import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import gaussian_filter

from cycler import cycler
colors = ['#0c6575', '#bbcbcb', '#23a98c', '#fc7a70','#a07060',
          '#003847', '#FFF7D6', '#5CA4B5', '#eeeeee']
plt.rcParams['axes.prop_cycle'] = cycler(color = colors)
```



### Simple imports
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Colab
```python
try:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)
    root_path = 'gdrive/My Drive/Colab Notebooks/WORKING_FOLDER_NAME/'
    print('Working on google colab')
except:
    root_path = '../'
    print('Working locally')


# to dowload files
from google.colab import files
files.download('my_file.txt') 
```

### On MacOS
```python
%config InlineBackend.figure_format = 'retina'
```

### Keras reproducibility
Use this to ensure seproducibility when using keras: notice the restriction to a single core
```python
import os, sys
os.environ['PYTHONHASHSEED']=str(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
tf.set_random_seed(0)

from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
```

