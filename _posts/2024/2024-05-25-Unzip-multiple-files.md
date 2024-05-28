---
layout: post
title:  "Unzip files with python"
date:   2024-05-26 20:00:00 +0100
categories: other
---


Unzip a single file:
```python
import gzip
import shutil
with gzip.open('my_compressed_file.csv.gz', 'rb') as f_in:
    with open('my_uncompressed_file.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
```


Unzip all the file in a folder:
```python
import gzip
import shutil
import glob
for p in glob.glob('./my_folder/*.gz'):
    with gzip.open(p, 'rb') as f_in:
        with open(p[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
```