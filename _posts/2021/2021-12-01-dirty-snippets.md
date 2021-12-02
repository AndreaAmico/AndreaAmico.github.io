---
layout: post
title:  "Dirty snippets"
date:   2021-12-01 20:00:00 +0100
categories: other
---



### Define and create a folder in one line
```python
new_dir = lambda d: d if os.path.isdir(d) else (d, os.makedirs(d))[0]

## Usage
output_path = new_dir('./models/2021')
```
`output_path` will be the path string, the directory will be automatically created if it does not exist.

