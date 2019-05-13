---
layout: post
title:  "Pandas cheat sheet"
date:   2019-05-09 23:00:00 +0100
categories: data_analysis
---

### Series (ordered key value store + numpy operations)
```python
s = pd.Series([7.8, 19.3, 7.8, 10.5])
s = pd.Series(data=[7.8, 19.3, 7.8, 10.5], index=['bronze', 'gold', 'iron', 'silver']) #s.index=[...] to reset
s = pd.Series({'bronze':7.8, 'gold':19.3}) #index=['gold', 'silver'] -> drop 'bronze' and assigns NaN to silver
```

### DataFrame
```python
df = pd.DataFrame({'val':[10, 20, 30, 40],
                   'patient_id':[1, 2, 3, 4],
                   'name':['Alice', 'Bob', 'Charlie', 'Devil']})

df[['patient_id', 'name', 'val']] # changing order
df.columns, df.dtypes # column names, column data types

x = df['name'] # get column as pd.Series (as a poiter)
x = df[['name']].copy() # get column as pd.DataFrame (as a copy)
x = df.loc[0] # get first row as pd.Series (as a poiter)
x = df.values # get all the DataFrame as a simple numpy array

df.shape # -> (4, 3) : (number of rows, number of columns)
df.head(2), df.tail(3) # get head or tail

df['year'] = 2019 # or[2019, 2019, 2020, 2020] add a new column (must match len)
df['year'] = pd.Series(['a', 'b', 'c'], index=[1, 2, 9]) # ok (fills with NaN and 'c' is dropped)

df.drop(['year'], axis=1, inplace=True) # drop column
df.drop([1], axis=0, inplace=True) # drop row
```

### Load data
```python
pd.read_csv('f.csv', header=None) # default header is 0, 
pd.read_csv('f.csv', sep='\s+') # sep can be RegEx (any number of spaces)
pd.read_csv('f.csv', index_col=['id', 'name']) # set colums as index
pd.read_csv('f.csv', skiprows=[1,3])
pd.read_csv('f.csv', nrows=2) #import only first 2 rows (+ header)

data_chunks = pd.read_csv('filename.csv', chunksize=2)
data_chunks.get_chunk() # or iterate over chunks with a for loop

pd.read_csv('f.csv', na_values=['?', -9999]) # fills with NaNs
```

### Indexing
```python
df.index = df['id'].astype(str) + df['name']
df.index.is_unique # if False df.loc may rerturns multiple Series
df.reindex(df.index[::-1]) # reverse (or change) order of rows
df.reindex(range(0, 10), fill_value='hey') # force new index and fill up missings
df.reindex(range(0, 10), method='ffill') # fill up with selected method
df.sort_index(ascending=False, axis=0) # sort row indexes
df.sort_index(ascending=True, axis=1) # sort columns indexes

df.set_index(['id', 'name']) # hierarchical indexing
df.loc[(2, 'bob')] # access hierarchical indexing
```

### Selection
```python
s[:-4], s[:'charlie'] # numpy slicing or slicing with labels
df.query('id>2') # equivalent to df[df['id']>2]
df.query('id > @variable') # use @ to use a variable in the current namespace
df.loc[2, ['id', 'name']] # get a row as a Series
```


### Functions
```python
df.apply(np.median, axis=0)
df[['name','id']].sort_values(ascending=[False,True], by=['id', 'name'])
df.isnull()
df.dropna() # drops entire rows in which one or more values are missing (how='all' if all the fields are missing)
df.fillna({'id': 0, 'name': 'unknown'}) # or just df.fillna(-999) or by interpolation method='bfill'
```








