---
layout: post
title:  "Pandas cheat sheet"
date:   2019-05-09 22:00:00 +0100
categories: data-analysis
---

### Global settings
{% include _html/menu_pandas_.html %}
```python
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 150)
```

### DataFrame
{% include _html/menu_pandas_.html %}
```python
## SERIES
s = pd.Series([7.8, 19.3, 7.8, 10.5])
s = pd.Series(data=[7.8, 19.3, 7.8], index=['bronze', 'gold', 'iron'])
s = pd.Series({'bronze':7.8, 'gold':19.3})
#index=['gold', 'silver'] -> drop 'bronze' and assigns NaN to silver
```

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

df.apply(np.median, axis='columns') # or 'rows'

df.shape # -> (4, 3) : (number of rows, number of columns)
df.head(2), df.tail(3) # get head or tail

df['year'] = 2019 # or[2019, 2019, 2020, 2020] add a new column (must match len)
df['year'] = pd.Series(['a', 'b', 'c'], index=[1, 2, 9]) # ok (fills with NaN and 'c' is dropped)

df.drop(['year'], axis=1, inplace=True) # drop column
df.drop([1], axis=0, inplace=True) # drop row

df.rename(columns={"col1": "A", "col2": "B"}, inplace=True) #rename columns with dictionary
df.rename(columns=lambda c: 'my_prefix_'+c, inplace=True) ##rename columns with lambda function
```

### Load data
{% include _html/menu_pandas_.html %}
```python
pd.read_clipboard(sep='\s\s+') # read from clipboard!

## from string in python 3
from io import StringIO
pd.read_csv(StringIO(my_string), sep=",")

pd.read_csv('f.csv', header=None) # default header is 0, 
pd.read_csv('f.csv', sep='\s+') # sep can be RegEx (any number of spaces)
pd.read_csv('f.csv', index_col=['id', 'name']) # set colums as index
pd.read_csv('f.csv', skiprows=[1,3])
pd.read_csv('f.csv', nrows=2) #import only first 2 rows (+ header)
pd.read_csv('foo.csv', parse_dates=['year'])

data_chunks = pd.read_csv('filename.csv', chunksize=2)
data_chunks.get_chunk() # or iterate over chunks with a for loop

pd.read_csv('f.csv', na_values=['?', -9999]) # fills with NaNs
```

### Indexing
{% include _html/menu_pandas_.html %}
```python
pd.read_csv('foo.csv', index_col=['id', 'name']) # set colums as index
df.index = df['id'].astype(str) + df['name']
df.index.is_unique # if False df.loc may rerturns multiple Series

df.reset_index(inplace=True, drop=True)

df.reindex(df.index[::-1]) # reverse (or change) order of rows
df.reindex(range(0, 10), fill_value='hey') # force new index and fill up missings
df.reindex(range(0, 10), method='ffill') # fill up with selected method
df.sort_index(ascending=False, axis=0) # sort row indexes
df.sort_index(ascending=True, axis=1) # sort columns indexes

df.set_index(['id', 'name']) # hierarchical indexing
df.loc[(2, 'bob')] # access hierarchical indexing
df.iloc[2] # indexing by column number
df.stack(level=1) # from rows to columns
df.unstack(level=1) # from columns to rows

## Flatten a hierarchical index in columns
df.columns = ['_'.join(col).strip() for col in df.columns.values]
# flatten the index if the df is obtained by grouping: df = df.reset_index()
```

### Selection
{% include _html/menu_pandas_.html %}
```python
s[:-4], s[:'charlie'] # numpy slicing or slicing with labels
df.query('id>2') # equivalent to df[df['id']>2]
df.query('id > @variable') # use @ to use a variable in the current namespace
df.loc[2, ['id', 'name']] # get a row as a Series
df.loc['row_start':'row_end', 'col_start':'col_end']
```



### TimeSeries
{% include _html/menu_pandas_.html %}
```python
pd.read_csv('foo.csv', parse_dates=['year'])
df.date = pd.to_datetime(df.date) # format='%Y-%m-%d'
df.resample('10AS') #resample every decade

## set date as index
df.set_index('date')

## if date is the index:
df.loc[df.index.month == 1].mean() # get the average in Genuary
df['2015-02-25':'2015:02-27'] # select range of dates

# Fill up missing dates
dt = pd.date_range('01-01-2017', '01-11-2017')
idx = pd.DatetimeIndex(dt)
df.reindex(idx)

# Time grouper
pd.Grouper(freq='M', key='date')

```
#### Resample strings

BusinessDay | 'B' | business day (weekday)
Week | 'W' | one week
MonthEnd | 'M' | calendar month end
MonthBegin | 'MS' | calendar month begin
BusinessMonthBegin | 'BMS' | business month begin
YearEnd | 'A' | calendar year end
YearBegin | 'AS' or 'BYS' | calendar year begin
BYearEnd | 'BA' | business year end
BYearBegin | 'BAS' | business year begin
Easter | None | Easter holiday
CustomBusinessHour | 'CBH' | custom business hour
Day | 'D' | one absolute day
Hour | 'H' | one hour
Minute | 'T' or 'min' | one minute
Second | 'S' | one second
Milli | 'L' or 'ms' | one millisecond
Micro | 'U' or 'us' | one microsecond
Nano | 'N' | one nanosecond

<br>

#### Datetime string format

%a |  : : Locale’s abbreviated weekday name.
%A |  : : Locale’s full weekday name.
%b |  : : Locale’s abbreviated month name.
%B |  : : Locale’s full month name.
%c |  : : Locale’s appropriate date and time representation.
%d |  : : Day of the month as a decimal number [01,31].
%f |  : : Microsecond as a decimal number [0,999999], zero-padded on the left
%H |  : : Hour (24-hour clock) as a decimal number [00,23].
%I |  : : Hour (12-hour clock) as a decimal number [01,12].
%j |  : : Day of the year as a decimal number [001,366].
%m |  : : Month as a decimal number [01,12].
%M |  : : Minute as a decimal number [00,59].
%p |  : : Locale’s equivalent of either AM or PM.
%S |  : : Second as a decimal number [00,61].
%U |  : : Week number of the year (Sunday as the first day of the week)
%w |  : : Weekday as a decimal number [0(Sunday),6].
%W |  : : Week number of the year (Monday as the first day of the week)
%x |  : : Locale’s appropriate date representation.
%X |  : : Locale’s appropriate time representation.
%y |  : : Year without century as a decimal number [00,99].
%Y |  : : Year with century as a decimal number.
%z |  : : UTC offset in the form +HHMM or -HHMM.
%Z |  : : Time zone name (empty string if the object is naive).
%% |  : : A literal '%' character.

<br>

### Clean
{% include _html/menu_pandas_.html %}
```python
df.isnull()
df.dropna() # how='all' if all the fields are missing, thres=2 if at least two good
df.fillna({'id': 0, 'name': 'unknown'}) # or just df.fillna(-999) or by interpolation method='bfill'
df['age'].fillna(df['age'].mean(skipna=True), inplace=True) # fix a signle column
df.fillna(method='ffill', inplace=True) # forward filling
df['A'] = df['A'].fillna(df['B']) # forward using a second column
df.interpolate() # linear interpolation on missing data
df['A'].fillna(value=df['A'].mode()[0], inplace=True) #most frequent entry for categorical data

df.replace({'col_name1':{
                'replace this':'with_this',
                'that':'with 0'}
            'col_name2':{...}})

df.replace('[A-Za-z]', '', regex=True)
df.replace(['poor', 'good', 'amazing'], [0,1,2])

df = df.loc[df['condition_column']=='my_filter', 'column'] = 42 # manually set 42 into a group of locations
```

### Explore
{% include _html/menu_pandas_.html %}
```python
df.shape
df.info()
df.describe() # include='all' describes non-numeric as well
df[['name','id']].sort_values(ascending=[False,True], by=['id', 'name'])
s.value_counts() # number of non NaN items
df.col_name.nunique() # number of unique values
```

### Grouping
{% include _html/menu_pandas_.html %}
```python
g = df.groupby('col_name')
for group_element_name, group_element_df in g:
    pass

group_keys = [key for key, _ in g] #faster than g.groups.keys()
g.get_group('group name')


# Aggregations
g.mean() # or sum or plot
g.mean().reset_index() # to flatten out the output
g.agg(['min', 'max'])
g.agg(
    b_min=pd.NamedAgg(column='B', aggfunc=np.min),
    c_sum=pd.NamedAgg(column='C', aggfunc=np.sum))
    
def my_agg_func(group):
    return pd.Series(dict(
        mean_B = np.mean(group['B']),
        sum_B_C = np.sum(group['B']) + np.sum(group['C'])
    ))
g.apply(my_agg_func)

# Time grouper
pd.Grouper(freq='M', key='date')

# Pivot
df.pivot(index='date', columns='city', values='temperature')
df.pivot_table(index='date', columns='city', aggfunc='mean') #margins=True
df.pivot_table(pd.Grouper(freq='M', key='date'), columns='temperature')
pd.melt(df, id_vars=['keep_col_1', 'keep_col_2', ...], value_vars=[value_to_melt_1, value_to_melt_2, ...])
pd.cross_tab(df.job_title, df.gender, aggfunc='count') # margins=True, normalize='index'

# Resampling non-timeseries data
df.reset_index(drop=True).groupby(by=lambda x: int(x/WINDOW_SIZE), axis=0).mean()

```
### Transform
{% include _html/menu_pandas_.html %}
```python
# List explosion
df = pd.DataFrame({'a':[1, 2, 3], 'b':['x', 'y', 'y']})
mapping_dict = {"x": ['x'], "y": ["W", "V", "Q"]}
df['b'] = df.b.apply(lambda x: mapping_dict[x])
df = df.explode('b')

```


### Join
{% include _html/menu_pandas_.html %}
```python
pd.concat([df1, df2], ignore_index=True, axis='columns') # concat with reindexing
pd.concat([df1, df2], ignore_index=False, keys=['df1_key_name', 'df2_key_name']) # concat with reindexing

# inner->intersection, outer->union, left->left+intersection, right
df_left.merge(df_right, how='inner', suffixes=('_left', '_right'))
df_left.merge(df_right, indicator=True) # add column with merging specifications

```

### SQL
{% include _html/menu_pandas_.html %}
```python
import pymysql
import sqlalchemy

eng = sqlalchemy.create_engine('mysql+pymysql://root:psw@localhost:3306/dbname')
eng.execute('DROP TABLE table_name_to_drop')

#db type (can be 'oracle://'), user, psw, host name, port, database name
df = pd.read_sql_table('table_name', eng) # columns=['col1', 'col2', ..]


query = '''
    SELECT users.name, users.email, orders.name
    FROM users INNER JOIN orders
    ON users.id = orders.id
'''
df = pd.read_sql_query(query, eng) # chunksize.. for large amount of data

```



