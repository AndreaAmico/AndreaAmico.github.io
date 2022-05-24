---
layout: post
title:  "Pyspark basics"
date:   2022-05-12 20:00:00 +0100
categories: data-analysis
---

# Pandas to spark
We can create a Spark dataframe starting from a Pandas dataframe using the function `spark.createDataFrame`:
```python
import pyspark

DATA_LEN = 5

df_pandas_foo_first = pd.DataFrame(data=dict(
  A = np.random.choice(['hey', 'oh'], size=DATA_LEN),
  B = np.random.random(size=DATA_LEN)
))

df_pandas_foo_second = pd.DataFrame(data=dict(
  A = np.random.choice(['hey', 'oh'], size=DATA_LEN),
  C = np.random.uniform(5, 10, size=DATA_LEN),
  D = np.random.choice(['one', 'two', 'five'], size=DATA_LEN)
))

df_foo_first = spark.createDataFrame(df_pandas_foo_first)
df_foo_second = spark.createDataFrame(df_pandas_foo_second)
```

We can **join** the two dataframes via the `.join` function and display the resulting dataframe using the `.display` method:
```python
df_foo = df_foo_first.join(df_foo_second, how='left', on=['A'])
df_foo.display()
```

# IO operations
## csv
By default a `.csv` write create multiple `.csv` files to facilitate parallel operations. One can force the creation of asingle output file using the `coalesce(1)` method.
```python
df_foo.coalesce(1).write.csv(DATA_PATH, header=True)  ## coalesce(1) to save single csv
df_foo = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
```
## parquet
```python
df_foo.write.format("parquet").save(DATA_PATH)
df_foo = spark.read.format("parquet").load(DATA_PATH)
```

## delta table
Delta tables are more complex data structures which allow functionality like high query performance, flexible data architecture and history rollback capabilities. We can easily create a delta table with pyspark:
```python
df_foo.write.format("delta").mode("append").partitionBy("A","D").save(DATA_PATH)
```
We can use delts tables as qsl tables with addictional functionalities:
```python
query_merge =   f"""
  MERGE INTO delta.`{DATA_PATH}` AS target
  ...
  """
spark.sql(query_merge)
```

```python
query_delete =   f"""
  DELETE  FROM delta.`{DATA_PATH}`
  WHERE A = "oh"
  """
spark.sql(query_delete)
```

We can check the history and rollback the data to a previous version via the instructions:
```sql
%sql
DESCRIBE HISTORY delta.`{DATA_PATH}`
```

```sql
%sql
RESTORE TABLE delta.`{DATA_PATH}` TO version as of 0
```

# Temporary view and sql operation
In order to perform sql operation within a spark table (not saved yet) one can create a temporary view using the function `createOrReplaceTempView`:
```python
df_foo.createOrReplaceTempView('foo_view')
df_foo_first.createOrReplaceTempView('foo_first_view')
df_foo_second.createOrReplaceTempView('foo_second_view')
```
This allows us to use the view as a real sql table:
```sql
%sql
describe table foo_view
```

## SELECT FROM WHERE syntax
```sql
%sql
SELECT *  
FROM foo_view
WHERE C >= 8
```

## Groupby operations
```sql
%sql
SELECT A, mean(C)
FROM foo_view
GROUP BY A
```

Ordering the output:
```sql
%sql
SELECT A, mean(C) as C_mean,  count(*) as num, max(B) as B_max
FROM foo_view
GROUP BY A
ORDER BY C_mean desc
```

## Join operations
```sql
%sql
SELECT foo_first.A, foo_first.B_max, foo_second.C_mean, num

FROM (
  SELECT A, max(B) as B_max
  FROM foo_first_view
  GROUP BY A
) foo_first

INNER JOIN (
  SELECT A, count(*) as num, mean(C) as C_mean
  FROM foo_second_view
  GROUP BY A
) foo_second


ON (foo_first.A = foo_second.A)
```





