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
By default a `.csv` write create multiple `.csv` files to facilitate parallel operations. One can force the creation of asingle output fileusing the 'coalesce(1)' method.
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

query_delete =   f"""
  DELETE  FROM delta.`{DATA_PATH}`
  WHERE A = "oh"
  """
spark.sql(query_delete)
```

We can rollback the data to a previous version:
```sql
%sql
RESTORE TABLE delta.`{DATA_PATH}` TO version as of 0
```



