---
layout: post
title:  "AWS snippets"
date:   2021-06-04 20:00:00 +0100
categories: other
---





### Command line from notebook
```bash
# aws s3 ls [bucket_name]
aws s3 ls s3://my-bucket/my-folder

# Copy file from s3 to local storage
# aws s3 cp [bucket_name/file_mane] [file_name]
aws s3 cp s3://my-bucket/my-folder/my-file.csv ./my-file.csv

# show the first few lines of a text file
head -n 4./my-file.csv
```
 

### Get AWS session, region, bucket, and accound_id with boto3 and sagemaker
```python
import boto3
import sagemaker

sess   = sagemaker.Session()
# S3 bucket name
bucket = sess.default_bucket()
# AWS region
region = boto3.Session().region_name

# Account ID 
sts = boto3.Session(region_name=region).client(service_name="sts", region_name=region)
account_id = sts.get_caller_identity()['Account']
```


### awswrangler
The first step is to create a glue catalog database:
```python
import awswrangler as wr

wr.catalog.create_database(
    name='...', # AWS Glue Catalog database name e.g. my_db_deep_learning
    exist_ok=True
)
```
We can now register a specific `.csv` file to the glue catalog:

```python
res = wr.catalog.create_csv_table(
    database='...', # AWS Glue Catalog database name
    path='s3://{}/data/transformed/'.format(bucket), # S3 object path for the data
    table='...', # table name e.g. my_table
    columns_types={
        'ids': 'int',        
        'my_feature': 'string',
        'my_target': 'int'      
    },
    mode='overwrite',
    skip_header_line_count=1,
    sep=','    
)
```

The table is now visible from the glue section of the AWS website.


### AWS Athena
AWS Athena allows for **sql** queries directly from an s3 bucket (it stores temporary binary files to speed up subsequent queries).
```python
wr.athena.create_athena_bucket()
```
We can now perform any SQL query and store the result in a pandas dataframe:
```python
my_query = """
SELECT my_target, COUNT(my_target) AS my_target_count
FROM my_table
ORDER BY my_target_count
"""

df = wr.athena.read_sql_query(
    sql=my_query,
    database=... # here it goes the database name e.g. my_db_deep_learning
)
```

