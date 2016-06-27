---
layout: post
title:  "Parse experiment csv with pandas"
date:   2016-06-27 12:33:42 +0100
categories: python
---

Load the "experiment_file.csv" in a pandas database:
{% highlight python %}
csv_file = ".\\experiment_file.csv"
df = pd.read_csv(csv_file, sep=',', header=0)
{% endhighlight %}


Find the name of the first column in the .csv containing the string "name".
The object `df.columns` is the list of all the columns name:
{% highlight python %}
filename = next(x for x in df.columns.values if ("name" in x))
my_variable = next(x for x in df.columns.values if ("time" in x))
my_data_point = next(x for x in df.columns.values if ("data" in x))
{% endhighlight %}

Delete all the rows with "omit" tick and all the rows that 
do not contain a number in the "my_variable" column:
{% highlight python %}
df = df[df["Omit"]!=True].dropna(subset=[my_variable])
{% endhighlight %}

To itarate over rows one can use `iterrows`. Similarely to enumerate, `i` is the row index 
and `r` is the dataframe row itself.
{% highlight python %}
for i, r in df.iterrows():
    print("This is the {0} row.".format(i))
    my_func(r[my_variable])
{% endhighlight %}

Syntax to modify an entire column, to filter it or to create an array from it:
{% highlight python %}
df[my_variable] = df[my_variable]/1000
df = df[df[my_variable]>0]
x = df[my_variable].values
{% endhighlight %}

In order to group data sharing the same value one can use the `groupby` function toghether with 
`agg` and `reset_index`. Nb: you need to import *scipy.stats* for the standard error of the mean. 
{% highlight python %}
grouped = df.groupby(my_variable, sort=True).agg([np.mean, stats.sem]).reset_index()
x = grouped[my_variable].values
y_mean = grouped[my_data_point].values.T[0]
y_sem = grouped[my_data_point].values.T[1]
{% endhighlight %}