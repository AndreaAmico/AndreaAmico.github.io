---
layout: post
title:  "Matplotlib"
date:   2019-07-21 22:00:00 +0100
categories: data_analysis
---

### My colors
```python
def color(index, alpha=1):
    color_list = ['#0c6575', '#bbcbcb', '#23a98c', '#fc7a70', '#a07060',
                  '#003847', '#FFF7D6', '#5CA4B5', '#eeeeee']
    return color_list[index % len(color_list)] + hex(int(255 * alpha))[2:]
```

### Plots
```python
plt.axis('off')
plt.close()

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])

ax.fig.savefig('./figurename.png', bbox_inches='tight', dpi=300)
```

### Plot, scatter and errorbar
<p style="text-align:center;"><img src="/asset/images/matplotlib/basic_plots.svg" alt="basic plots" width="800"></p>

```python
from sklearn.datasets import load_iris
iris_dataset = load_iris()

y = iris_dataset.data.mean(1)
x = np.linspace(1, 30, y.shape[0])

fig, (ax_plot, ax_scatter, ax_errorbar) = plt.subplots(1, 3, figsize=(12,3))

ax_plot.plot(x, y,
    color=color(0),
    linestyle='-',
    linewidth=1,
    markevery=10,
    marker='o',
    drawstyle='steps',
    fillstyle='bottom',
    markersize=15,
    markerfacecolor=color(1),
    markeredgecolor=color(2))

ax_scatter.scatter(x, y,
    c=iris_dataset.target,
    cmap=plt.cm.viridis,
    s=np.power(x, 2),
    edgecolors='#444444',
    marker='X')

df_iris = pd.DataFrame(y, columns=['data'])
df_iris['target'] = iris_dataset.target
RESAMPLE_BIN_SIZE = 10
df_iris_mean = df_iris.reset_index(drop=True).groupby(
    by=lambda x: int(x/RESAMPLE_BIN_SIZE), axis=0).mean()
df_iris_std = df_iris.reset_index(drop=True).groupby(
    by=lambda x: int(x/RESAMPLE_BIN_SIZE), axis=0).std()

ax_errorbar.errorbar(
    x=df_iris_mean.index,
    y=df_iris_mean.data.values,
    yerr=df_iris_std.data.values,
    fmt='H',
    c=color(2),
    ecolor=color(5),
    capsize=2,
    barsabove=False,
    errorevery=1,
    markersize=10,
    markeredgecolor=color(5),
    markeredgewidth=1)

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
fig.savefig('./basic_plots.svg', bbox_inches='tight')
```

### Dimensionality reduction
<p style="text-align:center;"><img src="/asset/images/matplotlib/dimensionality_reduction.svg" alt="dimensionality reduction" width="600"></p>

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

iris_dataset = load_iris()
X_TSNE = TSNE(n_components=2, random_state=1).fit_transform(iris_dataset.data)
X_PCA = PCA(n_components=2).fit_transform(iris_dataset.data)


fig, (ax_pca, ax_tsne) = plt.subplots(1, 2, figsize=(10, 4))

for target_class_index, target_class in enumerate(np.unique(iris_dataset.target)):
    marker = f'${target_class}$'
    class_mask = (iris_dataset.target == target_class)
    
    ax_tsne.scatter(X_TSNE[class_mask, 0], X_TSNE[class_mask, 1],
              marker=marker, color=color(target_class_index),
              label=iris_dataset.target_names[target_class])
    ax_pca.scatter(X_PCA[class_mask, 0], X_PCA[class_mask, 1],
              marker=marker, color=color(target_class_index))
    
ax_pca.set_title('PCA', color=color(5))
ax_tsne.set_title('t-SNE', color=color(5))

ax_tsne.legend(loc='lower left', bbox_to_anchor=(-1, 0.1), ncol=3,
    scatterpoints=3, frameon=True, fancybox=True, framealpha=0.2,
    facecolor=color(1), edgecolor=color(0))

ax_pca.axis('off')
ax_tsne.axis('off')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
    wspace=1, hspace=None)

fig.savefig('./plots/dimensionality_reduction.svg', bbox_inches='tight')
```

### Coloreful axes
<p style="text-align:center;"><img src="/asset/images/matplotlib/grid_spec.svg" alt="grid_spec" width="600"></p>

```python
# left=None, bottom=None, right=None, top=None, wspace=None,
# hspace=None, width_ratios=None, height_ratios=None
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3),
    sharey=True, gridspec_kw={'width_ratios':[1,2]})

ax1.set_xlabel('Small label')
ax1.set_ylabel('y label')

ax2.set_xlabel('Large label', size=14)

ax1.spines['top'].set_color('orange')
ax2.spines['right'].set_color('orange')

ax1.tick_params(axis='y', colors='green')
ax2.tick_params(axis='x', colors='red')

ax1.yaxis.label.set_color('purple')
ax2.xaxis.label.set_color('blue')

ax2.grid(alpha=0.4)

fig.savefig('./plots/grid_spec.svg', bbox_inches='tight')
```



