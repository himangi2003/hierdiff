# hierdiff

[![Build Status](https://travis-ci.com/agartland/hierdiff.svg?branch=master)](https://travis-ci.com/agartland/hierdiff)
[![PyPI version](https://badge.fury.io/py/afioregartland.svg)](https://badge.fury.io/py/hierdiff)
[![Coverage Status](https://coveralls.io/repos/github/agartland/hierdiff/badge.svg?branch=master)](https://coveralls.io/github/agartland/hierdiff?branch=master)

A package that is useful for clustering high-dimensional instances (e.g. T cell receptors) and testing whether clusters of instances are differentially abundant in two or more categorical conditions. The package provides d3/SVG rendering of scipy hierarchical clustering dendrograms with zooming, panning and tooltips. This uniquely allows for exploring large trees of datasets, conditioned on a categorical trait.

## Installation

```
pip install hierdiff
```

## Example

```python
import hierdiff
from scipy.spatial.distance import squareform

"""Contains categorical variable column 'trait1' and
instance counts in 'count'"""
dat, pwdist = generate_data()

res, Z = hierdiff.hcluster_tally(dat,
				                  pwmat=squareform(pwdist),
				                  x_cols=['trait1'],
				                  count_col='count',
				                  method='complete')

res = hierdiff.cluster_association_test(res, method='fishers')

"""Plot frequency of trait at nodes with p-value < 0.05"""
html = plot_hclust_props(Z, title='test_props2',
                            res=res, alpha=0.05, alpha_col='pvalue')
```

![example](https://raw.githubusercontent.com/agartland/hierdiff/master/example_hier_props.png)
