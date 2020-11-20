import pandas as pd
import numpy as np
from os.path import join as opj
import sys
#import matplotlib.pyplot as plt
#from matplotlib import cm
#import matplotlib as mpl
from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
import itertools

import json

"""jinja2 import triggers DeprecationWarning about imp module"""
from jinja2 import Environment, PackageLoader#, FileSystemLoader

from .modified_scipy_dendrogram import modified_dendrogram

__all__ = ['plot_hclust',
           'plot_hclust_props']

"""TODO:
  - Control x and y zoom independently
   https://stackoverflow.com/questions/61071276/d3-synchronizing-2-separate-zoom-behaviors/61164185#61164185"""
set1_colors = ["#e41a1c", "#377eb8", "#4daf4a",
               "#984ea3", "#ff7f00", "#ffff33",
               "#a65628", "#f781bf", "#999999"]

set3_colors = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
               "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
               "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f"]


def plot_hclust(Z, title='', leaf_counts=None):
    """Plot tree of linkage-based hierarchical clustering. Nodes
    annotated with cluster ID.

    Parameters
    ----------
    Z : linkage matrix
        Result of calling sch.linkage on a compressed pair-wise distance matrix
    leaf_counts : np.ndarray [Z.shape[0] - 1, ]
        Weights/counts for each leaf of the tree, using the same order as pwdist matrix
        used to generate linkage matrix Z.

    Returns
    -------
    html : str
        String that can be saved as HTML for viewing"""
    html = plot_hclust_props(Z, title=title, leaf_counts=leaf_counts)
    return html

def plot_hclust_props(Z, title='', res=None, alpha_col='pvalue', alpha=0.05, tooltip_cols=[], colors=None, prune_col=None, leaf_counts=None):
    """Plot tree of linkage-based hierarchical clustering, with nodes colored using stacked bars
    representing proportion of cluster members associated with specific conditions. Nodes also optionally
    annotated with pvalue, number of members or cluster ID.

    Parameters
    ----------
    Z : linkage matrix
        Result of calling sch.linkage on a compressed pair-wise distance matrix
    res : pd.DataFrame
        Result from calling hcluster_diff, with observed/frequencies and p-values for each node
    alpha_col : str
        Column in res to use for 'alpha' annotation
    alpha : float
        Threshold for plotting the stacked bars and annotation
    colors : tuple of valid colors
        Used for stacked bars of conditions at each node
    prune_col : str/column in res
        Column of res that indicates whether a result/node can be pruned from the tree.
        Tree will not print any pruned nodes that only contain other pruned nodes.
    leaf_counts : np.ndarray [Z.shape[0] - 1, ]
        Weights/counts for each leaf of the tree, using the same order as pwdist matrix
        used to generate Z linkage matrix.

    Returns
    -------
    html : str
        String that can be saved as HTML for viewing"""
    height=600
    width=900
    paths, lines, annotations, legend_data = _hclust_paths(Z, height=height, width=width,
                                                            res=res,
                                                            alpha_col=alpha_col,
                                                            alpha=alpha,
                                                            tooltip_cols=tooltip_cols,
                                                            colors=colors,
                                                            prune_col=prune_col,
                                                            leaf_counts=leaf_counts)

    #lines_df = 100 * pd.DataFrame({'x1':np.random.rand(10), 'y1':np.random.rand(10), 'x2':np.random.rand(10), 'y2':np.random.rand(10)})
    #lines_df = lines_df.assign(stroke='red', stroke_width=1.5)
    #lines_json = lines_df.to_json(orient='records')
    #circle_data = pd.DataFrame({'x':np.random.rand(10)*50 + width/2, 'y':np.random.rand(10)*50 + height/2}).to_json(orient='records')
    
    jinja_env = Environment(loader=PackageLoader('hierdiff', 'templates'))
    #jinja_env = Environment(loader=FileSystemLoader(opj(_git, 'hierdiff', 'hierdiff')))

    tree_tmp = jinja_env.get_template('tree_template.html')
    html = tree_tmp.render(mytitle=title,
                             line_data=json.dumps(lines),
                             annotation_data=json.dumps(annotations),
                             legend_data=json.dumps(legend_data),
                             path_data=json.dumps(paths),
                             height=height,
                             width=width)
    return html

def _encode(s):
    """Creates valid JSON strings that can be decoded by JS in th browser"""
    return str(s).replace('"', '@DBLQ').replace('<', '@LT').replace('>', '@GT').replace('/', '@SL')


def _hclust_paths(Z, height, width, margin=10, res=None, alpha_col='pvalue', alpha=0.05, tooltip_cols=[], colors=None, min_count=0, prune_col=None, leaf_counts=None):
    if colors is None:
        colors = set1_colors
    lines = []
    annotations = []
    paths = []

    if not res is None:
        """Use cts from every other element since cluster membership is 
        always the last feature of the counts array and is therefore the innermost loop"""
        x_val_cols = [c for c in res.columns if 'val_' in c][::2]
        x_ct_cols = [c.replace('val', 'ct') for c in x_val_cols]
        x_vals = ['|'.join(res[c].iloc[0][:-1]) for c in x_val_cols]

        legend_data = [{'label':v, 'color':c} for v,c in zip(x_vals, colors)]
    else:
        legend_data = []    

    dend = modified_dendrogram(Z, counts=leaf_counts)
    
    xscale = _linear_scale_factory((np.min(np.array(dend['icoord'])), np.max(np.array(dend['icoord']))),
                                  (0+margin, width-margin))
    yscale = _linear_scale_factory((np.min(np.array(dend['dcoord'])), np.max(np.array(dend['dcoord']))),
                                  (height-margin, 0+margin))

    for xx, yy, cid in zip(dend['icoord'], dend['dcoord'], dend['cid_list']):
        #axh.plot(xx, yy, zorder=1, lw=0.5, color='k', alpha=1)
        if not res is None:
            """Assumes there is only one matching cid (not true if multiple results are passed)"""
            cid_res = res.loc[res['cid'] == cid]
            if len(cid_res) > 1:
                print('Multiple matching results for each cluster id (i.e. cid)')
                return
            else:
                cid_res = cid_res.iloc[0]
            
            if not prune_col is None:
                """Prune (don't print) if all member results are also prune=True"""
                prune = res.loc[res['cid'].isin(cid_res['children']), prune_col].all()
                if prune:
                    continue
            paths.append(dict(coords=[[xscale(x), yscale(y)] for x,y in zip(xx, yy)], stroke='black', stroke_width=1))
            
            N = np.sum(cid_res['K_neighbors'])

            ann = [f'cid: {cid:d}',
                   f'n_uniq: {N:1.0f}',
                   f'n_ct: {np.sum(cid_res[x_ct_cols]):1.0f}']
            ann.extend(['%s: %s' % (tt, _encode(cid_res[tt])) for tt in tooltip_cols])
            annotations.append(dict(annotation=ann, x1=xscale(xx[1]), x2=xscale(xx[2]), y1=yscale(yy[1]), y2=yscale(yy[2])))
            if alpha is None or cid_res[alpha_col] <= alpha and N > min_count:
                cts = np.asarray(cid_res[x_ct_cols])
                obs = cts / np.sum(cts)
                L = (xx[2] - xx[1])
                xvec = L * np.concatenate(([0.], obs, [1.]))
                curX = xx[1]
                for i, c in zip(range(len(obs)), itertools.cycle(colors)):
                    lines.append(dict(x1=xscale(curX),
                                      x2=xscale(curX + L*obs[i]),
                                      y1=yscale(yy[1]),
                                      y2=yscale(yy[2]),
                                      stroke=c,
                                      stroke_width=10))
                    """axh.plot([curX, curX + L*obs[i]],
                             yy[1:3],
                             color=c,
                             lw=10,
                             solid_capstyle='butt')"""
                    curX += L*obs[i]

        else:
            paths.append(dict(coords=[[xscale(x), yscale(y)] for x,y in zip(xx, yy)], stroke='black', stroke_width=1))
            s = ['cid: %d' % cid]
            annotations.append(dict(annotation=s, x1=xscale(xx[1]), x2=xscale(xx[2]), y1=yscale(yy[1]), y2=yscale(yy[2])))
        paths = _translate_paths(paths)
    return paths, lines, annotations, legend_data

def _linear_scale_factory(domain, rng):
    scalar = (rng[1] - rng[0]) / (domain[1] - domain[0])
    offset = rng[0] - scalar*domain[0]

    def _scaler(x):
        return x * scalar + offset
    return _scaler

def _translate_paths(paths):
    """Simple translation of path coordinates to SVG path string"""
    svg = []
    for j,p in enumerate(paths):
        tmp = ''
        for i,c in enumerate(p['coords']):
            if i == 0:
                tmp += 'M%f,%f' % tuple(c)
            else:
                tmp += 'L%f,%f' % tuple(c)
        paths[j]['str_coords'] = tmp
    return paths
