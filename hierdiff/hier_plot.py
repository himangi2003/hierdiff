import pandas as pd
import numpy as np
from os.path import join as opj
import sys
#import matplotlib.pyplot as plt
#from matplotlib import cm
#import matplotlib as mpl
from scipy.spatial import distance
import scipy.cluster.hierarchy as sch

import json

"""jinja2 import triggers DeprecationWarning about imp module"""
from jinja2 import Environment, PackageLoader#, FileSystemLoader

__all__ = ['plot_hclust',
           'plot_hclust_props']

"""TODO:
 - Add tooltips
 - Add ability to export/dowload SVG
   https://github.com/edeno/d3-save-svg
 - Add simplified entry points
 - SVG should resize to window and tree should scale, dynamically
 - Control x and y zoom independently
   https://stackoverflow.com/questions/61071276/d3-synchronizing-2-separate-zoom-behaviors/61164185#61164185
 """
set1_colors = ["#e41a1c", "#377eb8", "#4daf4a",
               "#984ea3", "#ff7f00", "#ffff33",
               "#a65628", "#f781bf", "#999999"]

set3_colors = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
               "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
               "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f"]


def plot_hclust(Z, height=600, width=900, title=''):
    """Plot tree of linkage-based hierarchical clustering. Nodes
    annotated with cluster ID.

    Parameters
    ----------
    Z : linkage matrix
        Result of calling sch.linkage on a compressed pair-wise distance matrix

    Returns
    -------
    html : str
        String that can be saved as HTML for viewing"""
    html = plot_hclust_props(Z, height=height, width=width, title=title)
    return html

def plot_hclust_props(Z, height=600, width=900, title='', res=None, alpha_col='pvalue', alpha=0.05, colors=None):
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
    labels : list of condition labels
        Matched to tuples of colors and frequencies in res

    Returns
    -------
    html : str
        String that can be saved as HTML for viewing"""

    paths, lines, annotations = _hclust_paths(Z, height, width, res=res, alpha_col=alpha_col, alpha=alpha, colors=colors)

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
                             path_data=json.dumps(paths),
                             height=height,
                             width=width)
    return html

def _hclust_paths(Z, height, width, margin=10, res=None, alpha_col='pvalue', alpha=0.05, colors=None, min_count=0):
    if colors is None:
        colors = set1_colors
    lines = []
    annotations = []
    paths = []

    if not res is None:
        x_val_cols = [c for c in res.columns if 'x_val_' in c]
        x_freq_cols = [c for c in res.columns if 'x_freq_' in c]
        x_vals = [res[c].iloc[0] for c in x_val_cols]
    

    dend = sch.dendrogram(Z, no_plot=True,
                             color_threshold=None,
                             link_color_func=lambda lid: hex(lid),
                             above_threshold_color='FFFFF')
    
    xscale = _linear_scale_factory((np.min(np.array(dend['icoord'])), np.max(np.array(dend['icoord']))),
                                  (0+margin, width-margin))
    yscale = _linear_scale_factory((np.min(np.array(dend['dcoord'])), np.max(np.array(dend['dcoord']))),
                                  (height-margin, 0+margin))

    annotateCount = 0
    for xx, yy, hex_cid in zip(dend['icoord'], dend['dcoord'], dend['color_list']):
        paths.append(dict(coords=[[xscale(x), yscale(y)] for x,y in zip(xx, yy)], stroke='black', stroke_width=1))
        #axh.plot(xx, yy, zorder=1, lw=0.5, color='k', alpha=1)
        cid = int(hex_cid, 16)
        if not res is None:
            cid_res = res.loc[res['cid'] == cid].iloc[0]
            
            N = np.sum(cid_res['K_neighbors'])
            ann = ['cid: %d' % cid,
                   'n: %1.0f' % N,
                   '%s: %1.3f' % (alpha_col, cid_res[alpha_col])]
            annotations.append(dict(annotation=ann, x1=xscale(xx[1]), x2=xscale(xx[2]), y1=yscale(yy[1]), y2=yscale(yy[2])))
            if alpha is None or cid_res[alpha_col] <= alpha and N > min_count:
                obs = np.asarray(cid_res[x_freq_cols])
                obs = obs / np.sum(obs)
                L = (xx[2] - xx[1])
                xvec = L * np.concatenate(([0.], obs, [1.]))
                curX = xx[1]
                for i in range(len(obs)):
                    c = colors[i]
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
            s = ['cid: %d' % cid]
            annotations.append(dict(annotation=s, x1=xscale(xx[1]), x2=xscale(xx[2]), y1=yscale(yy[1]), y2=yscale(yy[2])))
        paths = _translate_paths(paths)
    return paths, lines, annotations

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

def _plot_hclust_props(figh, Z, res, alpha_col='pvalue', alpha=0.05, colors=None, ann='N', xLim=None, maxY=None, min_count=20):
    """Plot tree of linkage-based hierarchical clustering, with nodes colored with stacked bars
    representing proportion of cluster members associated with specific conditions. Nodes also optionally
    annotated with pvalue, number of members or cluster ID.

    Parameters
    ----------
    figh : mpl Figure() handle
    Z : linkage matrix
        Result of calling sch.linkage on a compressed pair-wise distance matrix 
    res : pd.DataFrame
        Result from calling testHClusters, with observed/frequencies and p-values for each node
    alpha_col : str
        Column in res to use for 'alpha' annotation
    alpha : float
        Threshold for plotting the stacked bars and annotation
    colors : tuple of valid colors
        Used for stacked bars of conditions at each node
    labels : list of condition labels
        Matched to tuples of colors and frequencies in res
    ann : str
        Indicates how nodes should be annotated: N, alpha, CID supported
    xLim : tuple
        Apply x-lims after plotting to focus on particular part of the tree"""

    """OLD MATPLOTLIB CODE FOR TRANSLATION"""
    '''
    x_val_cols = [c for c in res.columns if 'x_val_' in c]
    x_freq_cols = [c for c in res.columns if 'x_freq_' in c]
    x_vals = [res[c].iloc[0] for c in x_val_cols]
    
    if colors is None:
        colors = mpl.cm.Set1.colors[:len(x_vals)]
    
    dend = sch.dendrogram(Z, no_plot=True,
                             color_threshold=None,
                             link_color_func=lambda lid: hex(lid),
                             above_threshold_color='FFFFF')
    figh.clf()
    axh = plt.axes((0.05, 0.07, 0.8, 0.8), facecolor='w')

    lowestY = None
    annotateCount = 0
    for xx, yy, hex_cid in zip(dend['icoord'], dend['dcoord'], dend['color_list']):
        cid = int(hex_cid, 16)
        cid_res = res.loc[res['cid'] == cid].iloc[0]

        xx = np.array(xx) / 10
        axh.plot(xx, yy, zorder=1, lw=0.5, color='k', alpha=1)

        N = np.sum(cid_res['K_neighbors'])
        if alpha is None or cid_res[alpha_col] <= alpha and N > min_count:
            obs = np.asarray(cid_res[x_freq_cols])
            obs = obs / np.sum(obs)
            L = (xx[2] - xx[1])
            xvec = L * np.concatenate(([0.], obs, [1.]))
            curX = xx[1]
            for i in range(len(obs)):
                c = colors[i]
                axh.plot([curX, curX + L*obs[i]],
                         yy[1:3],
                         color=c,
                         lw=10,
                         solid_capstyle='butt')
                curX += L*obs[i]
            if ann == 'N':
                s = '%1.0f' % N
            elif ann == 'CID':
                s = cid
            elif ann == 'alpha':
                if cid_res[alpha_col] < 0.001:
                    s = '< 0.001'
                else:
                    s = '%1.3f' % cid_res[alpha_col]
            if not ann == '':# and annotateCount < annC:
                xy = (xx[1] + L/2, yy[1])
                # print(s,np.round(xy[0]), np.round(xy[1]))
                annotateCount += 1
                axh.annotate(s,
                             xy=xy,
                             size='x-small',
                             horizontalalignment='center',
                             verticalalignment='center')
            if lowestY is None or yy[1] < lowestY:
                lowestY = yy[1]
    yl = axh.get_ylim()
    if not lowestY is None:
        yl0 = 0.9*lowestY
    else:
        yl0 = yl[0]
    if not maxY is None:
        yl1 = maxY
    else:
        yl1 = yl[1]
    axh.set_ylim((yl0, yl1))
    
    axh.set_yticks(())
    if not xLim is None:
        if xLim[1] is None:
            xl1 = axh.get_xlim()[1]
            xLim = (xLim[0], xl1)
        axh.set_xlim(xLim)
    else:
        xLim = axh.get_xlim()

    xt = [x for x in range(0, Z.shape[0]) if x <= xLim[1] and x>= xLim[0]]
    xt = xt[::len(xt) // 10]
    # xtl = [x//10 for x in xt]
    axh.set_xticks(xt)
    # axh.set_xticklabels(xtl)
    legh = axh.legend([plt.Rectangle((0,0), 1, 1, color=c) for c in colors],
            x_vals,
            loc='upper left', bbox_to_anchor=(1, 1))
    '''