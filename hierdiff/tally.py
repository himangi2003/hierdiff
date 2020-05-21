import pandas as pd
import numpy as np
import itertools
import warnings

import scipy.cluster.hierarchy as sch
from scipy.spatial import distance

import fishersapi

__all__ = ['hcluster_tally',
		   'neighborhood_tally']

"""TODO:
 * Write a general function that accepts cluster labels? Should be easy enough
 * Functions for cluster introspection are TCR specific and should be included, while the basic
   stats could be largely excluded (included by example)
 * Plot function should take the counts output providing introspection with or without pvalues/testing"""

def _counts_to_cols(counts):
    """Encodes the counts Series as columns that can be added to a takky result row

    Example counts table:

    trait1  trait2  cmember
    0       0       0          233
                    1          226
            1       0           71
                    1           79
    1       0       0            0
                    1            0
            1       0            0
                    1            9"""
    j = 0
    cols = tuple(counts.index.names)
    levels = []
    for name, lev in zip(counts.index.names, counts.index.levels):
        if len(lev) == 1:
            """This solves the problem of when a variable with one level is included
                by accident or e.g. all instances are cmember = 1 (top node, big R)"""
            if name == 'cmember':
                levels.append(('MEM+', 'MEM-'))    
            elif not 0 in lev:
                levels.append(tuple(sorted((0, lev[0]))))
            else:
                levels.append(tuple(sorted(('REF', lev[0]))))
        else:
            levels.append(tuple(lev))
    levels = tuple(levels)

    out = {'ct_columns':cols}
    for xis in itertools.product(*(range(len(u)) for u in levels)):
        vals = []
        for ui, (col, u, xi) in enumerate(zip(counts.index.names, levels, xis)):
            vals.append(u[xi])
        try:
            ct = counts.loc[tuple(vals)]
        except KeyError:
            ct = 0
        out.update({'val_%d' % j:tuple(vals),
                    'ct_%d' % j:ct})
        j += 1
    return out

def _dict_to_nby2(d):
    """Takes the encoded columns of counts from a results row and re-creates the counts table"""
    cols = d['ct_columns']
    n = np.max([int(k.split('_')[1]) for k in d if 'val_' in k]) + 1
    cts = [d['ct_%d' % j] for j in range(n)]
    idx = pd.MultiIndex.from_tuples([d['val_%d' % j] for j in range(n)], names=cols)
    counts = pd.Series(cts, index=idx)
    return counts

def _prep_counts(cdf, xcols, ycol, count_col=None):
    """Returns a dict with keys that can be added to a result row to store tallies

    For a 2x2 table the data is encoded as follows
    X+MEM+ encodes the first level in Y (cluster membership = MEM+) and X
    and out contains columns named val_j and ct_j where j is ravel order, such that
    the values of a 2x2 table (a, b, c, d) are:
        ct_0    X-MEM+    a    First level of X and a cluster member ("M+" which sorts before "M-" so is also first level)
        ct_1    X-MEM-    b    First level of X and a non member
        ct_2    X+MEM+    c    Second level of X and a cluster member
        ct_3    X+MEM-    d    Second level of X and a non member

    val_j also encodes explictly the values of the X levels and cluster membership indicator (MEM+ = member)
    This means that an OR > 1 is enrichment of the SECOND level of X in the cluster.

    Longer tables are stored in ravel order with ct_j/val_j pairs with val_j containing the values
    of each column/variable.

    Key "ct_columns" contains the xcols and ycol as a list
    Ket levels contains the levels of xcols and ycol as lists from a pd.Series.MultiIndex"""
    if count_col is None:
        cdf = cdf.assign(count=1)
        count_col = 'count'
    counts = cdf.groupby(xcols + [ycol], sort=True)[count_col].agg(np.sum)
    out = _counts_to_cols(counts)
    counts = _dict_to_nby2(out)
    out['levels'] = [list(lev) for lev in counts.index.levels]

    if len(xcols) == 1 and counts.shape[0] == 4:
        """For a 2x2 add helpful count and probability columns
        Note that the first level of a column/variable is "negative"
        because its index in levels is 0"""
        n = counts.sum()
        levels = counts.index.levels
        tmp = {'X+MEM+':counts[(levels[0][1], 'MEM+')],
               'X+MEM-':counts[(levels[0][1], 'MEM-')],
               'X-MEM+':counts[(levels[0][0], 'MEM+')],
               'X-MEM-':counts[(levels[0][0], 'MEM-')]}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tmp.update({'X_marg':(tmp['X+MEM+'] + tmp['X+MEM-']) / n,
                        'MEM_marg':(tmp['X+MEM+'] + tmp['X-MEM+']) / n,
                        'X|MEM+':tmp['X+MEM+'] / (tmp['X+MEM+'] + tmp['X-MEM+']),
                        'X|MEM-':tmp['X+MEM-'] / (tmp['X+MEM-'] + tmp['X-MEM-']),
                        'MEM|X+':tmp['X+MEM+'] / (tmp['X+MEM+'] + tmp['X+MEM-']),
                        'MEM|X-':tmp['X-MEM+'] / (tmp['X-MEM+'] + tmp['X-MEM-'])})
        out.update(tmp)
    return out

def neighborhood_tally(df, pwmat, x_cols, count_col='count', knn_neighbors=50, knn_radius=None, subset_ind=None, cluster_ind=None):
    """Forms a cluster around each row of df and tallies the number of instances with/without traits
    in x_cols. The contingency table for each cluster/row of df can be used to test for enrichments of the traits
    in x_cols with the distances between each row provided in pwmat. The neighborhood is defined by the K closest neighbors
    using pairwise distances in pwmat, or defined by a distance radius.

    For TCR analysis this can be used to test whether the TCRs in a neighborhood are associated with a certain trait or
    phenotype. You can use hier_diff.cluster_association_test with the output of this function to test for
    significnt enrichment.

    Note on output: val_j/ct_j pairs provide the counts for each element of the n x 2 continency table where the last
    dimension is always 'cmember' (MEM+ or MEM-) indicating cluster membership for each row. The X+MEM+ notation
    is provided for convenience for 2x2 tables and X+ indicates the second level of x_col when sorted (e.g. 1 for [0, 1]).

    Params
    ------
    df : pd.DataFrame [nclones x metadata]
        Contains metadata for each clone.
    pwmat : np.ndarray [nclones x nclones]
        Square distance matrix for defining neighborhoods
    x_cols : list
        List of columns to be tested for association with the neighborhood
    count_col : str
        Column in df that specifies counts.
        Default none assumes count of 1 cell for each row.
    knn_neighbors : int
        Number of neighbors to include in the neighborhood, or fraction of all data if K < 1
    knn_radius : float
        Radius for inclusion of neighbors within the neighborhood.
        Specify K or R but not both.
    subset_ind : None or np.ndarray with partial index of df, optional
        Provides option to tally counts only within a subset of df, but to maintain the clustering
        of all individuals. Allows for one clustering of pooled TCRs,
        but tallying/testing within a subset (e.g. participants or conditions)
    cluster_ind : None or np.ndarray
        Indices into df specifying the neighborhoods for testing.

    Returns
    -------
    res_df : pd.DataFrame [nclones x results]
        Results from testing the neighborhood around each clone."""
    if knn_neighbors is None and knn_radius is None:
        raise(ValueError('Must specify K or radius'))
    if not knn_neighbors is None and not knn_radius is None:
        raise(ValueError('Must specify K or radius (not both)'))

    if pwmat.shape[0] != pwmat.shape[1] or pwmat.shape[0] != df.shape[0]:
        pwmat = distance.squareform(pwmat)
        if pwmat.shape[0] != pwmat.shape[1] or pwmat.shape[0] != df.shape[0]:
            raise ValueError('Shape of pwmat %s does not match df %s' % (pwmat.shape, df.shape))

    ycol = 'cmember'
    if cluster_ind is None:
        cluster_ind = df.index

    if not subset_ind is None:
        clone_tmp = df.copy()
        """Set counts to zero for all clones that are not in the group being tested"""
        not_ss = [ii for ii in df.index if not ii in subset_ind]
        clone_tmp.loc[not_ss, count_col] = 0
    else:
        clone_tmp = df
    
    res = []
    for clonei in cluster_ind:
        ii = np.nonzero(df.index == clonei)[0][0]
        if not knn_neighbors is None:
            if knn_neighbors < 1:
                frac = knn_neighbors
                K = int(knn_neighbors * df.shape[0])
                # print('Using K = %d (%1.0f%% of %d)' % (K, 100*frac, n))
            else:
                K = int(knn_neighbors)
            R = np.partition(pwmat[ii, :], K + 1)[K]
        else:
            R = knn_radius
        y_lu = {True:'MEM+', False:'MEM-'}
        y_float = (pwmat[ii, :] <= R).astype(float)
        y = np.array([y_lu[yy] for yy in y_float])
        K = np.sum(y_float)

        cdf = df.assign(**{ycol:y})[[ycol, count_col] + x_cols]
        out = _prep_counts(cdf, x_cols, ycol, count_col)

        out.update({'index':clonei,
                    'neighbors':list(df.index[np.nonzero(y_float)[0]]),
                    'K_neighbors':K,
                    'R_radius':R})

        res.append(out)

    res_df = pd.DataFrame(res)
    print('res_df', res_df.shape)
    return res_df

def hcluster_tally(df, pwmat, x_cols, Z=None, count_col='count', subset_ind=None, method='complete', optimal_ordering=True):
    """Tests for association of categorical variables in x_cols with each cluster/node
    in a hierarchical clustering of clones with distances in pwmat.

    Use Fisher's exact test (test='fishers') to detect enrichment/association of the neighborhood/cluster
    with one variable.

    Tests the 2 x 2 table for each clone:

    +----+----+-------+--------+
    |         |    Cluster     |
    |         +-------+--------+
    |         | Y     |    N   |
    +----+----+-------+--------+
    |VAR |  1 | a     |    b   |
    |    +----+-------+--------+
    |    |  0 | c     |    d   |
    +----+----+-------+--------+

    Use the chi-squared test (test='chi2') or logistic regression (test='logistic') to detect association across multiple variables.
    Note that with small clusters Chi-squared tests and logistic regression are unreliable. It is possible
    to pass an L2 penalty to the logistic regression using l2_alpha in kwargs, howevere this requires a permutation
    test (nperms also in kwargs) to compute a value.

    Use the Cochran-Mantel-Haenszel test (test='chm') to test stratified 2 x 2 tables: one VAR vs. cluster, over sever strata
    defined in other variables. Use x_cols[0] as the primary (binary) variable and other x_cols for the categorical
    strata-defining variables. This tests the overall null that OR = 1 for x_cols[0]. A test is also performed
    for homogeneity of the ORs among the strata (Breslow-Day test).

    Params
    ------
    df : pd.DataFrame [nclones x metadata]
        Contains metadata for each clone.
    pwmat : np.ndarray [nclones x nclones]
        Square or compressed (see scipy.spatial.distance.squareform) distance
        matrix for defining clusters.
    x_cols : list
        List of columns to be tested for association with the neighborhood
    count_col : str
        Column in df that specifies counts.
        Default none assumes count of 1 cell for each row.
    subset_ind : partial index of df, optional
        Provides option to tally counts only within a subset of df, but to maintain the clustering
        of all individuals. Allows for one clustering of pooled TCRs,
        but tallying/testing within a subset (e.g. participants or conditions)
    min_n : int
        Minimum size of a cluster for it to be tested.
    optimal_ordering : bool
        If True, the linkage matrix will be reordered so that the distance between successive
        leaves is minimal. This results in a more intuitive tree structure when the data are
        visualized. defaults to False, because this algorithm can be slow, particularly on large datasets.

    Returns
    -------
    res_df : pd.DataFrame [nclusters x results]
        A 2x2 table for each cluster.
    Z : linkage matrix [nclusters, df.shape[0] - 1, 4]
        Clustering result returned from scipy.cluster.hierarchy.linkage"""

    ycol = 'cmember'

    if Z is None:
        if pwmat.shape[0] == pwmat.shape[1] and pwmat.shape[0] == df.shape[0]:
            compressed = distance.squareform(pwmat)
        else:
            compressed = pwmat
            pwmat = distance.squareform(pwmat)
        Z = sch.linkage(compressed, method=method, optimal_ordering=optimal_ordering)

    else:
        """Shape of correct Z asserted here"""
        if not Z.shape == (df.shape[0] - 1, 4):
            raise ValueError('First dimension of Z (%d) does not match that of df (%d,)' % (Z.shape[0], df.shape[0]))
    
    clusters = {}
    for i, merge in enumerate(Z):
        """Cluster ID number starts at a number after all the leaves"""
        cid = 1 + i + Z.shape[0]
        clusters[cid] = [merge[0], merge[1]]

    def _get_indices(clusters, i):
        if i <= Z.shape[0]:
            return [int(i)]
        else:
            return _get_indices(clusters, clusters[i][0]) + _get_indices(clusters, clusters[i][1])

    def _get_cluster_indices(clusters, i):
        if i <= Z.shape[0]:
            return []
        else:
            return [int(i)] + _get_cluster_indices(clusters, clusters[i][0]) + _get_cluster_indices(clusters, clusters[i][1])

    members = {i:_get_indices(clusters, i) for i in range(Z.shape[0] + 1, max(clusters.keys()) + 1)}
    """Note that the list of clusters within each cluster includes the current cluster"""
    cluster_members = {i:_get_cluster_indices(clusters, i) for i in range(Z.shape[0] + 1, max(clusters.keys()) + 1)}

    n = df.shape[0]

    res = []
    """Setting non-group counts to zero"""
    if not subset_ind is None:
        clone_tmp = df.copy()
        """Set counts to zero for all clones that are not in the group being tested"""
        not_ss = [ii for ii in df.index if not ii in subset_ind]
        clone_tmp.loc[not_ss, count_col] = 0
    else:
        clone_tmp = df

    for cid, m in members.items():
        not_m = [i for i in range(n) if not i in m]
        y_float = np.zeros(n, dtype=np.int)
        y_float[m] = 1

        y_lu = {1:'MEM+', 0:'MEM-'}
        y = np.array([y_lu[yy] for yy in y_float])

        K = np.sum(y_float)
        R = np.max(pwmat[m, :][:, m])

        cdf = clone_tmp.assign(**{ycol:y})[[ycol, count_col] + x_cols]
        out = _prep_counts(cdf, x_cols, ycol, count_col)

        out.update({'cid':cid,
                    'members':list(clone_tmp.index[m]),
                    'members_i':m,
                    'children':cluster_members[cid],
                    'K_neighbors':K,
                    'R_radius':R})
        res.append(out)

    res_df = pd.DataFrame(res)
    return res_df, Z