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

Rerun tests
Write association test function
Write a general function that accepts cluster labels? Should be easy enough

 * Add useful marginal frequencies to NNdiff output (like hierdiff)
 * Make sure RR and OR make sense. Maybe include only one for fisher test?
 * Separate function that creates the count matrix for either NN or hier (or another) clustering
 * Functions for cluster introspection are TCR specific and should be included, while the basic
   stats could be largely excluded (included by example)
 * Output should allow for easy testing with existing methods in statsmodels or otherwise,
   that can be easily demonstrated?
 * Plot function should take the counts output providing introspection with or without pvalues/testing"""

def _prep_counts(cdf, xcols, ycol, count_col=None):
    """Returns a pd.DataFrame with rows for each (x_col, x_val) combination
    and columns for cluster memberhship (member+, member-)

    When values are raveled the values of a 2 x 2 (a, b, c, d) are:
        x_col_0|x_val_0+/member+
        x_col_0|x_val_0+/member-
        x_col_0|x_val_0-/member+
        x_col_0|x_val_0-/member-

    which means OR > 1 using fishersapi is enrichment"""
    if count_col is None:
        cdf = cdf.assign(count=1)
        count_col = 'count'
    counts = cdf.groupby(xcols + [ycol], sort=True)[count_col].agg(np.sum).unstack(ycol).fillna(0)
    for i in [0, 1]:
        if not i in counts.columns:
            counts.loc[:, i] = np.zeros(counts.shape[0])

    counts = counts[[1, 0]]
    return counts

def neighborhood_tally(df, pwmat, x_cols, count_col='count', knn_neighbors=50, knn_radius=None, subset_ind=None, cluster_ind=None):
    """Forms a cluster around each row of df and tallies the number of instances with/without traits
    in x_cols. The contingency table for each cluster/row of df can be used to test for enrichments of the traits
    in x_cols with the distances between each row provided in pwmat. The neighborhood is defined by the K closest neighbors
    using pairwise distances in pwmat, or defined by a distance radius.

    For TCR analysis this can be used to test whether the TCRs in a neighborhood are associated with a certain trait or
    phenotype. You can use hier_diff.cluster_association_test with the output of this function to test for
    significnt enrichment.

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
    print('cluster_ind', cluster_ind.shape, cluster_ind)
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
        y = (pwmat[ii, :] <= R).astype(float)
        K = np.sum(y)

        cdf = df.assign(**{ycol:y})[[ycol, count_col] + x_cols]
        counts = _prep_counts(cdf, x_cols, ycol, count_col)

        out = {'CTS%d' % i:v for i,v in enumerate(counts.values.ravel())}

        uY = [1, 0]
        out.update({'x_col_%d' % i:v for i,v in enumerate(x_cols)})
        for i,xvals in enumerate(counts.index.tolist()):
            if type(xvals) is tuple:
                val = '|'.join(xvals)
            else:
                val = xvals
            out.update({'x_val_%d' % i:val,
                        'x_freq_%d' % i: counts.loc[xvals, 1] / counts.loc[xvals].sum()})

        out.update({'index':clonei,
                    'neighbors':list(df.index[np.nonzero(y)[0]]),
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
        y = np.zeros(n)
        y[m] = 1

        K = np.sum(y)
        R = np.max(pwmat[m, :][:, m])

        cdf = clone_tmp.assign(**{ycol:y})[[ycol, count_col] + x_cols]
        counts = _prep_counts(cdf, x_cols, ycol, count_col)

        out = {'CTS%d' % i:v for i,v in enumerate(counts.values.ravel())}

        uY = [1, 0]
        out.update({'x_col_%d' % i:v for i,v in enumerate(x_cols)})
        for i,xvals in enumerate(counts.index.tolist()):
            if type(xvals) is tuple:
                val = '|'.join(xvals)
            else:
                val = xvals
            out.update({'x_val_%d' % i:val,
                        'x_freq_%d' % i: counts.loc[xvals, 1] / counts.loc[xvals].sum()})
        
        out.update({'cid':cid,
                    'members':list(clone_tmp.index[m]),
                    'members_i':m,
                    'children':cluster_members[cid],
                    'K_neighbors':K,
                    'R_radius':R})
        res.append(out)

    res_df = pd.DataFrame(res)
    return res_df, Z