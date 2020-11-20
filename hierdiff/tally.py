import pandas as pd
import numpy as np
import itertools
import warnings
import scipy.sparse

import scipy.cluster.hierarchy as sch
from scipy.spatial import distance

from joblib import Parallel, delayed

__all__ = ['hcluster_tally',
		   'neighborhood_tally',
           'running_neighborhood_tally',
           'any_cluster_tally',
           'find_radius_members']

"""TODO:
 * Incorporate running_neighbors into TCRdist, wrapping the standard metrics so they can work
   easily.
 * Verify that running_neighbor uses all CPUs and less memory: see how it could be further optimized
   with joblib caching, expecially for the metrics that include the CDR2 and CDR1.5 etc.
"""

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
            elif isinstance(lev[0], int):
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
        except (pd.core.indexing.IndexingError, KeyError):
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

def _prep_counts(cdf, xcols, ycol, count_col):
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

def neighborhood_tally(df_pop, pwmat, x_cols, df_centroids=None, count_col='count', knn_neighbors=None, knn_radius=25):
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
    df_pop : pd.DataFrame [nclones x metadata]
        Contains metadata for each clone in the population to be tallied.
    pwmat : np.ndarray [df_centroids.shape[0] x df_pop.shape[0]]
        Pairwise distance matrix for defining neighborhoods.
        Number of rows in pwmat must match the number of rows in df_centroids,
        which may be the number of rows in df_pop if df_centroids=None
    x_cols : list
        List of columns to be tested for association with the neighborhood
    df_centroids : pd.DataFrame [nclones x 1]
        An optional DataFrame containing clones that will act as centroids in the
        neighborhood clustering. These can be a subset of df_pop or not, however
        the number of rows in df_centroids must match the number of rows in pwmat.
        If df_centroids=None then df_centroids = df_pop and all clones in df_pop
        are used.
    count_col : str
        Column in df that specifies counts.
        Default none assumes count of 1 cell for each row.
    knn_neighbors : int
        Number of neighbors to include in the neighborhood, or fraction of all data if K < 1
    knn_radius : float
        Radius for inclusion of neighbors within the neighborhood.
        Specify K or R but not both.

    Returns
    -------
    res_df : pd.DataFrame [nclones x results]
        Counts of clones within each neighborhood, grouped by x_cols.
        The "neighbors" column provides the pd.DataFrame indices of the elements in
        df_pop that are within the neighborhood of each centroid (not the integer/vector
        based indices)"""
    if knn_neighbors is None and knn_radius is None:
        raise(ValueError('Must specify K or radius'))
    if not knn_neighbors is None and not knn_radius is None:
        raise(ValueError('Must specify K or radius (not both)'))

    if df_centroids is None:
        df_centroids = df_pop
        if pwmat.shape[0] != df_pop.shape[0]:
            raise ValueError(f'Number of rows in pwmat {pwmat.shape[0]} does not match df_pop {df_pop.shape[0]}')
        if pwmat.shape[1] != df_pop.shape[0]:
            raise ValueError(f'Number of columns in pwmat {pwmat.shape[1]} does not match df_pop {df_pop.shape[0]}')
    else:
        if pwmat.shape[0] != df_centroids.shape[0]:
            raise ValueError(f'Number of rows in pwmat {pwmat.shape[0]} does not match df_centroids {df_centroids.shape[0]}')
        if pwmat.shape[1] != df_pop.shape[0]:
            raise ValueError(f'Number of columns in pwmat {pwmat.shape[1]} does not match df_pop {df_pop.shape[0]}')

    if count_col is None:
        df = df_pop.assign(count=1)
        count_col = 'count'

    ycol = 'cmember'
        
    res = []
    for ii in range(df_centroids.shape[0]):
        if not knn_neighbors is None:
            if knn_neighbors < 1:
                frac = knn_neighbors
                K = int(knn_neighbors * df_pop.shape[0])
                # print('Using K = %d (%1.0f%% of %d)' % (K, 100*frac, n))
            else:
                K = int(knn_neighbors)
            if scipy.sparse.issparse(pwmat):
                R = np.partition(pwmat[ii, :].data, K)[K]
            else:
                R = np.partition(pwmat[ii, :], K)[K]
        else:
            R = knn_radius
        y_lu = {True:'MEM+', False:'MEM-'}

        """Possible efficiency issue here for large sparse matrices since
        a whole column of MEM+/MEM- labels is created"""
        y_float = find_radius_members(ii, pwmat, radius=R)
        y = np.array([y_lu[yy] for yy in y_float])
        K = int(np.sum(y_float))

        cdf = df_pop.assign(**{ycol:y})[[ycol, count_col] + x_cols]
        out = _prep_counts(cdf, x_cols, ycol, count_col)

        out.update({'index':ii,
                    'neighbors':list(df_pop.index[np.nonzero(y_float)[0]]),
                    'K_neighbors':K,
                    'R_radius':R})

        res.append(out)

    res_df = pd.DataFrame(res)
    return res_df

def find_radius_members(i, pwmat, radius, return_indices=False):
    """Find NN columns in pwmat[i, :] (iith row) with D <= radius.
    
    Accepts a sparse or non-sparse array. Sparse array should be csr_matrix
    and should encode D=0 as 0, with D>radius as NA/sparse.

    Return a binary indicator vector of shape (pwmat.shape[1], ) and
    dtype=float, with a 1 for every distance entry in the iith row
    of the pwmat with D <= radius

    Parameters
    ----------
    i : int
        Row index in pwmat
    pwmat : np.ndarray or scipy.sparse.csr_matrix
        Square or rectangular distance matrix
    radius : float or int (dtype = pwmat dtype preferably
    return_indices : bool
        Default, False, returns binary/bool index into cols of pwmat.
        True, returns "non-zero" integer column indices into pwmat

    Returns
    -------
    indicator or col_indices : bool, pwmat.shape[1] or vector of integers
    """
    if scipy.sparse.issparse(pwmat):
        # SLOW
        # row = np.asarray(pwmat[i, :].todense())
        # row[row == 0] = radius + 1
        # return np.nonzero(row <= radius)[1]
        ind = np.nonzero(pwmat.data[pwmat.indptr[i]:pwmat.indptr[i+1]] <= radius)[0]
        col_indices = pwmat.indices[pwmat.indptr[i]:pwmat.indptr[i+1]][ind]
        if return_indices:
            return col_indices
        else:
            y_float = np.zeros(pwmat.shape[1], dtype=bool)
            y_float[col_indices] = True
            return y_float
    else:
        if not return_indices:
            y_float = (pwmat[i, :] <= radius)
            return y_float
        else:
            col_indices = np.nonzero(y_float)[0]
            return col_indices
    
    

def any_cluster_tally(df, cluster_df, x_cols, cluster_ind_col='neighbors', count_col='count'):
    """Tallies clones inside (outside) each cluster for testing enrichment of other categorical
    variables defined by x_cols in df. Clusters are defined in cluster_df using the cluster_ind_col
    (default: 'neighbors') which should contain *positional* indices into df for cluster members.
    
    This function only organizes the counts for testing such that each row of the output represents
    a cluster that could be tested for enrichment.

    As an example, one could use Fisher's exact test to detect enrichment/association of the
    neighborhood/cluster with one variable.

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

    This and other tests are available with the cluster_association_test function that takes the output
    of this function as input.

    Params
    ------
    df : pd.DataFrame [nclones x metadata]
        Contains metadata for each clone.
    cluster_df : pd.DataFrame, one row per cluster
        Contains the column in cluster_ind_col (default: "neighbors") that should
        contain positional indices into df indicating cluster membership
    x_cols : list
        List of columns to be tested for association with the neighborhood
    count_col : str
        Column in df that specifies counts.
        Default none assumes count of 1 cell for each row.
    cluster_ind_col : str, column in cluster_df
        Values should be lists or tuples of positional indices into df


    Returns
    -------
    res_df : pd.DataFrame [nclusters x results]
        A 2xN table for each cluster."""

    ycol = 'cmember'

    if count_col is None:
        df = df.assign(count=1)
        count_col = 'count'

    n = df.shape[0]

    res = []
    for cid, m in cluster_df[cluster_ind_col].values:
        not_m = [i for i in range(n) if not i in m]
        y_float = np.zeros(n, dtype=np.int)
        y_float[m] = 1

        y_lu = {1:'MEM+', 0:'MEM-'}
        y = np.array([y_lu[yy] for yy in y_float])

        K = int(np.sum(y_float))

        cdf = df.assign(**{ycol:y})[[ycol, count_col] + x_cols]
        out = _prep_counts(cdf, x_cols, ycol, count_col)

        out.update({'cid':cid,
                    'neighbors':list(df.index[m]),
                    'neighbors_i':m,
                    'K_neighbors':K})
        res.append(out)

    res_df = pd.DataFrame(res)
    return res_df


def hcluster_tally(df, pwmat, x_cols, Z=None, count_col='count', subset_ind=None, method='complete', optimal_ordering=True):
    """Hierarchical clustering of clones with distances in pwmat. Tallies clones inside (outside) each cluster in preparation
    for testing enrichment of other categorical variables defined by x_cols. This function only organizes the counts for testing
    such that each row of the output represents a cluster that could be tested for enrichment.

    One example test is Fisher's exact test to detect enrichment/association of the neighborhood/cluster
    with one binary variable.

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

    This and other tests are available with the cluster_association_test function that takes the output
    of this function as input.
    
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
    optimal_ordering : bool
        If True, the linkage matrix will be reordered so that the distance between successive
        leaves is minimal. This results in a more intuitive tree structure when the data are
        visualized. defaults to False, because this algorithm can be slow, particularly on large datasets.

    Returns
    -------
    res_df : pd.DataFrame [nclusters x results]
        A 2xN table for each cluster.
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
    
    if count_col is None:
        df = df.assign(count=1)
        count_col = 'count'

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

        K = int(np.sum(y_float))
        R = np.max(pwmat[m, :][:, m])

        cdf = clone_tmp.assign(**{ycol:y})[[ycol, count_col] + x_cols]
        out = _prep_counts(cdf, x_cols, ycol, count_col)

        out.update({'cid':cid,
                    'neighbors':list(clone_tmp.index[m]),
                    'neighbors_i':m,
                    'children':cluster_members[cid],
                    'K_neighbors':K,
                    'R_radius':R})
        res.append(out)

    res_df = pd.DataFrame(res)
    return res_df, Z


def running_neighborhood_tally(df, dist_func, dist_cols, x_cols, count_col='count', knn_neighbors=50, knn_radius=None, cluster_ind=None, ncpus=1):
    """Forms a cluster around each row of df and tallies the number of instances with/without traits
    in x_cols. The contingency table for each cluster/row of df can be used to test for enrichments of the traits
    in x_cols. The neighborhood is defined by the K closest neighbors using dist_func, or defined by a distance radius.

    Identical output to neighborhood_tally, however memory footprint will be lower for large datasets, at the cost of
    increased computation. Computation is parallelized using joblib, with memory caching optional.

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
    dist_func : function
        Function that accepts two dicts representing the two TCRs being compared,
        as well as an optional third dict that will maintain a cache of components
        of the distance that should be stored for fast, repeated access (e.g. pairwise
        distances among CDR2 loops, which are much less diverse)
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

    if count_col is None:
        df = df.assign(count=1)
        count_col = 'count'

    if cluster_ind is None:
        cluster_ind = df.index

    tally_params = dict(df=df,
                        dist_func=dist_func,
                        dist_cols=dist_cols,
                        x_cols=x_cols,
                        count_col=count_col,
                        knn_neighbors=knn_neighbors,
                        knn_radius=knn_radius)
    
    res = Parallel(n_jobs=ncpus)(delayed(_tally_one)(clonei=clonei, **tally_params) for clonei in cluster_ind)
    """
    res = []
    for clonei in cluster_ind:
        out = _tally_one(df, clonei, dist_func, dist_cols, x_cols, count_col, knn_neighbors, knn_radius)
        res.append(out)
    """
    res_df = pd.DataFrame(res)
    return res_df

def _compute_pwslice(dist_dict, ii, dist_func):
    pwvec = np.zeros(len(dist_dict))
    for i in range(len(dist_dict)):
        pwvec[i] = dist_func(dist_dict[ii], dist_dict[i])
    return pwvec

def _tally_one(df, clonei, dist_func, dist_cols, x_cols, count_col, knn_neighbors, knn_radius):
    ycol = 'cmember'
    ii = np.nonzero(df.index == clonei)[0][0]
    """Operating on list of dicts is much faster the DataFrame
    though conversion may not be wort it for huge DataFrames"""
    records = df[dist_cols].to_dict(orient='records')
    pwvec = _compute_pwslice(records, ii, dist_func)
    if not knn_neighbors is None:
        if knn_neighbors < 1:
            frac = knn_neighbors
            K = int(knn_neighbors * df.shape[0])
            # print('Using K = %d (%1.0f%% of %d)' % (K, 100*frac, n))
        else:
            K = int(knn_neighbors)
        R = np.partition(pwvec, K)[K]
    else:
        R = knn_radius
    
    y_lu = {1.:'MEM+', 0.:'MEM-'}
    y_float = (pwvec <= R).astype(float)
    y = np.array([y_lu[yy] for yy in y_float])
    K = int(np.sum(y_float))

    cdf = df.assign(**{ycol:y})[[ycol, count_col] + x_cols]
    out = _prep_counts(cdf, x_cols, ycol, count_col)

    out.update({'index':clonei,
                'neighbors':list(df.index[np.nonzero(y_float)[0]]),
                'K_neighbors':K,
                'R_radius':R})
    return out

"""import pwseqdist as pwsd
sys.path.append(opj(_git, 'hierdiff'))
from hierdiff.tests.data_generator import generate_peptide_data
import hierdiff

import scipy

def _hamming_wrapper(a, b):
    return pwsd.metrics.hamming_distance(a['seq'], b['seq'])

st = time.time()
dat, pw = generate_peptide_data()
print('Generated data and computed distances (%1.0fs)' % (time.time() - st))
st = time.time()
res = hierdiff.neighborhood_tally(dat,
                  pwmat=scipy.spatial.distance.squareform(pw),
                  x_cols=['trait1'],
                  count_col='count',
                  knn_neighbors=None, knn_radius=3)
print('Tallied neighborhoods with pre-computed distances (%1.0fs)' % (time.time() - st))

st = time.time()
rres = running_neighborhood_tally(dat, dist_func=_hamming_wrapper, dist_cols=['seq'], x_cols=['trait1'], count_col='count', knn_neighbors=None, knn_radius=3)
print('Tallied neighborhoods without pre-computed distances (%1.0fs)' % (time.time() - st))
"""