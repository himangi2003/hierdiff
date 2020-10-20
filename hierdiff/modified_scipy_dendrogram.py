import numpy as np

__all__ = ['modified_dendrogram']

def modz(Z, counts):
    """Turns out modifying the count column of Z was not needed"""
    n = Z.shape[0] + 1
    outz = Z.copy()
    for i in range(outz.shape[0]):
        if outz[i, 0] < n:
            L = counts[int(outz[i, 0])]
        else:
            L = outz[int(outz[i, 0] - n), 3]
        if outz[i, 1] < n:
            R = counts[int(outz[i, 1])]
        else:
            R = outz[int(outz[i, 1] - n), 3]
        outz[i, 3] = L + R
        # print(f'{i}\t{int(outz[i, 0])}\t{int(outz[i, 1])}\t{int(L)}\t{int(R)}\t{L+R}')
    return outz

def modified_dendrogram(Z, counts=None, count_sort=False, distance_sort=False):
    """Modified from scipy.cluster.hierarchy so that node width can be weighted
    by observation counts"""
    Z = np.asarray(Z, order='c')
    if counts is None:
        counts = np.ones(Z.shape[0] + 1)

    Zs = Z.shape
    n = Zs[0] + 1

    icoord_list = []
    dcoord_list = []
    cid_list = []

    R = {'icoord': icoord_list, 'dcoord': dcoord_list, 'cid_list': cid_list}

    _dendrogram_calculate_info(
        Z=Z,
        counts=counts,
        count_sort=count_sort,
        distance_sort=distance_sort,
        i=2*n - 2,
        iv=0.0,
        n=n,
        icoord_list=icoord_list,
        dcoord_list=dcoord_list,
        cid_list=cid_list)

    return R


def _dendrogram_calculate_info(Z, counts,
                               count_sort=False, distance_sort=False,
                               i=-1, iv=0.0,
                               n=0, icoord_list=[], dcoord_list=[], cid_list=[]):

    # print(f'i{i}, iv={iv}\nivl={ivl}\nn={n}')
    if n == 0:
        raise ValueError("Invalid singleton cluster count n.")

    if i == -1:
        raise ValueError("Invalid root cluster index i.")

    # Only place leaves if they correspond to original observations.
    if i < n:
        # return (iv + 5.0, 10.0, 0.0, 0.0)
        return (iv + 5.0, 10.0 * counts[i], 0.0, 0.0)

    # !!! Otherwise, we don't have a leaf node, so work on plotting a
    # non-leaf node.
    # Actual indices of a and b
    aa = int(Z[i - n, 0])
    ab = int(Z[i - n, 1])
    if aa > n:
        # The number of singletons below cluster a
        na = Z[aa - n, 3]
        # The distance between a's two direct children.
        da = Z[aa - n, 2]
    else:
        na = 1
        da = 0.0
    if ab > n:
        nb = Z[ab - n, 3]
        db = Z[ab - n, 2]
    else:
        nb = 1
        db = 0.0

    if count_sort == 'ascending' or count_sort:
        # If a has a count greater than b, it and its descendents should
        # be drawn to the right. Otherwise, to the left.
        if na > nb:
            # The cluster index to draw to the left (ua) will be ab
            # and the one to draw to the right (ub) will be aa
            ua = ab
            ub = aa
        else:
            ua = aa
            ub = ab
    elif count_sort == 'descending':
        # If a has a count less than or equal to b, it and its
        # descendents should be drawn to the left. Otherwise, to
        # the right.
        if na > nb:
            ua = aa
            ub = ab
        else:
            ua = ab
            ub = aa
    elif distance_sort == 'ascending' or distance_sort:
        # If a has a distance greater than b, it and its descendents should
        # be drawn to the right. Otherwise, to the left.
        if da > db:
            ua = ab
            ub = aa
        else:
            ua = aa
            ub = ab
    elif distance_sort == 'descending':
        # If a has a distance less than or equal to b, it and its
        # descendents should be drawn to the left. Otherwise, to
        # the right.
        if da > db:
            ua = aa
            ub = ab
        else:
            ua = ab
            ub = aa
    else:
        ua = aa
        ub = ab

    # Updated iv variable and the amount of space used.
    (uiva, uwa, uah, uamd) = \
        _dendrogram_calculate_info(
            Z=Z,
            counts=counts,
            count_sort=count_sort,
            distance_sort=distance_sort,
            i=ua,
            iv=iv,
            n=n,
            icoord_list=icoord_list,
            dcoord_list=dcoord_list,
            cid_list=cid_list)

    h = Z[i - n, 2]

    (uivb, uwb, ubh, ubmd) = \
        _dendrogram_calculate_info(
            Z=Z,
            counts=counts,
            count_sort=count_sort,
            distance_sort=distance_sort,
            i=ub,
            iv=iv + uwa,
            n=n,
            icoord_list=icoord_list,
            dcoord_list=dcoord_list,
            cid_list=cid_list)

    max_dist = max(uamd, ubmd, h)

    icoord_list.append([uiva, uiva, uivb, uivb])
    dcoord_list.append([uah, h, h, ubh])
    cid_list.append(int(i))

    return (((uiva + uivb) / 2), uwa + uwb, h, max_dist)

