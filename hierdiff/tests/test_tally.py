"""
python -m pytest hierdiff/tests/test_tally.py
"""
import unittest
import numpy as np
import pandas as pd
import operator

from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
import scipy

import pwseqdist as pwsd
import hierdiff

def _generate_peptide_data(L=5, n=100, seed=110820):
    """Attempt to generate some random peptide data with a
    phenotype enrichment associated with a motif"""
    np.random.seed(seed)
    alphabet = 'ARNDCQEGHILKMFPSTWYVBZ'
    probs = np.random.rand(len(alphabet))
    probs = probs / np.sum(probs)

    seqs = [''.join(np.random.choice(list(alphabet), size=5, p=probs)) for i in range(n)]
    
    def _assign_trait2(seq):
        if seq[1] in 'KRQ' or seq[3] in 'KRQ':
            pr = 0.99
        elif seq[0] in 'QA':
            pr = 0.01
        else:
            pr = 0.03
        return np.random.choice([1, 0], p=[pr, 1-pr])
    
    def _assign_trait1(seq):
        d = np.sum([i for i in map(operator.__ne__, seq, seqs[0])])
        return {0:'ZERO', 1:'ONE'}[int(d <= 2)]

    def _assign_trait3(seq):
        return np.random.choice(['A', 'B', 'C'], p=[0.2, 0.4, 0.4])
    
    pw = pwsd.apply_pairwise_sq(seqs, metric=pwsd.metrics.hamming_distance)

    Z = sch.linkage(pw, method='complete')
    labels = sch.fcluster(Z, 50, criterion='maxclust')

    dat = pd.DataFrame({'seq':seqs,
                        'trait1':np.array([_assign_trait1(p) for p in seqs]),
                        'trait2':np.array([_assign_trait2(p) for p in seqs]),
                        'trait3':np.array([_assign_trait3(p) for p in seqs]),
                        'cluster':labels,
                        'count':np.random.randint(4, 10, size=n)})
    return dat, pw

class TestTally(unittest.TestCase):

    def test_hier_tally(self):
        dat, pw = _generate_peptide_data()
        res, Z = hierdiff.hcluster_tally(dat,
                          pwmat=scipy.spatial.distance.squareform(pw),
                          x_cols=['trait1'],
                          count_col='count',
                          method='complete')
        self.assertTrue(res.shape[0] == dat.shape[0] - 1)

        res2, Z = hierdiff.hcluster_tally(dat,
                          pwmat=scipy.spatial.distance.squareform(pw),
                          Z=Z,
                          x_cols=['trait1'],
                          count_col='count',
                          method='complete')

        self.assertTrue(res2.shape[0] == dat.shape[0] - 1)
        cols = ['ct_%d' % i for i in range(4)]
        mm = (res[cols]!=res2[cols]).any(axis=1)
        #print(res.loc[mm, cols])
        #print(res2.loc[mm, cols])

        self.assertTrue((res2[cols].values == res[cols].values).all())
        expected_cols = ['ct_columns', 'val_0', 'ct_0', 'val_1', 'ct_1', 'val_2',
                         'ct_2', 'val_3', 'ct_3', 'levels',
                         'X+Y+', 'X+Y-', 'X-Y+', 'X-Y-', 'X_marg', 'Y_marg', 'X|Y+',
                         'X|Y-', 'Y|X+', 'Y|X-', 'cid', 'members',
                         'members_i', 'children', 'K_neighbors', 'R_radius']
        self.assertTrue(np.all([c in res for c in expected_cols]))
    

    def test_hier_tally_2traits(self):
        dat, pw = _generate_peptide_data()
        res, Z = hierdiff.hcluster_tally(dat,
                          pwmat=scipy.spatial.distance.squareform(pw),
                          x_cols=['trait1', 'trait2'],
                          count_col='count',
                          method='complete')
        
        expected_cols = ['ct_columns', 'val_0', 'val_1', 'val_2', 'val_3',
                         'val_4', 'val_5', 'val_6', 'val_7']
                         
        self.assertTrue(np.all([c in res for c in expected_cols]))    
        self.assertTrue(res.shape[0] == dat.shape[0] - 1)

    def test_nn_tally(self):
        dat, pw = _generate_peptide_data()
        res = hierdiff.neighborhood_tally(dat,
                          pwmat=scipy.spatial.distance.squareform(pw),
                          x_cols=['trait1'],
                          count_col='count',
                          knn_neighbors=None, knn_radius=3)
        res = dat.join(res)
        self.assertTrue(res.shape[0] == dat.shape[0])

        res = hierdiff.neighborhood_tally(dat,
                          pwmat=scipy.spatial.distance.squareform(pw),
                          x_cols=['trait1'],
                          count_col='count',
                          knn_neighbors=30, knn_radius=None)
        res = dat.join(res)
        self.assertTrue(res.shape[0] == dat.shape[0])

        res = hierdiff.neighborhood_tally(dat,
                          pwmat=scipy.spatial.distance.squareform(pw),
                          x_cols=['trait1'],
                          count_col='count',
                          knn_neighbors=0.1, knn_radius=None)
        res = dat.join(res)
        self.assertTrue(res.shape[0] == dat.shape[0])

        res = hierdiff.neighborhood_tally(dat,
                          pwmat=scipy.spatial.distance.squareform(pw),
                          x_cols=['trait1'],
                          count_col='count',
                          knn_neighbors=0.1, knn_radius=None,
                          cluster_ind=np.arange(50))
        
        self.assertTrue(res.shape[0] == 50)


if __name__ == '__main__':
    unittest.main()
