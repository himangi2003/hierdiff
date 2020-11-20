"""
python -m pytest hierdiff/tests/test_tally.py
"""
import unittest
import numpy as np
import pandas as pd
import operator
import time

from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
import scipy

import pwseqdist as pwsd
import hierdiff

from .data_generator import generate_peptide_data


def _hamming_wrapper(a, b):
    return pwsd.metrics.hamming_distance(a['seq'], b['seq'])

class TestTally(unittest.TestCase):

    def test_hier_tally(self):
        dat, pw = generate_peptide_data()
        res, Z = hierdiff.hcluster_tally(dat,
                                          pwmat=pw,
                                          x_cols=['trait1'],
                                          count_col='count',
                                          method='complete')
        self.assertTrue(res.shape[0] == dat.shape[0] - 1)

        res2, Z = hierdiff.hcluster_tally(dat,
                                          pwmat=pw,
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
                         'X+MEM+', 'X+MEM-', 'X-MEM+', 'X-MEM-', 'X_marg', 'MEM_marg', 'X|MEM+',
                         'X|MEM-', 'MEM|X+', 'MEM|X-', 'cid', 'children', 'K_neighbors', 'R_radius']
        # print([c for c in expected_cols if not c in res.columns])
        # print(res.columns)
        self.assertTrue(np.all([c in res.columns for c in expected_cols]))

    def test_hier_tally_no_count(self):
        dat, pw = generate_peptide_data()
        res, Z = hierdiff.hcluster_tally(dat,
                                          pwmat=pw,
                                          x_cols=['trait1'],
                                          method='complete')
        self.assertTrue(res.shape[0] == dat.shape[0] - 1)
    
    def test_hier_tally_2traits(self):
        dat, pw = generate_peptide_data()
        res, Z = hierdiff.hcluster_tally(dat,
                          pwmat=pw,
                          x_cols=['trait1', 'trait2'],
                          count_col='count',
                          method='complete')
        
        expected_cols = ['ct_columns', 'val_0', 'val_1', 'val_2', 'val_3',
                         'val_4', 'val_5', 'val_6', 'val_7']
                         
        self.assertTrue(np.all([c in res for c in expected_cols]))    
        self.assertTrue(res.shape[0] == dat.shape[0] - 1)

    def test_running_nn_tally(self):
        st = time.time()
        dat, pw = generate_peptide_data()
        print('Generated data and computed distances (%1.0fs)' % (time.time() - st))
        st = time.time()
        res = hierdiff.neighborhood_tally(dat,
                          pwmat=pw,
                          x_cols=['trait1'],
                          count_col='count',
                          knn_neighbors=None, knn_radius=3)
        print('Tallied neighborhoods with pre-computed distances (%1.0fs)' % (time.time() - st))
        st = time.time()

        rres = hierdiff.running_neighborhood_tally(dat,
                          dist_func=_hamming_wrapper,
                          dist_cols=['seq'],
                          x_cols=['trait1'],
                          count_col='count',
                          knn_neighbors=None, knn_radius=3)
        print('Tallied neighborhoods without pre-computed distances (%1.0fs)' % (time.time() - st))
        self.assertTrue(res.shape[0] == rres.shape[0])
        self.assertTrue((res == rres).all().all())
    
    def test_nn_rect_tally(self):
        dat, pw = generate_peptide_data()
        res = hierdiff.neighborhood_tally(dat,
                          pwmat=pw[:10, :],
                          x_cols=['trait1'],
                          df_centroids=dat.iloc[:10],
                          count_col='count',
                          knn_neighbors=0.1, knn_radius=None)
        
        self.assertTrue(res.shape[0] == 10)

    def test_sparse_nn_rect_tally(self):
        dat, pw = generate_peptide_data()
        res = hierdiff.neighborhood_tally(dat,
                          pwmat=pw[:10, :],
                          x_cols=['trait1'],
                          df_centroids=dat.iloc[:10],
                          count_col='count',
                          knn_neighbors=0.1, knn_radius=None)
        self.assertTrue(res.shape[0] == 10)
        
        pw_sparse = pw.copy()
        pw_sparse[pw_sparse == 0] = -1
        pw_sparse = scipy.sparse.csr_matrix(pw_sparse)

        res_sparse = hierdiff.neighborhood_tally(dat,
                          pwmat=pw_sparse[:10, :],
                          x_cols=['trait1'],
                          df_centroids=dat.iloc[:10],
                          count_col='count',
                          knn_neighbors=0.1, knn_radius=None)
        self.assertTrue((res == res_sparse).all().all())
        

    def test_nn_tally(self):
        dat, pw = generate_peptide_data()
        res = hierdiff.neighborhood_tally(dat,
                          pwmat=pw,
                          x_cols=['trait1'],
                          count_col='count',
                          knn_neighbors=None, knn_radius=3)
        res = dat.join(res)
        self.assertTrue(res.shape[0] == dat.shape[0])

        res = hierdiff.neighborhood_tally(dat,
                          pwmat=pw,
                          x_cols=['trait1'],
                          count_col='count',
                          knn_neighbors=30, knn_radius=None)
        res = dat.join(res)
        self.assertTrue(res.shape[0] == dat.shape[0])

        res = hierdiff.neighborhood_tally(dat,
                          pwmat=pw,
                          x_cols=['trait1'],
                          count_col='count',
                          knn_neighbors=0.1, knn_radius=None)
        res = dat.join(res)
        self.assertTrue(res.shape[0] == dat.shape[0])


if __name__ == '__main__':
    #t = TestTally()
    #t.test_running_nn_tally()
    unittest.main()
