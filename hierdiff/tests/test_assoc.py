"""
python -m pytest hierdiff/tests/test_assoc.py
"""
import unittest
import numpy as np
import pandas as pd

import scipy.cluster.hierarchy as sch
import scipy

import hierdiff
from hierdiff import cluster_association_test, neighborhood_tally, hcluster_tally

from .data_generator import generate_peptide_data

class TestAssoc(unittest.TestCase):

    def test_hier_fisher(self):
        dat, pw = generate_peptide_data()
        res, Z = hcluster_tally(dat,
                          pwmat=pw,
                          x_cols=['trait1'],
                          count_col='count',
                          method='complete')
        res = cluster_association_test(res, method='fishers')

    def test_hier_chi2(self):
        dat, pw = generate_peptide_data()
        res, Z = hcluster_tally(dat,
                          pwmat=pw,
                          x_cols=['trait1'],
                          count_col='count',
                          method='complete')
        res = cluster_association_test(res, method='chi2')

    def test_hier_chi2_2vars(self):
        dat, pw = generate_peptide_data()
        res, Z = hcluster_tally(dat,
                          pwmat=pw,
                          x_cols=['trait1', 'trait3'],
                          count_col='count',
                          method='complete')
        res = cluster_association_test(res, method='chi2')

    def test_hier_chm(self):
        dat, pw = generate_peptide_data()
        res, Z = hcluster_tally(dat,
                          pwmat=pw,
                          x_cols=['trait1', 'trait2'],
                          count_col='count',
                          method='complete')
        res = cluster_association_test(res, method='chm')

    def test_nn_fishers(self):
        dat, pw = generate_peptide_data()
        res = neighborhood_tally(dat,
                          pwmat=pw,
                          x_cols=['trait1'],
                          count_col='count',
                          knn_neighbors=None, knn_radius=3)
        res = dat.join(res)
        res = cluster_association_test(res, method='fishers')

if __name__ == '__main__':
    unittest.main()
