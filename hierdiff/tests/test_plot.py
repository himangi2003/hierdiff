"""
python -m pytest hierdiff/tests/test_plot.py
"""
import sys
import unittest
import numpy as np
import pandas as pd

from os.path import join as opj

from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
import scipy

import palmotif
import pwseqdist as pwsd

from hierdiff import plot_hclust, hcluster_tally, plot_hclust_props, cluster_association_test

from .data_generator import generate_peptide_data

class TestHierDiff(unittest.TestCase):
    # @unittest.skip
    def test_d3_plot(self):
        np.random.seed(110820)
        pwmat = distance.pdist(np.random.rand(100, 4))
        Z = sch.linkage(pwmat, method='complete')
        html = plot_hclust(Z, title='test_d3_plot')

        with open(opj('hierdiff', 'tests', 'test.html'), 'w', encoding='utf-8') as fh:
            fh.write(html)

        self.assertTrue(True)

    # @unittest.skip
    def test_d3_plot_props(self):
        np.random.seed(110820)
        n = 1000
        pwmat = distance.pdist(np.random.randn(n, 4))
        # Z = sch.linkage(pwmat, method='complete')

        data = pd.DataFrame({'count':np.random.randint(low=1, high=20, size=n),
                             'condition':np.random.choice(['Positive', 'Negative'], size=n)})
        res, Z = hcluster_tally(data, distance.squareform(pwmat, force='matrix'),
                                    x_cols=['condition'],
                                    count_col='count',
                                    method='complete')
        # print(res.loc[res['pvalue'] < 0.5].head())

        html = plot_hclust_props(Z, title='test_d3_plot_props',
                                    res=res, alpha=None)# , alpha=0.5, alpha_col='count')

        with open(opj('hierdiff', 'tests', 'test_props.html'), 'w', encoding='utf-8') as fh:
            fh.write(html)

        self.assertTrue(True)

    def test_props2(self):
        dat, pw = generate_peptide_data(n=100)
        np.random.seed(110820)
        #pw = pw + np.random.rand(pw.shape[0])

        pw = scipy.spatial.distance.squareform(distance.pdist(np.random.randn(dat.shape[0], 5))) + pw
        
        res, Z = hcluster_tally(dat,
                          pwmat=pw,
                          x_cols=['trait1'],
                          count_col='count',
                          method='complete')
        res = cluster_association_test(res, method='fishers')

        html = plot_hclust_props(Z, title='test_props2',
                                    res=res, alpha=0.05, alpha_col='pvalue')

        with open(opj('hierdiff', 'tests', 'test_props2.html'), 'w', encoding='utf-8') as fh:
            fh.write(html)

        self.assertTrue(True)

    
    def test_props_motif(self):
        dat, pw = generate_peptide_data()
        np.random.seed(110820)
        #pw = pw + np.random.rand(pw.shape[0])

        # pw = scipy.spatial.distance.squareform(distance.pdist(np.random.randn(dat.shape[0], 5))) + pw
        pw = pwsd.apply_pairwise_rect(metric=pwsd.metrics.hamming_distance, seqs1=dat['seq'])
        
        res, Z = hcluster_tally(dat,
                          pwmat=pw,
                          x_cols=['trait1'],
                          count_col='count',
                          method='complete')
        res = cluster_association_test(res, method='fishers')


        svg = []
        for i,r in res.iterrows():
            if r['pvalue'] < 0.05:
                m = palmotif.compute_motif(dat['seq'].values[r['neighbors_i']])
                s = palmotif.svg_logo(m, return_str=True, return_html=False, svg_height='500px', svg_width='500px')
                svg.append(s)
            else:
                svg.append('')
        res = res.assign(motif=svg)

        html = plot_hclust_props(Z, title='test_props_motif', tooltip_cols=['motif'],
                                    res=res, alpha=0.05, alpha_col='pvalue')

        with open(opj('hierdiff', 'tests', 'test_props_motif.html'), 'w', encoding='utf-8') as fh:
            fh.write(html)

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
