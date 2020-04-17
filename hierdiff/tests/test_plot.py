"""
python -m unittest hierdiff/tests/test_plot.py
"""
import sys
import unittest
import numpy as np
import pandas as pd

from os.path import join as opj

from scipy.spatial import distance
import scipy.cluster.hierarchy as sch

# from fg_shared import _git
# output_folder = opj(_git, 'hierdiff', 'tests')
# template_fn = opj(_git, 'hierdiff', 'tree_template.html')

from hierdiff import plot_hclust, hcluster_diff

class TestHierDiff(unittest.TestCase):
    
    def test_d3_plot(self):
        np.random.seed(110820)
        pwmat = distance.pdist(np.random.rand(1000, 4))
        Z = sch.linkage(pwmat, method='complete')
        html = plot_hclust(Z, height=600, width=900)

        with open(opj('hierdiff', 'tests', 'test.html'), 'w', encoding='utf-8') as fh:
            fh.write(html)

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
