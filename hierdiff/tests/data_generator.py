import numpy as np
import pandas as pd
import operator

from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
import scipy

import pwseqdist as pwsd

def generate_peptide_data(L=5, n=300, seed=110820):
    """Attempt to generate some random peptide data with a
    phenotype enrichment associated with a motif"""
    np.random.seed(seed)
    alphabet = 'ARNDCQEGHILKMFPSTWYVBZ'
    probs = np.random.rand(len(alphabet))
    probs = probs / np.sum(probs)

    seqs = [''.join(np.random.choice(list(alphabet), size=4, p=probs)) for i in range(n)]
    
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
        return {0:'ZERO', 1:'ONE'}[int((d <= 3) * (np.random.rand() < 0.6))]

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