import pandas as pd
import numpy as np
import itertools
import sys
import warnings
from functools import partial

import statsmodels.api as sm
import patsy

from scipy.stats import chi2_contingency
from scipy.stats.contingency import expected_freq
from scipy import stats

import fishersapi

from .tally import _dict_to_nby2

__all__ = ['cluster_association_test']

def cluster_association_test(res, y_col='cmember', method='fishers'):
    """Use output of cluster tallies to test for enrichment of traits within a cluster.

    Use Fisher's exact test (test='fishers') to detect enrichment/association of the neighborhood
    with one variable.

    Tests the 2 x 2 table for each clone:

    +----+--------+-------+--------+
    |             |  Neighborhood  |
    |             +-------+--------+
    |             |   Y   |    N   |
    +----+----+-------+--------+
    |    |  0 (+) |   a   |    b   |
    | X  +--------+-------+--------+
    |    |  1 (-) |   c   |    d   |
    +----+--------+-------+--------+
    
    Note that the first level of an x_col (defined by sort order in the pd.DataFrame index) will
    be encoded as "+" in the output. Similarly, cluster membership indicate by a value of 1

    Use the chi-squared test (test='chi2') to detect association across multiple variables.
    Note that with sparse neighborhoods Chi-squared tests are unreliable.

    Use the Cochran-Mantel-Haenszel test (test='chm') to test stratified 2 x 2 tables:
    one X-var vs. neighborhood (Y), over several strata defined in other X variables.
    Use x_cols[0] as the primary (binary) variable and other x_cols for the categorical
    strata-defining variables. This tests the overall null that OR = 1 for x_cols[0].
    A test is also performed for homogeneity of the ORs among the strata (Breslow-Day test).

    Parameters
    ----------
    res : pd.DataFrame
        Result from one of the "tally" functions
    y_col : col in res
        Column indicating cluster membership. Almost certainly is 'cmember' if used a "tally" function
    method : str
        Method for testing: fishers, chi2, chm"""
    
    n = np.max([int(k.split('_')[1]) for k in res.columns if 'val_' in k]) + 1
    ct_cols = ['ct_%d' % i for i in range(n)]
    val_cols = ['val_%d' % i for i in range(n)]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if method == 'fishers':
            if n != 4:
                raise ValueError("Number of ct_cols must equal 4 (2x2) to use Fisher's exact test")
            out = _fisherNBR(res, ct_cols=['ct_%d' % i for i in range(4)])
            res = res.assign(**out)

        else:
            if method in ['chisq', 'chi2']:
                tmp = {'chisq':np.nan * np.zeros(res.shape[0]),
                        'pvalue':np.nan * np.zeros(res.shape[0])}
                for i in range(res.shape[0]):
                    tab = res[ct_cols].iloc[i].values.reshape((len(ct_cols) // 2, 2))
                    tmp['chisq'][i], tmp['pvalue'][i] = _chi2NBR(tab)
                res = res.assign(**tmp)                
                
            elif method in ['chm', 'cmh']:
                """Need to figure out how to efficiently refactor this test that wants the counts gby from tally"""
                tmp = []
                for i in range(res.shape[0]):
                    counts = _dict_to_nby2(res[ct_cols + val_cols + ['ct_columns', 'levels']].iloc[i].to_dict())
                    """Flip columns so that (0, 0) is cluster member (1)"""
                    counts = counts.unstack(y_col)[['MEM+', 'MEM-']]
                    tables = []
                    for i, gby in counts.groupby(level=counts.index.names[1:]):
                        if gby.shape == (2, 2):
                            tmp_tab = gby.values
                            """Flip the rows of each table so that (0, 0) is X+ (second value of X)"""
                            # tmp_tab = tmp_tab[::-1, :]
                            tables.append(tmp_tab)

                    tmp.append(_CMH_NBR(tables))
                tmp = pd.DataFrame(tmp)
                res = pd.concat((res, tmp), axis=1)              

    for c in [c for c in res.columns if 'pvalue' in c]:
        res = res.assign(**{c.replace('pvalue', 'FWERp'):fishersapi.adjustnonnan(res[c].values, method='holm'),
                                  c.replace('pvalue', 'FDRq'):fishersapi.adjustnonnan(res[c].values, method='fdr_bh')})
    return res


def _chi2NBR(tab):
    """Applies a chi2 test to a table tab encoded in every row of res_df.
    For each row, the vector of counts in ct_cols can be reshaped into
    a n x 2 table for providing to scipy.stats.chi2_contingency

    Parameters
    ----------
    tab : n x 2 table, np.ndarray

    Returns
    -------
    res : dict
        A dict of two numpy vectors containing the chisq statistic and associated
        p-value for each test. Vectors will have length ntests, same as res_df.shape[0]"""

    """Squeeze out rows where there were no instances inside or outside the cluster
    (happens with subset_ind option)"""
    both_zero_ind = np.all(tab==0, axis=1)
    tab = tab[~both_zero_ind, :]
    try:
        chisq, pvalue, dof, expected = chi2_contingency(tab)
    except ValueError:
        chisq, pvalue = np.nan, np.nan
    return chisq, pvalue

def _CMH_NBR(tables, continuity_correction=True):
    """Applies a Cochran-Mantel-Haenszel test to a set of 2 x 2 tables,
    where each table in the set has cluster membership as one variable
    and the first x_col as the other variable (requires that x_cols[0] is binary).
    Each table in the set is from a different strata defined by the categorical
    variables in x_cols[1:]. This test only applies to tests of more than one variable.

    Parameters
    ----------
    tables : list of 2x2 tables

    continuity_correction : bool
        Whether to use a continuity correct in the CMH test.

    Returns
    -------
    out : dict
        Results from the test including a pooled OR, RR and pvalues
        for testing the overall null of OR = 1 for x_cols[0] and
        null of OR_1 = OR_2 = OR_3 ... (Breslow-Day test of homogeneity)"""
    
    st = sm.stats.StratifiedTable(tables)
    out = {'equal_odds_pvalue':st.test_equal_odds().pvalue,
           'null_odds_pvalue':st.test_null_odds(correction=continuity_correction).pvalue,
           'OR_pooled':st.oddsratio_pooled,
           'RR_pooled':st.riskratio_pooled}
    return out

def _fisherNBR(res_df, ct_cols):
    """Applies a Fisher's exact test to every row of res_df using the 4 columns provided
    in count_cols. For each row, the vector of counts in count_cols can
    be reshaped into a 2 x 2 contingency table.
    
    The count_cols should be in the following order:
    
    a   X+/MEM+
    b   X+/MEM-
    c   X-/MEM+
    d   X-/MEM-

    where X+ indicates the second level of x_col (e.g. 1 for [0, 1]). The result is that
    that an OR > 1 = [(a / c) / (b / d)] indicates enrichment of X
    within the cluster.

    Relative-rate of X in vs. out of the cluster is also provided.


    Parameters
    ----------
    res_df : pd.DataFrame [ntests x 4]
        Each row contains a set of 4 counts to be tested.
    count_cols : list
        Columns containing the counts in a "flattened" order such that
        it can be reshaped into a 2 x 2 contingency table

    Returns
    -------
    res : dict
        A dict of three numpy vectors containing the OR, the RR and the p-value.
        Vectors will have length ntests, same as res_df.shape[0]"""
    a = res_df[ct_cols[0]].values
    b = res_df[ct_cols[1]].values
    c = res_df[ct_cols[2]].values
    d = res_df[ct_cols[3]].values

    OR, p = fishersapi.fishers_vec(a, b, c, d, alternative='two-sided')

    RR = (a / (a + c)) / (b / (b + d))
    return {'RR':RR, 'OR':OR, 'pvalue':p}
'''
def _glmCatNBR(df, x_cols, y_col='NBR', count_col=None, l2_alpha=0, nperms=100):
    """Applies a logisitic regression with cluster membership as the outcome and
    other variables in x_cols as predictors. The major advantage of this method
    is the ability to test for multiple associations simultaneously while adjusting
    for covariates. The major problem with this method is that if the cluster
    is small and/or covariates are sparsely populated from the data then the
    model will not properly converge or may be a "perfect seperation".
    Worse, it may appear to converge but will not produce valid
    confidence intervals. Penalized regression is one way to handle these
    circumstances, however valid inference is a challenge and its not obvious
    how best to decide on the magnitude of the penalty. Bootstrapping is one way
    to compute a valid p-value, but it is slow unless parallelized. Therefore,
    this is a work in progress.

    This and other regressions may more naturally be applied to the df of
    observations as opposed to the tallied counts.

    Parameters
    ----------
    df : pd.DataFrame [ncells x ycols and xcols]
        A raw data matrix to be used in logistic regression.
    x_cols : list
        Predictors in the regression model.
    y_col : str
        Column in df to be used as the regression outcome,
        typically NBR representing cluster membership [1, 0]
    count_col : str
        Optionally provide weights for each covariate/outcome combination
        instead of having one observation per row.
    l2_alpha : float
        Magnitude of the L2-penalty
    nperms : int
        Number of permutations for the permutation test that is required by
        penalized regression to get a p-value

    Returns
    -------
    out : dict
        A dict of numpy vectors, one value per parameter providing: OR, coef, pvalue"""
    if count_col is None:
        freq_weights = None
    else:
        freq_weights = df[count_col]

    formula = ' + '.join(['C(%s)' % c for c in x_cols])
    X = patsy.dmatrix(formula, df, return_type='dataframe')
    glmParams = dict(exog=X,
                     family=sm.families.Binomial(link=sm.families.links.logit),
                     freq_weights=freq_weights,
                     hasconst=True)
    mod = sm.GLM(endog=df[y_col].values, **glmParams)
    if l2_alpha == 0:
        res = mod.fit()
        out = {'%s_pvalue' % c:res.pvalues[c] for c in X.columns if not 'Intercept' in c}
    else:
        res = mod.fit_regularized(L1_wt=0, alpha=l2_alpha)
        rparams = np.zeros((len(res.params), nperms))
        for permi in range(nperms):
            randy = df[y_col].sample(frac=1, replace=False).values
            rres = sm.GLM(endog=randy, **glmParams).fit_regularized(L1_wt=0, alpha=l2_alpha)
            rparams[:, permi] = rres.params

        perm_values = ((np.abs(res.params[:, None]) < np.abs(rparams)).sum(axis=1) + 1) / (nperms + 1)
        out = {'%s_pvalue' % c:v for c,v in zip(X.columns, perm_values) if not 'Intercept' in c}

    out.update({'%s_coef' % c:res.params[c] for c in X.columns if not 'Intercept' in c})
    out.update({'%s_OR' % c:np.exp(res.params[c]) for c in X.columns if not 'Intercept' in c})
    return out
'''