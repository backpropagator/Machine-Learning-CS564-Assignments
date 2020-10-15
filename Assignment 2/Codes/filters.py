import numpy as np
import warnings

from scipy import special, stats
from scipy.sparse import issparse

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr,
                     safe_mask)
from sklearn.utils.extmath import safe_sparse_dot, row_norms
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _deprecate_positional_args
from _base import SelectorMixin


def _clean_nans(scores):
    scores = as_float_array(scores, copy=True)
    scores[np.isnan(scores)] = np.finfo(scores.dtype).min
    return scores


def f_oneway(*args):

    n_classes = len(args)
    args = [as_float_array(a) for a in args]
    n_samples_per_class = np.array([a.shape[0] for a in args])
    n_samples = np.sum(n_samples_per_class)
    ss_alldata = sum(safe_sqr(a).sum(axis=0) for a in args)
    sums_args = [np.asarray(a.sum(axis=0)) for a in args]
    square_of_sums_alldata = sum(sums_args) ** 2
    square_of_sums_args = [s ** 2 for s in sums_args]
    sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
    ssbn = 0.
    for k, _ in enumerate(args):
        ssbn += square_of_sums_args[k] / n_samples_per_class[k]
    ssbn -= square_of_sums_alldata / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    constant_features_idx = np.where(msw == 0.)[0]
    if (np.nonzero(msb)[0].size != msb.size and constant_features_idx.size):
        warnings.warn("Features %s are constant." % constant_features_idx,
                      UserWarning)
    f = msb / msw
    # flatten matrix to vector in sparse case
    f = np.asarray(f).ravel()
    prob = special.fdtrc(dfbn, dfwn, f)
    return f, prob


def f_classif(X, y):

    X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'])
    args = [X[safe_mask(X, y == k)] for k in np.unique(y)]
    return f_oneway(*args)


def _chisquare(f_obs, f_exp):

    f_obs = np.asarray(f_obs, dtype=np.float64)

    k = len(f_obs)
    # Reuse f_obs for chi-squared statistics
    chisq = f_obs
    chisq -= f_exp
    chisq **= 2
    with np.errstate(invalid="ignore"):
        chisq /= f_exp
    chisq = chisq.sum(axis=0)
    return chisq, special.chdtrc(k - 1, chisq)


def chi2(X, y):

    X = check_array(X, accept_sparse='csr')
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")

    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    observed = safe_sparse_dot(Y.T, X)          # n_classes * n_features

    feature_count = X.sum(axis=0).reshape(1, -1)
    class_prob = Y.mean(axis=0).reshape(1, -1)
    expected = np.dot(class_prob.T, feature_count)

    return _chisquare(observed, expected)


@_deprecate_positional_args
def f_regression(X, y, *, center=True):

    X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                     dtype=np.float64)
    n_samples = X.shape[0]

    # compute centered values
    # note that E[(x - mean(x))*(y - mean(y))] = E[x*(y - mean(y))], so we
    # need not center X
    if center:
        y = y - np.mean(y)
        if issparse(X):
            X_means = X.mean(axis=0).getA1()
        else:
            X_means = X.mean(axis=0)
        # compute the scaled standard deviations via moments
        X_norms = np.sqrt(row_norms(X.T, squared=True) -
                          n_samples * X_means ** 2)
    else:
        X_norms = row_norms(X.T)

    # compute the correlation
    corr = safe_sparse_dot(y, X)
    corr /= X_norms
    corr /= np.linalg.norm(y)

    # convert to p-value
    degrees_of_freedom = y.size - (2 if center else 1)
    F = corr ** 2 / (1 - corr ** 2) * degrees_of_freedom
    pv = stats.f.sf(F, 1, degrees_of_freedom)
    return F, pv



class _BaseFilter(SelectorMixin, BaseEstimator):

    def __init__(self, score_func):
        self.score_func = score_func

    def fit(self, X, y):

        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc'],
                                   multi_output=True)

        if not callable(self.score_func):
            raise TypeError("The score function should be a callable, %s (%s) "
                            "was passed."
                            % (self.score_func, type(self.score_func)))

        self._check_params(X, y)
        score_func_ret = self.score_func(X, y)
        if isinstance(score_func_ret, (list, tuple)):
            self.scores_, self.pvalues_ = score_func_ret
            self.pvalues_ = np.asarray(self.pvalues_)
        else:
            self.scores_ = score_func_ret
            self.pvalues_ = None

        self.scores_ = np.asarray(self.scores_)

        return self

    def _check_params(self, X, y):
        pass

    def _more_tags(self):
        return {'requires_y': True}


class SelectPercentile(_BaseFilter):

    @_deprecate_positional_args
    def __init__(self, score_func=f_classif, *, percentile=10):
        super().__init__(score_func=score_func)
        self.percentile = percentile

    def _check_params(self, X, y):
        if not 0 <= self.percentile <= 100:
            raise ValueError("percentile should be >=0, <=100; got %r"
                             % self.percentile)

    def _get_support_mask(self):
        check_is_fitted(self)

        # Cater for NaNs
        if self.percentile == 100:
            return np.ones(len(self.scores_), dtype=np.bool)
        elif self.percentile == 0:
            return np.zeros(len(self.scores_), dtype=np.bool)

        scores = _clean_nans(self.scores_)
        threshold = np.percentile(scores, 100 - self.percentile)
        mask = scores > threshold
        ties = np.where(scores == threshold)[0]
        if len(ties):
            max_feats = int(len(scores) * self.percentile / 100)
            kept_ties = ties[:max_feats - mask.sum()]
            mask[kept_ties] = True
        return mask


class SelectKBest(_BaseFilter):

    @_deprecate_positional_args
    def __init__(self, score_func=f_classif, *, k=10):
        super().__init__(score_func=score_func)
        self.k = k

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError("k should be >=0, <= n_features = %d; got %r. "
                             "Use k='all' to return all features."
                             % (X.shape[1], self.k))

    def _get_support_mask(self):
        check_is_fitted(self)

        if self.k == 'all':
            return np.ones(self.scores_.shape, dtype=bool)
        elif self.k == 0:
            return np.zeros(self.scores_.shape, dtype=bool)
        else:
            scores = _clean_nans(self.scores_)
            mask = np.zeros(scores.shape, dtype=bool)

            mask[np.argsort(scores, kind="mergesort")[-self.k:]] = 1
            return mask


class SelectFpr(_BaseFilter):

    @_deprecate_positional_args
    def __init__(self, score_func=f_classif, *, alpha=5e-2):
        super().__init__(score_func=score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        check_is_fitted(self)

        return self.pvalues_ < self.alpha


class SelectFdr(_BaseFilter):

    @_deprecate_positional_args
    def __init__(self, score_func=f_classif, *, alpha=5e-2):
        super().__init__(score_func=score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        check_is_fitted(self)

        n_features = len(self.pvalues_)
        sv = np.sort(self.pvalues_)
        selected = sv[sv <= float(self.alpha) / n_features *
                      np.arange(1, n_features + 1)]
        if selected.size == 0:
            return np.zeros_like(self.pvalues_, dtype=bool)
        return self.pvalues_ <= selected.max()


class SelectFwe(_BaseFilter):

    @_deprecate_positional_args
    def __init__(self, score_func=f_classif, *, alpha=5e-2):
        super().__init__(score_func=score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        check_is_fitted(self)

        return (self.pvalues_ < self.alpha / len(self.pvalues_))



class GenericUnivariateSelect(_BaseFilter):

    _selection_modes = {'percentile': SelectPercentile,
                        'k_best': SelectKBest,
                        'fpr': SelectFpr,
                        'fdr': SelectFdr,
                        'fwe': SelectFwe}

    @_deprecate_positional_args
    def __init__(self, score_func=f_classif, *, mode='percentile', param=1e-5):
        super().__init__(score_func=score_func)
        self.mode = mode
        self.param = param

    def _make_selector(self):
        selector = self._selection_modes[self.mode](score_func=self.score_func)

        # Now perform some acrobatics to set the right named parameter in
        # the selector
        possible_params = selector._get_param_names()
        possible_params.remove('score_func')
        selector.set_params(**{possible_params[0]: self.param})

        return selector

    def _check_params(self, X, y):
        if self.mode not in self._selection_modes:
            raise ValueError("The mode passed should be one of %s, %r,"
                             " (type %s) was passed."
                             % (self._selection_modes.keys(), self.mode,
                                type(self.mode)))

        self._make_selector()._check_params(X, y)

    def _get_support_mask(self):
        check_is_fitted(self)

        selector = self._make_selector()
        selector.pvalues_ = self.pvalues_
        selector.scores_ = self.scores_
        return selector._get_support_mask()