import numpy as np
from scipy.sparse import issparse
from scipy.special import digamma

from sklearn.metrics.cluster import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from sklearn.utils.fixes import _astype_copy_false
from sklearn.utils.validation import check_array, check_X_y
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.multiclass import check_classification_targets




def _compute_mi_cd(c, d, n_neighbors):

    n_samples = c.shape[0]
    c = c.reshape((-1, 1))

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors()
    for label in np.unique(d):
        mask = d == label
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(c[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count

    # Ignore points with unique labels.
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]

    nn.set_params(algorithm='kd_tree')
    nn.fit(c)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    m_all = np.array([i.size for i in ind])

    mi = (digamma(n_samples) + np.mean(digamma(k_all)) -
          np.mean(digamma(label_counts)) -
          np.mean(digamma(m_all + 1)))

    return max(0, mi)


def _compute_mi(x, y, x_discrete, y_discrete, n_neighbors=3):

    if x_discrete and y_discrete:
        return mutual_info_score(x, y)
    elif x_discrete and not y_discrete:
        return _compute_mi_cd(y, x, n_neighbors)
    elif not x_discrete and y_discrete:
        return _compute_mi_cd(x, y, n_neighbors)
    else:
        return _compute_mi_cc(x, y, n_neighbors)


def _iterate_columns(X, columns=None):

    if columns is None:
        columns = range(X.shape[1])

    if issparse(X):
        for i in columns:
            x = np.zeros(X.shape[0])
            start_ptr, end_ptr = X.indptr[i], X.indptr[i + 1]
            x[X.indices[start_ptr:end_ptr]] = X.data[start_ptr:end_ptr]
            yield x
    else:
        for i in columns:
            yield X[:, i]


def _estimate_mi(X, y, discrete_features='auto', discrete_target=False,
                 n_neighbors=3, copy=True, random_state=None):

    X, y = check_X_y(X, y, accept_sparse='csc', y_numeric=not discrete_target)
    n_samples, n_features = X.shape

    if isinstance(discrete_features, (str, bool)):
        if isinstance(discrete_features, str):
            if discrete_features == 'auto':
                discrete_features = issparse(X)
            else:
                raise ValueError("Invalid string value for discrete_features.")
        discrete_mask = np.empty(n_features, dtype=bool)
        discrete_mask.fill(discrete_features)
    else:
        discrete_features = check_array(discrete_features, ensure_2d=False)
        if discrete_features.dtype != 'bool':
            discrete_mask = np.zeros(n_features, dtype=bool)
            discrete_mask[discrete_features] = True
        else:
            discrete_mask = discrete_features

    continuous_mask = ~discrete_mask
    if np.any(continuous_mask) and issparse(X):
        raise ValueError("Sparse matrix `X` can't have continuous features.")

    rng = check_random_state(random_state)
    if np.any(continuous_mask):
        if copy:
            X = X.copy()

        if not discrete_target:
            X[:, continuous_mask] = scale(X[:, continuous_mask],
                                          with_mean=False, copy=False)

        # Add small noise to continuous features as advised in Kraskov et. al.
        X = X.astype(float, **_astype_copy_false(X))
        means = np.maximum(1, np.mean(np.abs(X[:, continuous_mask]), axis=0))
        X[:, continuous_mask] += 1e-10 * means * rng.randn(
                n_samples, np.sum(continuous_mask))

    if not discrete_target:
        y = scale(y, with_mean=False)
        y += 1e-10 * np.maximum(1, np.mean(np.abs(y))) * rng.randn(n_samples)

    mi = [_compute_mi(x, y, discrete_feature, discrete_target, n_neighbors) for
          x, discrete_feature in zip(_iterate_columns(X), discrete_mask)]

    return np.array(mi)




@_deprecate_positional_args
def mutual_info_classif(X, y, *, discrete_features='auto', n_neighbors=3,
                        copy=True, random_state=None):

    check_classification_targets(y)
    return _estimate_mi(X, y, discrete_features, True, n_neighbors,
                        copy, random_state)