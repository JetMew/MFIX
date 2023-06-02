import numpy as np
from scipy import stats


def bilog_transform(X: np.ndarray):
    """Magnify the difference between X and 0"""
    X = X.copy()
    idx = (X >= 0)
    X[idx] = np.log(1 + X[idx])
    X[~idx] = -np.log(1 - X[~idx])
    return X


def gaussian_transform(X: np.ndarray):
    """
    Transform data into Gaussian by applying psi = Phi^{-1} o F where F is the truncated ECDF.
    References:
    [1] Andrew Gordon Wilson and Zoubin Ghahramani. Copula processes.
        In Proceedings of the 23rd International Conference on Neural Information Processing
        Systems - Volume 2, NIPS’10, pages 2460–2468, USA, 2010. Curran Associates Inc.
    [2] Salinas, D.; Shen, H.; and Perrone, V. 2020.
        A Quantile-based Approach for Hyperparameter Transfer Learning.
        In International conference on machine learning, 7706–7716.
    """
    if X.ndim == 2:
        z = np.hstack([
            gaussian_transform(x.reshape(-1)).reshape(-1, 1)
            for x in np.hsplit(X, X.shape[1])
        ])
        return z
    assert X.ndim == 1

    def winsorized_delta(n):
        return 1.0 / (4.0 * n ** 0.25 * np.sqrt(np.pi * np.log(n)))

    def truncated_quantile(X):
        idx = np.argsort(X)
        rank = np.argsort(idx)
        quantile = rank / (X.shape[0] - 1)
        delta = winsorized_delta(X.shape[0])
        return np.clip(quantile, a_min=delta, a_max=1 - delta)

    return stats.norm.ppf(truncated_quantile(X))


_func_dict = {
    'bilog': bilog_transform,
    'gaussian': gaussian_transform,
    None: lambda x: x,
}


def get_transform_function(transform: str):
    if transform in _func_dict.keys():
        return _func_dict[transform]
    else:
        raise ValueError('Invalid transform: %s' % (transform, ))