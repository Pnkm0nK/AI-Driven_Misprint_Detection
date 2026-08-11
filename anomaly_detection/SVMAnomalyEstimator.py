from sklearn.svm import OneClassSVM
import numpy as np

class SVMAnomalyEstimator(OneClassSVM):
    def __init__(self, kernel='rbf', degree=3, gamma='scale', threshold_percentile=95, nu=0.1, max_iter=300):
        super().__init__(kernel=kernel, degree=degree, gamma=gamma, nu=nu, max_iter=max_iter)
        self._last_scores = None
        self._last_scores_key = None
        self.threshold_percentile = threshold_percentile
    def _cache_key(self, X):
    # create a cache key based on the id and length of data to avoid recomputing scores
        return (id(X), len(X))

    def score_samples(self, X):
        key = self._cache_key(X)
        if self._last_scores_key == key and self._last_scores is not None:
            return self._last_scores
        self._last_scores = -super().score_samples(X)
        self._last_scores_key = key
        return self._last_scores

    def fit(self, X, y=None):
        super().fit(X)
        scores = self.score_samples(X)
        self._threshold = np.percentile(scores, self.threshold_percentile)
        self.is_fitted_ = True
        return self
    def predict(self, X):
        scores = super().score_samples(X)
        predictions = (scores > self._threshold).astype(int)
        return predictions