from sklearn.svm import OneClassSVM
import numpy as np

class SVMAnomalyEstimator(OneClassSVM):
    def __init__(self, kernel='rbf', gamma='scale', threshold_percentile=95, nu=0.1, max_iter=300):
        super().__init__(kernel=kernel, gamma=gamma, nu=nu, max_iter=max_iter)
        self._last_scores = None
        self._last_scores_key = None
        self.threshold_percentile = 100 -threshold_percentile
    def _cache_key(self, X):
    # create a cache key based on the id and length of data to avoid recomputing scores
        return (id(X), len(X))
    def fit(self, X, y=None):
        super().fit(X)
        scores = super().score_samples(X)
        self._threshold = np.percentile(scores, self.threshold_percentile)
        self.is_fitted_ = True
        return self
    def predict(self, X):
        key = self._cache_key(X)
        if self._last_scores_key == key and self._last_scores is not None:
            return self._last_scores
        scores = super().score_samples(X)
        predictions = (scores < self._threshold).astype(int)
        self._last_scores = predictions
        self._last_scores_key = key
        return predictions