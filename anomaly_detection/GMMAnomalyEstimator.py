import numpy as np

class GMMAnomalyEstimator:
    def __init__(self, n_components=1, threshold_percentile=10, covariance_type='full', random_state=22):
        self.n_components = n_components
        self.percentile = threshold_percentile
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self._last_scores_key = None
        self._last_scores = None
        self._threshold = None
    
    def _cache_key(self, X):
        return (id(X), len(X))
    
    def _get_gmm_scores(self, X):
        key = self._cache_key(X)
        if self._last_scores_key == key and self._last_scores is not None:
            train_scores = self._last_scores
        else:
            train_scores = 1 - self.model.score_samples(X)
            self._last_scores_key = key
            self._last_scores = train_scores
        return train_scores

    def fit(self, X, y=None):
        from sklearn.mixture import GaussianMixture
        self.model = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type, random_state=self.random_state)
        self.model.fit(X)
        train_scores = self._get_gmm_scores(X)
        self._threshold = np.percentile(train_scores, self.percentile)
        self.is_fitted_ = True
        return self

    def score_samples(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self._get_gmm_scores(X)
    
    def predict(self, X):
        scores = self.score_samples(X)
        return (scores > self._threshold).astype(int)
        