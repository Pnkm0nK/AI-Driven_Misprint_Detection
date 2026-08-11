import numpy as np
from sklearn.mixture import GaussianMixture

class GMMAnomalyEstimator(GaussianMixture):
    def __init__(self, n_components=1, threshold_percentile=95,
                  covariance_type='diag', reg_covar=1e-06, n_init=3, max_iter=100, random_state=20):
        super().__init__(n_components=n_components, covariance_type=covariance_type,
                         reg_covar=reg_covar, n_init=n_init, max_iter=max_iter, random_state=random_state)
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._last_scores_key = None
        self._last_scores = None
        self._threshold = None
    
    def _cache_key(self, X):
        return (id(X), len(X))
    
    def _get_gmm_scores(self, X):
        '''helper function for gmm scoring with caching'''
        key = self._cache_key(X)
        if self._last_scores_key == key and self._last_scores is not None:
            train_scores = self._last_scores
        else:
            # GMM's score_samples returns log-likelihood, we want negative log-likelihood for anomaly scoring
            # higher scores indicate more anomalous samples
            train_scores = -super().score_samples(X)
            self._last_scores_key = key
            self._last_scores = train_scores
        return train_scores

    def fit(self, X, y=None):
        super().fit(X)
        self.train_scores = self._get_gmm_scores(X)
        self._threshold = np.percentile(self.train_scores, self.threshold_percentile)
        self.is_fitted_ = True
        return self

    def score_samples(self, X):
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet.")
        return self._get_gmm_scores(X)
    
    def predict(self, X):
        scores = self.score_samples(X)
        return (scores > self._threshold).astype(int)
        