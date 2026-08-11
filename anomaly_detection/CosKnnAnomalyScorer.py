import numpy as np
import importlib
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors


class CosKnnAnomalyScorer(BaseEstimator):
    """kNN-based anomaly scorer using cosine distance.
    
    Higher scores indicate more anomalous samples (Standard high-is-anomaly convention).
    """
    def __init__(self, n_neighbors=5, threshold_percentile=97, use_faiss=True):
        self.n_neighbors = n_neighbors
        self.threshold_percentile = threshold_percentile
        self.use_faiss = use_faiss
        self._threshold = None
        self._last_key = None
        self._last_scores = None
        self.is_fitted_ = False

    def _cache_key(self, X):
        return (id(X), len(X))

    def _l2_normalize(self, X):
        """Strictly enforces L2 normalization to protect angular cosine math."""
        X = np.asarray(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return X / norms

    def fit(self, X, y=None):
        X = self._l2_normalize(X)
        
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("CosKnnAnomalyScorer.fit expects a non-empty 2D array.")

        self.train_embeddings_ = X
        self.n_features_in_ = X.shape[1]
        self._effective_k = min(self.n_neighbors, X.shape[0])
        self._faiss_index = None
        self._nn = None

        if self.use_faiss:
            try:
                faiss = importlib.import_module("faiss")
                index = faiss.IndexFlatIP(X.shape[1])
                index.add(X)
                self._faiss_index = index
            except Exception:
                self._faiss_index = None

        if self._faiss_index is None:
            self._nn = NearestNeighbors(n_neighbors=self._effective_k, metric="cosine")
            self._nn.fit(X)
            
        self.is_fitted_ = True
        
        self._last_key = None
        self._last_scores = None
        
        train_distances = self.score_samples(X)
        
        self._threshold = np.percentile(train_distances, self.threshold_percentile)
        return self

    def __sklearn_is_fitted__(self):
        return self.is_fitted_

    def _faiss_mean_distances(self, Xn):
        sims, _ = self._faiss_index.search(Xn, self._effective_k)
        dists = 1.0 - sims
        return dists.mean(axis=1)

    def _sklearn_mean_distances(self, Xn):
        distances, _ = self._nn.kneighbors(Xn, n_neighbors=self._effective_k, return_distance=True)
        return distances.mean(axis=1)

    def score_samples(self, X):
        if not self.is_fitted_:
            raise RuntimeError("CosKnnAnomalyScorer must be fitted before calling score_samples.")
        
        key = self._cache_key(X)
        if self._last_key == key and self._last_scores is not None:
            return self._last_scores
            
        Xn = self._l2_normalize(X)
        
        if self._faiss_index is not None:
            mean_dist = self._faiss_mean_distances(Xn)
        else:
            mean_dist = self._sklearn_mean_distances(Xn)
            
        self._last_key = key
        self._last_scores = mean_dist
        return mean_dist
    
    def predict(self, X):
        scores = self.score_samples(X)
        return (scores > self._threshold).astype(int)
