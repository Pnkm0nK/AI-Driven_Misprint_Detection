import numpy as np
import importlib
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors


class CosKnnAnomalyScorer(BaseEstimator):
    """kNN-based anomaly scorer using cosine distance.

    Higher values indicate more normal samples (compatible with score_samples thresholding
    where low-score samples are treated as anomalies).
    """

    def __init__(self, n_neighbors=5, use_faiss=False, normalize=True):
        self.n_neighbors = n_neighbors
        self.use_faiss = use_faiss
        self.normalize = normalize

    def _l2_normalize(self, X):
        X = np.asarray(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return X / norms

    def _prepare_vectors(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.normalize:
            X = self._l2_normalize(X)
        return X

    def fit(self, X, y=None):
        X = self._prepare_vectors(X)
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
                self._faiss = faiss
            except Exception:
                self._faiss_index = None

        if self._faiss_index is None:
            self._nn = NearestNeighbors(n_neighbors=self._effective_k, metric="cosine")
            self._nn.fit(X)

        self.is_fitted_ = True

        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "is_fitted_") and self.is_fitted_

    def _faiss_mean_distances(self, Xn):
        sims, _ = self._faiss_index.search(Xn, self._effective_k)
        dists = 1.0 - sims
        return dists.mean(axis=1)

    def _sklearn_mean_distances(self, Xn):
        distances, _ = self._nn.kneighbors(Xn, n_neighbors=self._effective_k, return_distance=True)
        return distances.mean(axis=1)

    def score_samples(self, X):
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("CosKnnAnomalyScorer must be fitted before calling score_samples.")

        Xn = self._prepare_vectors(X)
        if Xn.ndim != 2:
            raise ValueError("CosKnnAnomalyScorer.score_samples expects a 2D array.")

        if self._faiss_index is not None:
            mean_dist = self._faiss_mean_distances(Xn)
        else:
            mean_dist = self._sklearn_mean_distances(Xn)

        # Convert distance to normality score: lower distance -> higher score.
        return -mean_dist
