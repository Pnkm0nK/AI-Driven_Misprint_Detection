import sklearn as skl
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
import cv2
import skimage.feature as skif
import numpy as np

class ORBFeatureExtractor(TransformerMixin, BaseEstimator):
    def __init__(self,
                 n_features=250
                 ):
        self.n_features = n_features

    def fit(self, X, y=None):
        return self

    def _orb_to_fixed_vector(self, descriptors):
        descriptor_len = 32
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_features * descriptor_len, dtype=np.float32)

        descriptors = descriptors[:self.n_features]
        if descriptors.shape[0] < self.n_features:
            pad_rows = self.n_features - descriptors.shape[0]
            padding = np.zeros((pad_rows, descriptor_len), dtype=descriptors.dtype)
            descriptors = np.vstack((descriptors, padding))

        return descriptors.reshape(-1).astype(np.float32)
    
    def transform(self, X):
        feature_vectors = []
        orb = cv2.ORB_create(nfeatures=self.n_features) if self.strategy == "orb" else None

        for image in tqdm(X, desc=f"Extracting ORB features"):
            keypoints, features = orb.detectAndCompute(image, None)
            feature_vectors.append(self._orb_to_fixed_vector(features))

        return np.asarray(feature_vectors, dtype=np.float32)
            