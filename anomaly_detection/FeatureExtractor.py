import sklearn as skl
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
import cv2
import skimage.feature as skif
import numpy as np

class FeatureExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, strategy="orb",
                 orientations=9,
                 pixels_per_cell=(8,8),
                 cells_per_block=(3,3),
                 block_norm='L2-Hys',
                 n_features=250
                 ):
        self.strategy = strategy
        self.orientations=orientations
        self.pixels_per_cell=pixels_per_cell
        self.cells_per_block=cells_per_block
        self.block_norm=block_norm
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

        for image in tqdm(X, desc=f"Extracting {self.strategy} features"):
            if self.strategy == "hog":
                features = skif.hog(image,
                                     orientations=self.orientations,
                                     pixels_per_cell=self.pixels_per_cell,
                                     cells_per_block=self.cells_per_block,
                                     block_norm=self.block_norm)
                feature_vectors.append(features)

            elif self.strategy == "orb":
                keypoints, features = orb.detectAndCompute(image, None)
                feature_vectors.append(self._orb_to_fixed_vector(features))

            else:
                raise ValueError(f"Unsupported strategy: {self.strategy}")

        return np.asarray(feature_vectors, dtype=np.float32)
            

