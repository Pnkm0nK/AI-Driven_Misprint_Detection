import sklearn as skl
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
import cv2
import skimage.feature as skif
import numpy as np

class HogFeatureExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, 
                 orientations=9,
                 pixels_per_cell=(8,8),
                 cells_per_block=(3,3),
                 block_norm='L2-Hys',
                 ):
        self.orientations=orientations
        self.pixels_per_cell=pixels_per_cell
        self.cells_per_block=cells_per_block
        self.block_norm=block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feature_vectors = []

        for image in tqdm(X, desc=f"Extracting HOG features"):
            features = skif.hog(image,
                                    orientations=self.orientations,
                                    pixels_per_cell=self.pixels_per_cell,
                                    cells_per_block=self.cells_per_block,
                                    block_norm=self.block_norm)
            feature_vectors.append(features)

        return np.asarray(feature_vectors, dtype=np.float32)
            

