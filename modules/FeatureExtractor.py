import sklearn as skl
from sklearn import BaseEstimator, TransformerMixin
import cv2
import skimage.feature as skif

class FeatureExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, strategy="hog",
                 orientations=9,
                 pixels_per_cell=(8,8),
                 cells_per_block=(3,3),
                 block_norm='L2-Hys',
                 ):
        self.strategy = strategy
        self.orientations=orientations
        self.pixels_per_cell=pixels_per_cell
        self.cells_per_block=cells_per_block
        self.block_norm=block_norm

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for image in X:
            if self.strategy == "hog":
                features = skif.hog(image,
                                     orientations=self.orientations,
                                     pixels_per_cell=self.pixels_per_cell,
                                     cells_per_block=self.cells_per_block,
                                     block_norm=self.block_norm)


            elif self.strategy == "orb":
                orb = cv2.ORB_create()
                keypoints, features = orb.detectAndCompute(image, None)

