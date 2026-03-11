import sklearn.mixture as skm
import skimage.feature as skif


def get_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
    """
    Extract HOG features from an image.

    Parameters:
    - image: Input image (grayscale).
    - orientations: Number of orientation bins.
    - pixels_per_cell: Size of a cell in pixels (height, width).
    - cells_per_block: Number of cells in each block (height, width).

    Returns:
    - hog_features: HOG feature vector.
    """
    hog_features = skif.hog(image,
                           orientations=orientations,
                           pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block,
                           block_norm='L2-Hys',
                           transform_sqrt=True)
    return hog_features

def fit_gmm(features, n_components=1):
    gmm = skm.GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(features)
    return gmm

if __name__ == "__main__":

    hog_features = get_hog_features(image)
    gmm_model = fit_gmm(hog_features.reshape(-1, 1), n_components=1)

