import sklearn.mixture as skm
from anomaly_detection.ClipFeatureExtractor import ClipFeatureExtractor
from anomaly_detection.HogFeatureExtractor import HogFeatureExtractor
from anomaly_detection.OrbFeatureExtractor import ORBFeatureExtractor
from anomaly_detection.CosKnnAnomalyScorer import CosKnnAnomalyScorer

gmm_orb_less_pca_reduction_components = {
    "normalizer__strategy": "orb",
    "normalizer__n_features": 150,
    "feature_extractor": ORBFeatureExtractor(),
    "feature_extractor__n_features": 250,
    "dim_reduction__n_components": 15,
    "dim_reduction__svd_solver": "full",
    "estimator": skm.GaussianMixture(),
    "estimator__n_components": 1,
    "estimator__covariance_type": "diag",
    "estimator__reg_covar": 1e-4,
    "estimator__n_init": 3,
    "estimator__max_iter": 300
}

gmm_orb_pca = {
    "normalizer__strategy": "orb",
    "normalizer__n_features": 150,
    "feature_extractor": ORBFeatureExtractor(),
    "feature_extractor__n_features": 250,
    "dim_reduction__n_components": 50,
    "dim_reduction__svd_solver": "full",
    "estimator": skm.GaussianMixture(),
    "estimator__n_components": 1,
    "estimator__covariance_type": "diag",
    "estimator__reg_covar": 1e-4,
    "estimator__n_init": 3,
    "estimator__max_iter": 300
}

gmm_hog_pca = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.3,
    "normalizer__n_features": 150,
    "feature_extractor": HogFeatureExtractor(),
    "feature_extractor__orientations":9,
    "feature_extractor__pixels_per_cell":(8,8),
    "feature_extractor__cells_per_block":(3,3),
    "feature_extractor__block_norm":'L2-Hys',
    "dim_reduction__n_components": 50,
    "dim_reduction__svd_solver": "full",
    "estimator": skm.GaussianMixture(),
    "estimator__n_components": 1,
    "estimator__covariance_type": "diag",
    "estimator__reg_covar": 1e-4,
    "estimator__n_init": 3,
    "estimator__max_iter": 300
}

gmm_hog_pca_improved_alignment = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.3,
    "normalizer__n_features": 200,
    "normalizer__max_matches": 30,
    "feature_extractor": HogFeatureExtractor(),
    "feature_extractor__orientations":9,
    "feature_extractor__pixels_per_cell":(8,8),
    "feature_extractor__cells_per_block":(3,3),
    "feature_extractor__block_norm":'L2-Hys',
    "dim_reduction__n_components": 30,
    "dim_reduction__svd_solver": "full",
    "estimator": skm.GaussianMixture(),
    "estimator__n_components": 1,
    "estimator__covariance_type": "diag",
    "estimator__reg_covar": 1e-4,
    "estimator__n_init": 3,
    "estimator__max_iter": 300
}


gmm_hog_dim_reduction_more_resolution = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.5,
    "normalizer__n_features": 200,
    "normalizer__max_matches": 30,
    "feature_extractor__strategy": "hog",
    "feature_extractor__orientations":9,
    "feature_extractor__pixels_per_cell":(8,8),
    "feature_extractor__cells_per_block":(3,3),
    "feature_extractor__block_norm":'L2-Hys',
    "dim_reduction__n_components": 30,
    "dim_reduction__svd_solver": "full",
    "estimator": skm.GaussianMixture(),
    "estimator__n_components": 1,
    "estimator__covariance_type": "diag",
    "estimator__reg_covar": 1e-4,
    "estimator__n_init": 3,
    "estimator__max_iter": 300
}


cos_knn_hog_pca = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.5,
    "normalizer__n_features": 200,
    "normalizer__max_matches": 30,
    "feature_extractor": HogFeatureExtractor(),
    "feature_extractor__orientations": 9,
    "feature_extractor__pixels_per_cell": (8, 8),
    "feature_extractor__cells_per_block": (3, 3),
    "feature_extractor__block_norm": "L2-Hys",
    "dim_reduction__n_components": 30,
    "dim_reduction__svd_solver": "full",
    "estimator": CosKnnAnomalyScorer(),
    "estimator__n_neighbors": 5,
    "estimator__use_faiss": True,
    "estimator__normalize": True,
}
cos_knn_clip= {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 200,
    "normalizer__max_matches": 30,
    "scaler": "passthrough",
    "feature_extractor": ClipFeatureExtractor(),
    "feature_extractor__batch_size": 16,
    "dim_reduction": "passthrough",
    "estimator": CosKnnAnomalyScorer(),
    "estimator__n_neighbors": 7,
    "estimator__use_faiss": True,
    "estimator__normalize": False,
}
cos_knn_clip_PE_Core = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 200,
    "normalizer__max_matches": 30,
    "scaler": "passthrough",
    "feature_extractor": ClipFeatureExtractor(),
    "feature_extractor__batch_size": 16,
    "feature_extractor__model_name": "PE-Core-L-14-336",
    "feature_extractor__pretrained": "meta",
    "dim_reduction": "passthrough",
    "estimator": CosKnnAnomalyScorer(),
    "estimator__n_neighbors": 5,
    "estimator__use_faiss": True,
    "estimator__normalize": False,
}