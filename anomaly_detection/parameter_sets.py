import sklearn.mixture as skm
from anomaly_detection.FeatureExtractors import ClipFeatureExtractor, ORBFeatureExtractor, HogFeatureExtractor, CNNFeatureExtractor
from anomaly_detection.ClipZeroShotEstimator import ClipZeroShotEstimator, WinClipZeroShotEstimator
from anomaly_detection.CosKnnAnomalyScorer import CosKnnAnomalyScorer
from anomaly_detection.AnomalyDetectors import PatchcoreEstimator
import utilities.config as cfg 

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

zero_shot_clip = {    
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 200,
    "normalizer__max_matches": 30,
    "feature_extractor": "passthrough",
    "scaler": "passthrough",
    "dim_reduction": "passthrough",
    "estimator": ClipZeroShotEstimator(),
    "estimator__batch_size": 16,
    "estimator__model_name": "PE-Core-L-14-336",
    "estimator__pretrained": "meta",
}

resnet18_cos_knn = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 200,
    "normalizer__max_matches": 30,
    "scaler": "passthrough",
    "feature_extractor": CNNFeatureExtractor(),
    "feature_extractor__batch_size": 16,
    "feature_extractor__model_name": "resnet18",
    "feature_extractor__pretrained": True,
    "dim_reduction": "passthrough",
    "estimator": CosKnnAnomalyScorer(),
    "estimator__n_neighbors": 5,
    "estimator__use_faiss": True,
    "estimator__normalize": False,
}

winclip = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 200,
    "normalizer__max_matches": 30,
    "feature_extractor": "passthrough",
    "scaler": "passthrough",
    "dim_reduction": "passthrough",
    "estimator": WinClipZeroShotEstimator(),
    "estimator__batch_size": 16,
    "estimator__threshold_percentile": 90,
}

patchcore = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.5,
    "normalizer__n_features": 200,
    "normalizer__max_matches": 30,
    "feature_extractor": "passthrough",
    "scaler": "passthrough",
    "dim_reduction": "passthrough",
    "estimator": PatchcoreEstimator(),
    "estimator__checkpoint_path": str(cfg.BASE_DIR / "anomaly_detection" / "experiments" / "patchcore_results" / "Patchcore" / "v0" / "weights" / "lightning" / "model.ckpt"),
    "estimator__backbone": "wide_resnet50_2",
    "estimator__layers": ("layer2", "layer3"),
    "estimator__pre_trained": True,
    "estimator__coreset_sampling_ratio": 0.01,
    "estimator__num_neighbors": 9,
    "estimator__threshold_percentile": 95,
    "estimator__tile_size": 224,
    "estimator__batch_size": 16,
    "estimator__num_workers": 0,
}