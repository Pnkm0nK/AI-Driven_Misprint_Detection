import sklearn.mixture as skm
from anomaly_detection.SVMAnomalyEstimator import SVMAnomalyEstimator
from anomaly_detection.GMMAnomalyEstimator import GMMAnomalyEstimator
from sklearn.preprocessing import StandardScaler, Normalizer
from anomaly_detection.FeatureExtractors import ClipFeatureExtractor, ORBVladGridFeatureExtractor, HogFeatureExtractor, DNNFeatureExtractor, DenseBRIEFFeatureExtractor
from anomaly_detection.ClipZeroShotEstimator import ClipZeroShotEstimator, WinClipZeroShotEstimator
from anomaly_detection.CosKnnAnomalyScorer import CosKnnAnomalyScorer
from anomaly_detection.AnomalyDetectors import PatchcoreEstimator, DinomalyEstimator
from anomaly_detection.AnomalyDino import AnomalyDinoEstimator
import utilities.config as cfg 

gmm_orb_pca = {
    "normalizer__strategy": "orb",
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
    "feature_extractor": ORBVladGridFeatureExtractor(),
    "feature_extractor__n_features": 250,
    "dim_reduction__n_components": 50,
    "dim_reduction__svd_solver": "full",
    "estimator": GMMAnomalyEstimator(),
    "estimator__n_components": 1,
    "estimator__covariance_type": "diag",
    "estimator__reg_covar": 1e-4,
    "estimator__n_init": 3,
    "estimator__max_iter": 300
}

gmm_orb_pca_vlad= {
    "normalizer__strategy": "orb",
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
    "feature_extractor": ORBVladGridFeatureExtractor(),
    "feature_extractor__n_features": 450,
    "feature_extractor__n_visual_words": 32,
    "feature_extractor__cells_per_image": (3,3),
    "scaler": "passthrough",
    "dim_reduction__n_components": 70,
    "dim_reduction__svd_solver": "full",
    "estimator": GMMAnomalyEstimator(), 
    "estimator__n_components": 1,
    "estimator__covariance_type": "diag",
    "estimator__reg_covar": 1e-4,
    "estimator__n_init": 3,
    "estimator__max_iter": 300
}

gmm_dense_brief_pca = {
    "normalizer__strategy": "orb",
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
    "feature_extractor": DenseBRIEFFeatureExtractor(),
    "feature_extractor__n_points": 500,
    "dim_reduction__n_components": 70,
    "dim_reduction__svd_solver": "full",
    "scaler": Normalizer(),
    "scaler__norm": "l2",
    "estimator": GMMAnomalyEstimator(),
    "estimator__n_components": 1,
    "estimator__covariance_type": "diag",
    "estimator__reg_covar": 1e-4,
    "estimator__n_init": 3,
    "estimator__max_iter": 300
}

svm_rbf_dense_brief_pca = {
    "normalizer__strategy": "orb",
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
    "feature_extractor": DenseBRIEFFeatureExtractor(),
    "feature_extractor__n_points": 500,
    "dim_reduction__n_components": 70,
    "dim_reduction__svd_solver": "full",
    "scaler": Normalizer(),
    "scaler__norm": "l2",
    "estimator": SVMAnomalyEstimator(),
    "estimator__threshold_percentile": 98,
}


svm_rbf_hog_pca = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.3,
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
    "feature_extractor": HogFeatureExtractor(),
    "feature_extractor__orientations":9,
    "feature_extractor__pixels_per_cell":(8,8),
    "feature_extractor__cells_per_block":(3,3),
    "feature_extractor__block_norm":'L2-Hys',
    "dim_reduction__n_components": 50,
    "dim_reduction__svd_solver": "full",
    "estimator": SVMAnomalyEstimator(threshold_percentile=95),
}

gmm_hog_pca = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.3,
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
    "feature_extractor": HogFeatureExtractor(),
    "feature_extractor__orientations":9,
    "feature_extractor__pixels_per_cell":(8,8),
    "feature_extractor__cells_per_block":(3,3),
    "feature_extractor__block_norm":'L2-Hys',
    "dim_reduction__n_components": 50,
    "dim_reduction__svd_solver": "full",
    "estimator": GMMAnomalyEstimator(), 
    "estimator__n_components": 1,
    "estimator__covariance_type": "diag",
    "estimator__reg_covar": 1e-4,
    "estimator__n_init": 3,
    "estimator__max_iter": 300
}

gmm_hog_pca_improved_alignment = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.3,
    "normalizer__n_features": 250,
    "normalizer__max_matches": 30,
    "feature_extractor": HogFeatureExtractor(),
    "feature_extractor__orientations":9,
    "feature_extractor__pixels_per_cell":(8,8),
    "feature_extractor__cells_per_block":(3,3),
    "feature_extractor__block_norm":'L2-Hys',
    "dim_reduction__n_components": 30,
    "dim_reduction__svd_solver": "full",
    "estimator": GMMAnomalyEstimator(),
    "estimator__n_components": 1,
    "estimator__covariance_type": "diag",
    "estimator__reg_covar": 1e-4,
    "estimator__n_init": 3,
    "estimator__max_iter": 300
}


gmm_hog_dim_reduction_more_resolution = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.5,
    "normalizer__n_features": 250,
    "normalizer__max_matches": 30,
    "feature_extractor__strategy": "hog",
    "feature_extractor__orientations":9,
    "feature_extractor__pixels_per_cell":(8,8),
    "feature_extractor__cells_per_block":(3,3),
    "feature_extractor__block_norm":'L2-Hys',
    "dim_reduction__n_components": 30,
    "dim_reduction__svd_solver": "full",
    "estimator": GMMAnomalyEstimator(),
    "estimator__n_components": 1,
    "estimator__covariance_type": "diag",
    "estimator__reg_covar": 1e-4,
    "estimator__n_init": 3,
    "estimator__max_iter": 300
}


cos_knn_hog_pca = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.5,
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
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
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
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
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
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
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
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
    "normalizer__downsample_factor": 1,
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
    "scaler": "passthrough",
    "feature_extractor": DNNFeatureExtractor(),
    "feature_extractor__batch_size": 16,
    "feature_extractor__model_name": "resnet18",
    "feature_extractor__pretrained": True,
    "dim_reduction__n_components": 64,
    "dim_reduction__svd_solver": "full",
    "estimator": CosKnnAnomalyScorer(),
    "estimator__n_neighbors": 5,
    "estimator__threshold_percentile": 97,
    "estimator__use_faiss": True,
    "estimator__normalize": False,
}
resnet18_cos_knn_layer2 = resnet18_cos_knn.copy()
resnet18_cos_knn_layer2["feature_extractor__layer"] = 1
resnet18_cos_knn_layer3 = resnet18_cos_knn.copy()
resnet18_cos_knn_layer3["feature_extractor__layer"] = 2



resnet18d_cos_knn = resnet18_cos_knn.copy()
resnet18d_cos_knn["feature_extractor__model_name"] = "resnet18d.ra2_in1k"

resnet50_fb_swsl_ig1b_cos_knn = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.3,
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
    "scaler": "passthrough",
    "feature_extractor": DNNFeatureExtractor(),
    "feature_extractor__batch_size": 16,
    "feature_extractor__model_name": "resnet50.fb_swsl_ig1b_ft_in1k",
    "feature_extractor__pretrained": True,
    "dim_reduction__n_components": 64,
    "dim_reduction__svd_solver": "full",
    "estimator": CosKnnAnomalyScorer(),
    "estimator__n_neighbors": 5,
    "estimator__threshold_percentile": 97,
    "estimator__use_faiss": True,
    "estimator__normalize": False,
}
resnet50_layer4_pooling3x3_cos_knn = resnet50_fb_swsl_ig1b_cos_knn.copy()
resnet50_layer4_pooling3x3_cos_knn["feature_extractor__pool_out_size"] = (3,3)
resnet50_layer4_pooling3x3_cos_knn["feature_extractor__layer"] = 4
    

winclip = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
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
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
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
patchcore2 = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
    "feature_extractor": "passthrough",
    "scaler": "passthrough",
    "dim_reduction": "passthrough",
    "estimator": PatchcoreEstimator(),
    "estimator__backbone": "wide_resnet50_2",
    "estimator__layers": ("layer2", "layer3"),
    "estimator__pre_trained": True,
    "estimator__coreset_sampling_ratio": 0.01,
    "estimator__num_neighbors": 9,
    "estimator__threshold_percentile": 95,
    "estimator__tile_size": 128,
    "estimator__batch_size": 16,
    "estimator__num_workers": 0,
}

dinomaly = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.5,
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
    "feature_extractor": "passthrough",
    "scaler": "passthrough",
    "dim_reduction": "passthrough",
    "estimator": DinomalyEstimator(),
    "estimator__default_root_dir": str(cfg.BASE_DIR / "anomaly_detection" / "experiments"),
    "estimator__encoder_name": "dinov2reg_vit_base_14",
    "estimator__bottleneck_dropout": 0.2,
    "estimator__decoder_depth": 8,
    "estimator__remove_class_token": False,
    "estimator__threshold_percentile": 95,
    "estimator__batch_size": 8,
    "estimator__num_workers": 0,
    "estimator__random_state": 20,
    "estimator__batch_size": 16,
    "estimator__num_workers": 0
}

patchcore_dino = patchcore.copy()
patchcore_dino["estimator__encoder_name"] = "vit_base_patch14_dinov2.lvd142m"
patchcore_dino["estimator__layers"] = ["blocks.8", "blocks.11"]


dinomaly_no_cls_token = dinomaly.copy()
dinomaly_no_cls_token["estimator__remove_class_token"] = True

anomalydino = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 250,
    "normalizer__max_matches": 50,
    "feature_extractor": "passthrough",
    "scaler": "passthrough",
    "dim_reduction": "passthrough",
    "estimator": AnomalyDinoEstimator(),
    "estimator__threshold_percentile": 97,
}


parameter_set_collection = {
    "gmm_orb_pca": gmm_orb_pca,
    "gmm_orb_pca_vlad": gmm_orb_pca_vlad,
    "gmm_dense_brief_pca": gmm_dense_brief_pca,
    "svm_rbf_dense_brief_pca": svm_rbf_dense_brief_pca,
    "svm_rbf_hog_pca": svm_rbf_hog_pca,
    "gmm_hog_pca": gmm_hog_pca,
    "gmm_hog_pca_improved_alignment": gmm_hog_pca_improved_alignment,
    "gmm_hog_dim_reduction_more_resolution": gmm_hog_dim_reduction_more_resolution,
    "cos_knn_clip": cos_knn_clip,
    "cos_knn_hog_pca": cos_knn_hog_pca,
    "cos_knn_clip_PE_Core": cos_knn_clip_PE_Core,
    "zero_shot_clip": zero_shot_clip,
    "resnet18_cos_knn": resnet18_cos_knn,
    "resnet18_cos_knn_layer2": resnet18_cos_knn_layer2,
    "resnet18_cos_knn_layer3": resnet18_cos_knn_layer3,
    "resnet18d_cos_knn": resnet18d_cos_knn,
    "resnet50_fb_swsl_ig1b_cos_knn": resnet50_fb_swsl_ig1b_cos_knn,
    "resnet50_layer4_pooling3x3_cos_knn": resnet50_layer4_pooling3x3_cos_knn,
    "winclip": winclip,
    "patchcore": patchcore,
    "patchcore2": patchcore2,
    "dinomaly": dinomaly
}