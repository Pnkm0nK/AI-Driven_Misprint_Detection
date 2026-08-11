import sklearn.mixture as skm
from anomaly_detection.SVMAnomalyEstimator import SVMAnomalyEstimator
from anomaly_detection.GMMAnomalyEstimator import GMMAnomalyEstimator
from sklearn.preprocessing import StandardScaler, Normalizer
from anomaly_detection.FeatureExtractors import ClipFeatureExtractor,  HogFeatureExtractor, DNNFeatureExtractor, LBPFeatureExtractor
import sklearn as skl
from anomaly_detection.ClipZeroShotEstimator import ClipZeroShotEstimator, WinClipZeroShotEstimator
from anomaly_detection.CosKnnAnomalyScorer import CosKnnAnomalyScorer
from anomaly_detection.AnomalyDetectors import PatchcoreEstimator, DinomalyEstimator
from anomaly_detection.AnomalyDino import AnomalyDinoEstimator
import utilities.config as cfg

best_anomalydino = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 500,
    "normalizer__max_matches": 100,
    "feature_extractor": "passthrough",
    "scaler": "passthrough",
    "dim_reduction": "passthrough",
    "estimator": AnomalyDinoEstimator(),
    "estimator__n_neighbors": 3,
    "estimator__threshold_percentile": 95,
    }

best_hog_pca_gmm = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.5,
    "normalizer__n_features": 500,
    "normalizer__max_matches": 100,
    "feature_extractor": HogFeatureExtractor(),
    "feature_extractor__orientations": 9,
    
    "feature_extractor__pixels_per_cell": (16, 16),
    
    "feature_extractor__cells_per_block": (2, 2),
    
    "feature_extractor__block_norm": "L2-Hys",
    "dim_reduction__n_components": 50,
    "dim_reduction__svd_solver": "full",
    "estimator": GMMAnomalyEstimator(),
    "estimator__n_components": 3,
    "estimator__covariance_type": "diag",
    "estimator__reg_covar": 1e-4,
    "estimator__n_init": 3,
    "estimator__max_iter": 300,
    "estimator__threshold_percentile": 83,
}

best_dnn = {
    "normalizer__strategy": "orb",
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 500,
    "normalizer__max_matches": 100,
    "feature_extractor": DNNFeatureExtractor(),
    "feature_extractor__layer": 4,
    "feature_extractor__pool_out_size": (3, 3),
    "dim_reduction__n_components": 150,
    "dim_reduction__svd_solver": "full",
    "scaler": "passthrough",
    "estimator": SVMAnomalyEstimator(),
    "estimator__nu": 0.1,
    "estimator__threshold_percentile":100,
}

best_patchcore = {
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 500,
    "normalizer__max_matches": 100,
    "feature_extractor": "passthrough",
    "scaler": "passthrough",
    "dim_reduction": "passthrough",
    "estimator": PatchcoreEstimator(),
    "estimator__coreset_sampling_ratio": 0.05,
    "estimator__num_neighbors": 9,
    "estimator__threshold_percentile": 100
}

best_clip = {
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 500,
    "normalizer__max_matches": 100,
    "feature_extractor": "passthrough",
    "scaler": "passthrough",
    "dim_reduction": "passthrough",
    "estimator": ClipZeroShotEstimator(),
    "estimator__model_name": "PE-Core-L-14-336",
    "estimator__pretrained": "meta",
    "estimator__threshold_percentile": 63,
    "estimator__prompts": ["a standard medical device product label in mint condition", "a medical device product label with compromised quality and damaged parts"]
}

best_dinomaly = {
    "normalizer__downsample_factor": 0.8,
    "normalizer__n_features": 500,
    "normalizer__max_matches": 100,
    "feature_extractor": "passthrough",
    "scaler": "passthrough",
    "dim_reduction": "passthrough",
    "estimator": DinomalyEstimator(),
    "estimator__default_root_dir": str(cfg.BASE_DIR / "anomaly_detection" / "experiments"),
    "estimator__threshold_percentile": 82,
    "estimator__encoder_name": "dinov2reg_vit_base_14",
    "estimator__bottleneck_dropout": 0.4,
    "estimator__remove_class_token": True,
}


normalization_grid = {
    "normalizer__strategy": ["orb"],
    "normalizer__downsample_factor": [0.8],
    "normalizer__n_features": [500],
    "normalizer__max_matches": [100]
}

hog_grid = {
    "feature_extractor": [HogFeatureExtractor()],
    "feature_extractor__orientations": [9, 12],
    
    "feature_extractor__pixels_per_cell": [(8, 8), (16, 16)],
    
    "feature_extractor__cells_per_block": [(2, 2)],
    
    "feature_extractor__block_norm": ["L2-Hys"]}

pca_grid = {
    "dim_reduction__n_components": [150, 50],
    "dim_reduction__svd_solver": ["full"]
}
pca_passthrough = {
    "dim_reduction": ["passthrough"]
}

scaler_passthrough = {
    "scaler": ["passthrough"]
}
scaler_standard = {
    "scaler": [StandardScaler()]
}
feature_extractor_passthrough = {
    "feature_extractor": ["passthrough"]
}

svm_grid = {
    "estimator": [SVMAnomalyEstimator()],
    "estimator__kernel": ["rbf"],
    "estimator__gamma": ["scale"],
    "estimator__nu": [0.1,0.2],
    "estimator__threshold_percentile": [97],
    }

gmm_grid = {
    "estimator": [GMMAnomalyEstimator()],
    "estimator__n_components": [3],
    "estimator__covariance_type": ["diag"],
    "estimator__reg_covar": [1e-4],
    "estimator__n_init": [3],
    "estimator__max_iter": [300]
}
clip_zeroshot_grid =  {
    "estimator": [ClipZeroShotEstimator()],
    "estimator__model_name": ["PE-Core-L-14-336"],
    "estimator__pretrained": ["meta"],
    "estimator__threshold_percentile": [75],
    "estimator__prompts": [["a standard medical device product label in mint condition", "a medical device product label with compromised quality and damaged parts"]]
    # "estimator__prompts": [["a standard medical device product label in mint condition", "a medical device product label with compromised quality and damaged parts"], ["a normal label of a medical device, adhering to all standards with all symbols and information clear and legible", "an anomalous label with misprints, smudges, or other defects that deviate from the normal appearance"],["a photo of a perfect medical device label","a photo of a defective medical spine implant label with structural flaws, misprints, streaks, smudges, or other defects that deviate from the normal appearance"]],
}

dnn_extractor_grid = {
    "feature_extractor": [DNNFeatureExtractor()],
    "feature_extractor__batch_size": [16],
    "feature_extractor__model_name": ["resnet18", "resnet50.fb_swsl_ig1b_ft_in1k"],
    "feature_extractor__pretrained": [True],
    "feature_extractor__layer": [None,4],
    "feature_extractor__pool_out_size": [(3,3)],
}

cos_knn_grid = {
    "estimator": [CosKnnAnomalyScorer()],
    "estimator__n_neighbors": [3,5,10],
    "estimator__threshold_percentile": [97],
    "estimator__use_faiss": [True],
}

patchcore_grid = {
    "estimator": [PatchcoreEstimator()],
    "estimator__backbone": ["wide_resnet50_2"],
    "estimator__layers": [("layer2", "layer3")],
    "estimator__pre_trained": [True],
    "estimator__coreset_sampling_ratio": [0.01, 0.05],
    "estimator__num_neighbors": [3,5,9],
    "estimator__threshold_percentile": [95],
    "estimator__tile_size": [224],
    "estimator__stride": [112, 56],
    "estimator__batch_size": [16],
    "estimator__num_workers": [0],
}

anomalydino_grid = {
    "estimator": [AnomalyDinoEstimator()],
    "estimator__threshold_percentile": [95],
    "estimator__n_neighbors": [3, 5, 9],
    "estimator__batch_size": [16],
}


hog_pca_svm = {
    "normalization_grid": normalization_grid,
    "feature_extr_grid": hog_grid,
    "scaler_grid": scaler_standard,
    "dim_reduction_grid": pca_grid,
    "estimator_grid": svm_grid,
}

hog_pca_gmm = hog_pca_svm.copy()
hog_pca_gmm["estimator_grid"] = gmm_grid

dnn_svm = {
    "normalization_grid": normalization_grid,
    "feature_extr_grid": dnn_extractor_grid,
    "scaler_grid": scaler_passthrough,
    "dim_reduction_grid": pca_grid,
    "estimator_grid": svm_grid,
}


hog_pca_cos_knn = hog_pca_svm.copy()
hog_pca_cos_knn["estimator_grid"] = cos_knn_grid

zero_shot_clip = {
    "normalization_grid": normalization_grid,
    "dim_reduction_grid": pca_passthrough,
    "scaler_grid": scaler_passthrough,
    "feature_extr_grid": feature_extractor_passthrough,
    "estimator_grid": clip_zeroshot_grid,
}

winclip_zeroshot = {
    "normalization_grid": normalization_grid,
    "dim_reduction_grid": pca_passthrough,
    "scaler_grid": scaler_passthrough,
    "feature_extr_grid": feature_extractor_passthrough,
    "estimator_grid": {
        "estimator": [WinClipZeroShotEstimator()],
        "estimator__batch_size": [16],
        "estimator__k_shot": [16],
        "estimator__threshold_percentile": [98],
        "estimator__class_name": ["medical device label"],
    },
}



dnn_cos_knn = {
    "normalization_grid": normalization_grid,
    "scaler_grid": scaler_passthrough,
    "feature_extr_grid": dnn_extractor_grid,
    "dim_reduction_grid": pca_grid,
    "estimator_grid": cos_knn_grid,
}

dnn_gmm = {
    "normalization_grid": normalization_grid,
    "scaler_grid": scaler_passthrough,
    "feature_extr_grid": dnn_extractor_grid,
    "dim_reduction_grid": pca_grid,
    "estimator_grid": gmm_grid,
}
patchcore= {
    "normalization_grid": normalization_grid,
    "scaler_grid": scaler_passthrough,
    "feature_extr_grid": feature_extractor_passthrough,
    "dim_reduction_grid": pca_passthrough,
    "estimator_grid": patchcore_grid,
}
dinomaly={
    "normalization_grid": normalization_grid,
    "scaler_grid": scaler_passthrough,
    "feature_extr_grid": feature_extractor_passthrough,
    "dim_reduction_grid": pca_passthrough,
    "estimator_grid": {
        "estimator": [DinomalyEstimator()],
        "estimator__default_root_dir": [str(cfg.BASE_DIR / "anomaly_detection" / "experiments")],
        "estimator__encoder_name": ["dinov2reg_vit_base_14"],
        "estimator__bottleneck_dropout": [0.4],
        "estimator__decoder_depth": [8],
        "estimator__remove_class_token": [True],
        "estimator__threshold_percentile": [87],
        "estimator__batch_size": [16],
        "estimator__num_workers": [0],
        "estimator__random_state": [20]
    },
}


anomalydino = {
    "normalization_grid": normalization_grid,
    "scaler_grid": scaler_passthrough,
    "feature_extr_grid": feature_extractor_passthrough,
    "dim_reduction_grid": pca_passthrough,
    "estimator_grid": anomalydino_grid
}

bests = {
    "best_hog_pca_gmm": best_hog_pca_gmm,
    "best_dnn": best_dnn,
    "best_patchcore": best_patchcore,
    "best_anomalydino": best_anomalydino,
}

# resnet50_fb_swsl_ig1b_cos_knn = {
#     "normalizer__strategy": "orb",
#     "normalizer__downsample_factor": 0.3,
#     "normalizer__n_features": 250,
#     "normalizer__max_matches": 50,
#     "scaler": "passthrough",
#     "feature_extractor": DNNFeatureExtractor(),
#     "feature_extractor__batch_size": 16,
#     "feature_extractor__model_name": "resnet50.fb_swsl_ig1b_ft_in1k",
#     "feature_extractor__pretrained": True,
#     "dim_reduction__n_components": 64,
#     "dim_reduction__svd_solver": "full",
#     "estimator": CosKnnAnomalyScorer(),
#     "estimator__n_neighbors": 5,
#     "estimator__threshold_percentile": 97,
#     "estimator__use_faiss": True,
#     "estimator__normalize": False,
# }
# resnet50_layer4_pooling3x3_cos_knn = resnet50_fb_swsl_ig1b_cos_knn.copy()
# resnet50_layer4_pooling3x3_cos_knn["feature_extractor__pool_out_size"] = (3,3)
# resnet50_layer4_pooling3x3_cos_knn["feature_extractor__layer"] = 4
    

# winclip = {
#     "normalizer__strategy": "orb",
#     "normalizer__downsample_factor": 0.8,
#     "normalizer__n_features": 250,
#     "normalizer__max_matches": 50,
#     "feature_extractor": "passthrough",
#     "scaler": "passthrough",
#     "dim_reduction": "passthrough",
#     "estimator": WinClipZeroShotEstimator(),
#     "estimator__batch_size": 16,
#     "estimator__threshold_percentile": 90,
# }

# patchcore = {
#     "normalizer__strategy": "orb",
#     "normalizer__downsample_factor": 0.5,
#     "normalizer__n_features": 250,
#     "normalizer__max_matches": 50,
#     "feature_extractor": "passthrough",
#     "scaler": "passthrough",
#     "dim_reduction": "passthrough",
#     "estimator": PatchcoreEstimator(),
#     "estimator__checkpoint_path": str(cfg.BASE_DIR / "anomaly_detection" / "experiments" / "patchcore_results" / "Patchcore" / "v0" / "weights" / "lightning" / "model.ckpt"),
#     "estimator__backbone": "wide_resnet50_2",
#     "estimator__layers": ("layer2", "layer3"),
#     "estimator__pre_trained": True,
#     "estimator__coreset_sampling_ratio": 0.01,
#     "estimator__num_neighbors": 9,
#     "estimator__threshold_percentile": 95,
#     "estimator__tile_size": 224,
#     "estimator__batch_size": 16,
#     "estimator__num_workers": 0,
# }
# patchcore2 = {
#     "normalizer__strategy": "orb",
#     "normalizer__downsample_factor": 0.8,
#     "normalizer__n_features": 250,
#     "normalizer__max_matches": 50,
#     "feature_extractor": "passthrough",
#     "scaler": "passthrough",
#     "dim_reduction": "passthrough",
#     "estimator": PatchcoreEstimator(),
#     "estimator__backbone": "wide_resnet50_2",
#     "estimator__layers": ("layer2", "layer3"),
#     "estimator__pre_trained": True,
#     "estimator__coreset_sampling_ratio": 0.01,
#     "estimator__num_neighbors": 9,
#     "estimator__threshold_percentile": 95,
#     "estimator__tile_size": 128,
#     "estimator__batch_size": 16,
#     "estimator__num_workers": 0,
# }

# dinomaly = {
#     "normalizer__strategy": "orb",
#     "normalizer__downsample_factor": 0.5,
#     "normalizer__n_features": 250,
#     "normalizer__max_matches": 50,
#     "feature_extractor": "passthrough",
#     "scaler": "passthrough",
#     "dim_reduction": "passthrough",
#     "estimator": DinomalyEstimator(),
#     "estimator__default_root_dir": str(cfg.BASE_DIR / "anomaly_detection" / "experiments"),
#     "estimator__encoder_name": "dinov2reg_vit_base_14",
#     "estimator__bottleneck_dropout": 0.2,
#     "estimator__decoder_depth": 8,
#     "estimator__remove_class_token": False,
#     "estimator__threshold_percentile": 95,
#     "estimator__batch_size": 8,
#     "estimator__num_workers": 0,
#     "estimator__random_state": 20,
#     "estimator__batch_size": 16,
#     "estimator__num_workers": 0
# }

# patchcore_dino = patchcore.copy()
# patchcore_dino["estimator__encoder_name"] = "vit_base_patch14_dinov2.lvd142m"
# patchcore_dino["estimator__layers"] = ["blocks.8", "blocks.11"]


# dinomaly_no_cls_token = dinomaly.copy()
# dinomaly_no_cls_token["estimator__remove_class_token"] = True

# anomalydino = {
#     "normalizer__strategy": "orb",
#     "normalizer__downsample_factor": 0.8,
#     "normalizer__n_features": 250,
#     "normalizer__max_matches": 50,
#     "feature_extractor": "passthrough",
#     "scaler": "passthrough",
#     "dim_reduction": "passthrough",
#     "estimator": AnomalyDinoEstimator(),
#     "estimator__threshold_percentile": 97,
# }
