import cv2
from pathlib import Path
import sklearn.mixture as skm
import sklearn as skl
import numpy as np
import mlflow
import mlflow.sklearn

from anomaly_detection.LabelNormalizer import LabelNormalizer
from anomaly_detection.FeatureExtractor import FeatureExtractor

def load_label_dataset(label_type, test_size=0.1, seed=20):
    base_dir = Path(__file__).parent.resolve()
    data_path = base_dir /"anomaly_detection" / "train_data" / label_type

    anomalous_data_path = base_dir / "anomaly_detection" / "anomalous_data" / label_type
    images = [cv2.imread(str(img), cv2.IMREAD_GRAYSCALE) for img in data_path.glob("*.jpg")]
    print(f"Loaded {len(images)} normal images from {data_path}")
    test_anomaly = [cv2.imread(str(img), cv2.IMREAD_GRAYSCALE) for img in anomalous_data_path.glob("*.jpg")]

    train, test_normal = skl.model_selection.train_test_split(images, test_size=test_size, random_state=seed)
    return train, test_normal, test_anomaly

def run_experiment(
    experiment_name: str = "feature-anomaly-detection",
    run_name: str = "gmm_orb_pca",
    label_type: str = "151", seed: int = 20,
):
    base_dir = Path(__file__).parent.resolve()
    tracking_dir = base_dir / "anomaly_detection" / "experiments"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    threshold_percentile = 5
    pipeline_parameters = {
        "normalizer__strategy": "orb",
        "normalizer__n_features": 150,
        "feature_extractor__strategy": "orb",
        "feature_extractor__n_features": 250,
        "pca__n_components": 50,
        "pca__svd_solver": "full",
        "estimator": skm.GaussianMixture(),
        "estimator__n_components": 1,
        "estimator__covariance_type": "diag",
        "estimator__reg_covar": 1e-4,
        "estimator__n_init": 3,
        "estimator__max_iter": 300,
    }

    train, test_normal, test_anomaly = load_label_dataset(label_type=label_type, seed=seed)

    mlflow.set_experiment(experiment_name)

    train_pipeline = skl.pipeline.Pipeline(steps=[
        ("normalizer", LabelNormalizer()),
        ("feature_extractor", FeatureExtractor()),
        ("scaler", skl.preprocessing.StandardScaler()),
        ("pca", skl.decomposition.PCA(random_state=seed, whiten=True)),
        ("estimator", skm.GaussianMixture(
            n_components=1,
        )),
    ])
    train_pipeline.set_params(**pipeline_parameters)

    with mlflow.start_run(run_name=run_name):
        additional_params = {
            "data_path": str(base_dir / "anomaly_detection"),
            "label_type": label_type,
            "n_images": len(train) + len(test_anomaly) + len(test_normal),
            "train_size": len(train),
            "normal_test_size": len(test_normal),
            "anomalous_test_size": len(test_anomaly),
            "threshold_percentile": threshold_percentile,
            "seed": seed,
        }

        mlflow.log_params({
            **pipeline_parameters,
            **additional_params,
        })

        train_pipeline.fit(train)

        train_scores = train_pipeline.score_samples(train)
        threshold = np.percentile(train_scores, threshold_percentile)
        normal_test_scores = train_pipeline.score_samples(test_normal)
        anomalous_test_scores = train_pipeline.score_samples(test_anomaly)

        normal_test_is_anomaly = normal_test_scores < threshold
        anomalous_test_is_anomaly = anomalous_test_scores < threshold

        y_true = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_anomaly))])
        y_pred = np.concatenate([normal_test_is_anomaly, anomalous_test_is_anomaly])

        tn, fp, fn, tp = skl.metrics.confusion_matrix(y_true, y_pred).ravel().tolist()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        f1_score = skl.metrics.f1_score(y_true, y_pred)

        metrics = {
            "avg_train_log_likelihood": float(train_scores.mean()),
            "avg_normal_test_log_likelihood": float(normal_test_scores.mean()),
            "avg_anomalous_test_log_likelihood": float(anomalous_test_scores.mean()),
            "threshold": float(threshold),
            "n_normal_test_anomalies": int(normal_test_is_anomaly.sum()),
            "n_anomalous_test_anomalies": int(anomalous_test_is_anomaly.sum()),
            "test_anomaly_fraction": float(normal_test_is_anomaly.mean()),
            "min_train_score": float(train_scores.min()),
            "max_train_score": float(train_scores.max()),
            "min_normal_test_score": float(normal_test_scores.min()),
            "max_normal_test_score": float(normal_test_scores.max()),
            "min_anomalous_test_score": float(anomalous_test_scores.min()),
            "max_anomalous_test_score": float(anomalous_test_scores.max()),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "tn": float(tn),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "false_positive_rate": float(false_positive_rate),
            "accuracy": float(accuracy),
            "f1_score": float(f1_score),
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(train_pipeline, artifact_path="pipeline_model")

        print("\n=== Experiment Summary ===")
        print(f"Experiment: {experiment_name}")
        print(f"Run name: {run_name}")
        print(f"Label type: {label_type}")
        print(f"Seed: {seed}")
        print(f"MLflow tracking dir: {tracking_dir}")
        print(f"Train/NormalTest/AnomalyTest: {len(train)}/{len(test_normal)}/{len(test_anomaly)}")

        print("\n=== Score Statistics ===")
        print(f"Train mean score: {train_scores.mean():.4f}  min: {train_scores.min():.4f}  max: {train_scores.max():.4f}")
        print(f"Normal test mean score: {normal_test_scores.mean():.4f}  min: {normal_test_scores.min():.4f}  max: {normal_test_scores.max():.4f}")
        print(f"Anomaly test mean score: {anomalous_test_scores.mean():.4f}  min: {anomalous_test_scores.min():.4f}  max: {anomalous_test_scores.max():.4f}")
        print(f"Threshold ({threshold_percentile}th percentile): {threshold:.4f}")

        print("\n=== Detection Statistics ===")
        print(f"TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"False positive rate: {false_positive_rate:.4f}")
        print(f"F1 score: {f1_score:.4f}")
        print(f"MLflow run id: {mlflow.active_run().info.run_id}")



if __name__ == "__main__":
    run_experiment()