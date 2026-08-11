import time

import cv2
import utilities.config as config
from pathlib import Path
import sklearn as skl
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import ParameterGrid
import utilities.plots as plt

from anomaly_detection.LabelNormalizer import LabelNormalizer
from anomaly_detection.FeatureExtractors import HogFeatureExtractor
from anomaly_detection.GMMAnomalyEstimator import GMMAnomalyEstimator
import anomaly_detection.parameter_sets as param_sets

JSON_PARAMS_PATH = Path(__file__).parent.resolve() / "anomaly_detection" / "parameter_sets"


def _serialize_params_for_logging(params):
    serialized = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)):
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


def load_label_dataset(label_type, load_test=True):
    base_dir = Path(__file__).parent.resolve()
    data_path = base_dir /"anomaly_detection" / "train_data" / label_type
    test_data_path = base_dir / "anomaly_detection" / "test" / label_type
    anomalous_data_path = base_dir / "anomaly_detection" / "anomalous_data" / label_type
    train_images = [cv2.imread(str(img), cv2.IMREAD_GRAYSCALE) for img in data_path.glob("*.jpg")]
    print(f"Loaded {len(train_images)} normal images from {data_path} for training")
    if load_test:
        test_normal = [cv2.imread(str(img), cv2.IMREAD_GRAYSCALE) for img in test_data_path.glob("*.jpg")]
        test_anomaly = [cv2.imread(str(img), cv2.IMREAD_GRAYSCALE) for img in anomalous_data_path.glob("*.jpg")]

        return train_images, test_normal, test_anomaly
    return train_images

def _build_pipeline(label_type: str, seed: int):
    return skl.pipeline.Pipeline(steps=[
        ("normalizer", LabelNormalizer(template_type=label_type)),
        ("feature_extractor", HogFeatureExtractor()),
        ("scaler", skl.preprocessing.StandardScaler()),
        ("dim_reduction", skl.decomposition.PCA(random_state=seed, whiten=True)),
        ("estimator", GMMAnomalyEstimator(random_state=seed))]
    )


def _evaluate_detector(y_scores, y_true, threshold):
    '''
    Evaluates the performance of the anomaly detection pipeline on the provided normal and anomalous samples, returning a dictionary of evaluation metrics

    :param pipeline: The fitted anomaly detection pipeline to evaluate
    :param normal_samples: A list of normal samples to evaluate
    :param anomalous_samples: A list of anomalous samples to evaluate
    :return: A dictionary containing evaluation metrics such as:
    TP, FP, FN, TN, precision, recall, specificity, false positive rate, accuracy, and F1 score
    
    '''
    print(f"Evaluating detector on {len(y_true)} samples")
    y_pred = (y_scores > threshold).astype(int)

    tn, fp, fn, tp = skl.metrics.confusion_matrix(y_true, y_pred).ravel().tolist()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1_score = skl.metrics.f1_score(y_true, y_pred)

    return {
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
        "test_anomaly_fraction": float(y_true.mean()),
    }

def run_custom_grid_search(normalization_grid, feature_extr_grid, scaler_grid, dim_reduction_grid, estimator_grid, 
                           experiment_name="feature-anomaly-detection-validation", run_name="custom_grid_search", label_type="151"):
    
    raw_param_grid = {
        **normalization_grid,
        **feature_extr_grid,
        **scaler_grid,
        **dim_reduction_grid,
        **estimator_grid
    }
    
    param_combinations = list(ParameterGrid(raw_param_grid))

    base_dir = Path(__file__).parent.resolve()
    db_path = base_dir / "anomaly_detection" / "experiments" / "mlflow.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment(experiment_name)
    
    best_auroc = -1.0
    best_params = None
    
    # Start Parent Run
    with mlflow.start_run(run_name=run_name) as parent_run:
        print(f"Starting grid search over {len(param_combinations)} combinations...")
        
        for idx, params in enumerate(param_combinations):
            print(f"Evaluating candidate {idx + 1}/{len(param_combinations)}...")
            
            metrics_summary = validate_pipeline(
                pipeline_parameters=params,
                run_name=f"candidate_{idx}",
                label_type=label_type,
                experiment_name=experiment_name,
                nested=True
            )
            
            current_auroc = metrics_summary.get("cv_mean_auroc", 0.0)
            if current_auroc > best_auroc:
                best_auroc = current_auroc
                best_params = params

        if best_params:
            mlflow.log_params({f"best_{k}": str(v) for k, v in best_params.items()})
            mlflow.log_metric("best_cv_mean_auroc", best_auroc)
            print(f"Grid search complete. Best CV Mean AUROC: {best_auroc:.4f}")

def validate_pipeline(pipeline_parameters,
                      run_name: str,
                      label_type: str="151",
                      seed: int = 20,
                      cv_folds: int = 5,
                      experiment_name: str = "feature-anomaly-detection-validation",
                      anomaly_mix_in_ratio: float = 1.0,
                      nested: bool = False):
        '''
        Validates the anomaly detection pipeline using k-fold cross-validation on the training dataset, with mixing in of anomalous samples into the validation folds. Logs results and metrics to MLflow

        :param pipeline_parameters: A dictionary of parameters to set on the pipeline before fitting
        :param run_name: The name of the MLflow run to log results under
        :param label_type: The label type to load and validate on
        :param seed: The random seed to use for reproducibility
        :param cv_folds: The number of cross-validation folds to use
        :param experiment_name: The name of the MLflow experiment to log results under
        :param anomaly_mix_in_ratio: The ratio of anomalous samples to mix into the validation folds (e.g. 0.5 means mix in anomalous samples equal to 50% of the normal samples in each validation fold)
        '''

        base_dir = Path(__file__).parent.resolve()
        anomaly_set_path = base_dir / "anomaly_detection" / "anomalous_data" / label_type / "validation"
        anomaly_set = [cv2.imread(str(img), cv2.IMREAD_GRAYSCALE) for img in anomaly_set_path.glob("*.jpg")]
        if not nested:
            db_path = base_dir / "anomaly_detection" / "experiments" / "mlflow.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
            mlflow.set_experiment(experiment_name)
        
        train = load_label_dataset(label_type=label_type, load_test=False)

        threshold_percentile = pipeline_parameters.get("estimator__threshold_percentile", 10)
        additional_params = {
            "data_path": str(base_dir / "anomaly_detection"),
            "label_type": label_type,
            "n_images": len(train),
            "train_size": len(train),
            "cv_folds": cv_folds,
            "anomaly_mix_in_ratio": anomaly_mix_in_ratio,
            "threshold_percentile": threshold_percentile,
            "seed": seed,
        }

        with mlflow.start_run(run_name=run_name, nested=nested):
            mlflow.log_params({
                **_serialize_params_for_logging(pipeline_parameters),
                **additional_params,
            })

            rng = np.random.default_rng(seed)
            kfold = skl.model_selection.KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            fold_metrics = []
            trace = {}

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train), start=1):
                fold_pipeline = _build_pipeline(label_type=label_type, seed=seed)
                fold_pipeline.set_params(**pipeline_parameters)
                fold_train = [train[i] for i in train_idx]
                fold_val_normal = [train[i] for i in val_idx]

                n_anomaly = min(len(anomaly_set), int(len(fold_val_normal) * anomaly_mix_in_ratio))
                if n_anomaly > 0:
                    anomaly_indices = rng.choice(len(anomaly_set), size=n_anomaly, replace=False)
                    fold_val_anomaly = [anomaly_set[i] for i in anomaly_indices]
                else:
                    fold_val_anomaly = []

                fold_pipeline.fit(fold_train)
                fold_scores = fold_pipeline.named_steps['estimator']._last_scores
                trace[f"fold_{fold_idx}_train_scores"] = fold_scores.tolist()
                start_time = time.time()
                y_scores = fold_pipeline.score_samples(fold_val_normal)
                time_taken = time.time() - start_time
                time_per_sample = time_taken / len(fold_val_normal) if len(fold_val_normal) > 0 else 0.0
                trace[f"fold_{fold_idx}_val_normal_scores"] = y_scores.tolist()
                y_true = np.zeros(len(fold_val_normal))
                fold_threshold = np.percentile(fold_scores, threshold_percentile)
                fold_metrics_current = {
                    "avg_train_score": float(fold_scores.mean()),
                    "avg_val_normal_score": float(y_scores.mean()),
                    "time_per_sample": float(time_per_sample),
                    "threshold": float(fold_threshold),
                    "val_normal_size": len(fold_val_normal),
                    "val_anomaly_size": len(fold_val_anomaly),

                }

                if len(fold_val_anomaly) > 0:
                    fold_val_anomaly_scores = fold_pipeline.score_samples(fold_val_anomaly)
                    trace[f"fold_{fold_idx}_val_anomaly_scores"] = fold_val_anomaly_scores.tolist()
                    y_true = np.concatenate([y_true, np.ones(len(fold_val_anomaly))])
                    y_scores = np.concatenate([y_scores, fold_val_anomaly_scores])

                    fpr, tpr, thresholds=skl.metrics.roc_curve(y_true, y_scores)
                    distances = np.sqrt(fpr**2 + (1 - tpr)**2)
                    best_idx = np.argmin(distances)
                    best_threshold = thresholds[best_idx]
                    best_percentile = np.floor(np.mean(fold_scores <= best_threshold) * 100)

                    fold_metrics_current.update({
                        "auroc": skl.metrics.roc_auc_score(y_true, y_scores),
                        "average_precision": skl.metrics.average_precision_score(y_true, y_scores),
                        "avg_val_anomaly_score": float(fold_val_anomaly_scores.mean()),       
                        "best_threshold": float(best_threshold),
                        "best_percentile": float(best_percentile),
                        **_evaluate_detector(
                            y_scores=y_scores,
                            y_true=y_true,
                            threshold=best_threshold,
                        )
                    })

                fold_metrics.append(fold_metrics_current)

            if fold_metrics:
                metric_keys = {k for metrics in fold_metrics for k in metrics.keys()}
                summary = {}
                for key in metric_keys:
                    values = [m[key] for m in fold_metrics if key in m]
                    if values:
                        summary[f"cv_mean_{key}"] = float(np.mean(values))
                        summary[f"cv_std_{key}"] = float(np.std(values))
                mlflow.log_figure(plt.generate_roc_plot(y_true, y_scores), f"{run_name}_roc_curve.png")
                mlflow.log_metrics(summary)
                mlflow.log_dict(trace, f"{run_name}_trace.json")
                return summary
            return {}


def run_pipeline(
    pipeline_parameters: dict,
    experiment_name: str = "feature-anomaly-detection",
    run_name: str = "gmm_orb_pca",
    label_type: str = "151",
    seed: int = 20,
    model_path: str | None = None,
):
    base_dir = config.BASE_DIR
    db_path = base_dir / "anomaly_detection" / "experiments" / "mlflow.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

    threshold_percentile = pipeline_parameters.get("estimator__threshold_percentile", 10)

    train, test_normal, test_anomaly = load_label_dataset(label_type=label_type)
    additional_params = {
        "data_path": str(base_dir / "anomaly_detection"),
        "label_type": label_type,
        "n_images": len(train),
        "train_size": len(train),
        "test_normal_size": len(test_normal),
        "test_anomaly_size": len(test_anomaly),
        "threshold_percentile": threshold_percentile,
        "seed": seed,
    }

    mlflow.set_experiment(experiment_name)

    if model_path is not None:
        train_pipeline = mlflow.sklearn.load_model(model_path)
    else:
        train_pipeline = _build_pipeline(label_type=label_type, seed=seed)
        train_pipeline.set_params(**pipeline_parameters)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **_serialize_params_for_logging(pipeline_parameters),
            **additional_params,
        })
        trace = {}
        if model_path is None:
            print(f"Fitting model...")
            train_pipeline.fit(train)
            train_scores = train_pipeline.named_steps['estimator']._last_scores
            trace["train_scores"] = train_scores.tolist()
        else:
            train_scores = train_pipeline.score_samples(train)
            trace["train_scores"] = train_scores.tolist()


        print(f"Scoring normal test data...")
        start_time = time.time()
        normal_test_scores = train_pipeline.score_samples(test_normal)
        time_taken = time.time() - start_time
        trace["normal_test_scores"] = normal_test_scores.tolist()


        print(f"Scoring anomalous test data...")
        start_time = time.time()
        anomalous_test_scores = train_pipeline.score_samples(test_anomaly)
        time_taken += time.time() - start_time
        trace["anomalous_test_scores"] = anomalous_test_scores.tolist()
        time_per_sample = time_taken / (len(test_normal) + len(test_anomaly)) if (len(test_normal) + len(test_anomaly)) > 0 else 0.0

        y_true = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_anomaly))])
        y_scores = np.concatenate([normal_test_scores, anomalous_test_scores])

        trace["y_true"] = y_true.tolist()
        trace["y_scores"] = y_scores.tolist()
        fpr, tpr, thresholds = skl.metrics.roc_curve(y_true, y_scores)

        roc_auc = skl.metrics.roc_auc_score(y_true, y_scores)
        

        if train_scores is None:
            threshold = np.percentile(y_scores, threshold_percentile)
        else:
            threshold = np.percentile(train_scores, threshold_percentile)
        fpr, tpr, thresholds = skl.metrics.roc_curve(y_true, y_scores)

        metrics = {
            "avg_normal_test_score": float(normal_test_scores.mean()),
            "avg_anomalous_test_score": float(anomalous_test_scores.mean()),
            "threshold": float(threshold),
            "min_normal_test_score": float(normal_test_scores.min()),
            "max_normal_test_score": float(normal_test_scores.max()),
            "time_per_sample": float(time_per_sample),
            "min_anomalous_test_score": float(anomalous_test_scores.min()),
            "max_anomalous_test_score": float(anomalous_test_scores.max()),
            "AP": float(skl.metrics.average_precision_score(y_true, y_scores)),
            "AUROC": float(roc_auc),
            **_evaluate_detector(
                y_scores=y_scores,
                y_true=y_true,
                threshold=threshold
            ),
        }
        if train_scores is not None:
            metrics.update({
                "avg_train_score": float(train_scores.mean()),
                "min_train_score": float(train_scores.min()),
                "max_train_score": float(train_scores.max()),
            })
        
        roc_plot = plt.generate_roc_plot(fpr, tpr)
        pr_plot = plt.generate_pr_plot(y_true, y_scores)
        mlflow.log_figure(roc_plot, "roc_curve.png")
        mlflow.log_figure(pr_plot, "pr_curve.png")
        mlflow.log_dict(trace, "execution_trace.json")
        mlflow.log_metrics(metrics)

        if model_path is None:
            mlflow.sklearn.log_model(train_pipeline, artifact_path="pipeline_model")

        print("\n=== Experiment Summary ===")
        print(f"Experiment: {experiment_name}")
        print(f"Run name: {run_name}")
        print(f"Label type: {label_type}")
        print(f"Seed: {seed}")
        print(f"Train/NormalTest/AnomalyTest: {len(train)}/{len(test_normal)}/{len(test_anomaly)}")

        print("\n=== Score Statistics ===")
        if train_scores is not None:
            print(f"Train mean score: {train_scores.mean():.4f}  min: {train_scores.min():.4f}  max: {train_scores.max():.4f}")
        print(f"Normal test mean score: {normal_test_scores.mean():.4f}  min: {normal_test_scores.min():.4f}  max: {normal_test_scores.max():.4f}")
        print(f"Anomaly test mean score: {anomalous_test_scores.mean():.4f}  min: {anomalous_test_scores.min():.4f}  max: {anomalous_test_scores.max():.4f}")
        print(f"Threshold ({threshold_percentile}th percentile): {threshold:.4f}")

        print("\n=== Detection Statistics ===")
        print(f"AUROC: {roc_auc:.4f}")
        print(f"TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}  TN: {metrics['tn']}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"False positive rate: {metrics['false_positive_rate']:.4f}")
        print(f"F1 score: {metrics['f1_score']:.4f}")
        print(f"MLflow run id: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":

    experiment_name = "test-anomaly-detection"
    run_name = "dinomaly_no_crop"

    # experiment_name = "cross-val-anomaly-detection"
    # run_name = "dinomaly"

    parameters = param_sets.best_dinomaly

    # run_custom_grid_search(**parameters, run_name=run_name, experiment_name=experiment_name)

    # validate_pipeline(pipeline_parameters=parameters, run_name=run_name)

    run_pipeline(
        pipeline_parameters=parameters,
        experiment_name=experiment_name,
        run_name=run_name,
        label_type="151",
        seed=20,
        # model_path=str("file:///C:/Users/zyhmam2/OneDrive - Medtronic PLC/Desktop/Bachelor thesis/OCR_bachelor/mlruns/4/models/m-bfab3cd08b7d47479d4af38c5880c2a4/artifacts")
        # model_path=str("file:///C:/Users/zyhmam2/OneDrive - Medtronic PLC/Desktop/Bachelor thesis/OCR_bachelor/mlruns/4/models/m-84e22c7158c94b7e947fa8f5493cade2/artifacts")
    )

    # )
    # for run_name, params in param_sets.bests.items():
    #     run_pipeline(
    #         pipeline_parameters=params,
    #         experiment_name=experiment_name,
    #         run_name=run_name,
    #         label_type="151",
    #         seed=20,
    #     )
