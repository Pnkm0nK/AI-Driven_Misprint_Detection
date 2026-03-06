import mlflow
from ultralytics import YOLO
from pathlib import Path

root = Path(__file__).parent.resolve()

db_path = root / "yolo_runs" / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

model = YOLO("yolo26n-cls.pt")
data_path = root.parent / "yolo_data"
runs_path = root / "yolo_runs"
params = {
    "imgsz": 640,
    "data": str(data_path),
    "batch": 16,
    "epochs": 60,
    "name": "label_classify",
    "project": str(runs_path),
    "exist_ok": True,
    "degrees": 180,
    "fliplr": 0.0,
    "flipud": 0.0,
}

if __name__ == "__main__":
    mlflow.set_experiment("yolo_label_classify")

    with mlflow.start_run():
        mlflow.log_params(params)
        results = model.train(**params)

        if results and hasattr(results, "results_dict"):
            mlflow.log_metrics({k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))})