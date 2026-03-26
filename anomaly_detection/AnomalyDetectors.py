import numpy as np
from sklearn.base import BaseEstimator
from pathlib import Path
import shutil
import tempfile

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from anomaly_detection.ImageDataset import AnomalibImageDataset, _collate_skip_none_anomalib

from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.callbacks import TilerConfigurationCallback

class PatchcoreEstimator(BaseEstimator):
    def __init__(
        self,
        checkpoint_path=None,
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        threshold_percentile=95,
        tile_size=224,
        batch_size=8,
        num_workers=0,
        random_state=22,
        default_root_dir=None,
    ):
        self.checkpoint_path = checkpoint_path
        self.backbone = backbone
        self.layers = layers
        self.pre_trained = pre_trained
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.threshold_percentile = threshold_percentile
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.default_root_dir = default_root_dir
    

        self.engine = None
        self.model = None
        self.transforms = transforms.ToTensor()
        self._threshold = None
        self._tmp_root = None
        self._last_scores_key = None
        self._last_scores = None

    def __del__(self):
        self._cleanup_tmp_root()

    def _cleanup_tmp_root(self):
        if self._tmp_root is None:
            return
        try:
            shutil.rmtree(self._tmp_root, ignore_errors=True)
        finally:
            self._tmp_root = None

    def _ensure_tmp_root(self):
        if self._tmp_root is None:
            self._tmp_root = Path(tempfile.mkdtemp(prefix="patchcore_estimator_"))

    def _write_images(self, images, directory: Path, prefix: str):
        directory.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for i, image in enumerate(images):
            image_path = directory / f"{prefix}_{i:06d}.png"
            if image is None:
                image_paths.append(None)
                continue
            ok = cv2.imwrite(str(image_path), image)
            if not ok:
                raise RuntimeError(f"Failed to write image {image_path}")
            image_paths.append(image_path)
        return image_paths
    
    def _cache_key(self, X):
        return (id(X), len(X))

    def _prepare_dataset(self, images, split_name: str, is_anomaly=None):
        self._ensure_tmp_root()
        split_dir = self._tmp_root / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir, ignore_errors=True)
        image_paths = self._write_images(images, split_dir, prefix=split_name)
        return AnomalibImageDataset(
            images,
            preprocess=self.transforms,
            is_anomaly=is_anomaly,
            image_paths=image_paths,
        )

    def _get_scores(self, X) -> np.ndarray:
        # caching last scores to avoid redundant computation if the same data is scored multiple times
        key = self._cache_key(X)
        if self._last_scores_key == key and self._last_scores is not None:
            return self._last_scores
        
        predict_dataloader = DataLoader(
            dataset=self._prepare_dataset(X, split_name="predict", is_anomaly=False),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_skip_none_anomalib
        )
        predictions = self.engine.predict(model=self.model, dataloaders=predict_dataloader)
        scores = np.full(len(X), np.nan, dtype=np.float32)
        for pred in predictions:
            pred_score = getattr(pred, "pred_score", None)
            pred_paths = getattr(pred, "image_path", None)
            if pred_score is None:
                raise RuntimeError(f"Could not extract pred_score from prediction type {type(pred)}")

            if torch.is_tensor(pred_score):
                pred_score_values = pred_score.detach().cpu().flatten().tolist()
            elif isinstance(pred_score, (list, tuple, np.ndarray)):
                pred_score_values = [float(score) for score in pred_score]
            else:
                pred_score_values = [float(pred_score)]

            if pred_paths is None:
                continue
            if isinstance(pred_paths, str):
                pred_paths = [pred_paths]

            for path_str, score in zip(pred_paths, pred_score_values):
                index_str = Path(path_str).stem.split("_")[-1]
                try:
                    sample_idx = int(index_str)
                except ValueError:
                    continue
                if 0 <= sample_idx < len(scores):
                    scores[sample_idx] = float(score)

        missing = int(np.isnan(scores).sum())
        if missing > 0:
            raise RuntimeError(f"Missing anomaly scores for {missing} samples. Ensure all input images are valid 2D arrays.")
        
        self._last_scores_key = key
        scores = scores.astype(np.float32)
        self._last_scores = scores
        return scores

    def fit(self, X, y=None):

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        tiler_config_callback = TilerConfigurationCallback(enable=True, tile_size=[self.tile_size, self.tile_size], stride=self.tile_size)

        root_dir = self.default_root_dir or "./anomaly_detection/experiments/patchcore_results"
        self.engine = Engine(accelerator="gpu", default_root_dir=root_dir, callbacks=[tiler_config_callback])

        if self.checkpoint_path is not None:
            self.model = Patchcore.load_from_checkpoint(self.checkpoint_path)

        else:
            if X is None or len(X) == 0:
                raise ValueError("X must contain at least one normal training image.")

            train_dataloader = DataLoader(
                dataset=self._prepare_dataset(X, split_name="train", is_anomaly=False),
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
                collate_fn=_collate_skip_none_anomalib
            )

            self.model = Patchcore(
                backbone=self.backbone,
                layers=list(self.layers),
                pre_trained=self.pre_trained,
                coreset_sampling_ratio=self.coreset_sampling_ratio,
                num_neighbors=self.num_neighbors,
            )

            self.engine.fit(model=self.model, train_dataloaders=train_dataloader)

        train_scores = self.score_samples(X)
        self._threshold = float(np.percentile(train_scores, self.threshold_percentile))
        self.is_fitted_ = True
        return self

    def score_samples(self, X):
        if self.engine is None or self.model is None:
            raise ValueError("Estimator has not been fitted yet.")
        if X is None or len(X) == 0:
            return np.asarray([], dtype=np.float32)
        return self._get_scores(X) 

    def predict(self, X):
        if self._threshold is None:
            raise ValueError("Estimator has not been fitted yet.")
        scores = self.score_samples(X)
        return (scores > self._threshold).astype(int)

class DinomalyEstimator(BaseEstimator):
    def __init__(self):
        pass
