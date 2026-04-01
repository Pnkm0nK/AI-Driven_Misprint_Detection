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
from abc import ABC, abstractmethod

from anomalib.models import Patchcore, Dinomaly
from anomalib.engine import Engine
from anomalib.callbacks import TilerConfigurationCallback

from lightning.pytorch import LightningDataModule

class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset,
        category="label",
        name="dataset",
        val_dataset=None,
        max_epochs=20,
        batch_size=8,
        num_workers=0,
        collate_fn=None
    ):
        super().__init__()
        # anomalib expects dataset metadata to be present and path-safe strings
        self.category = str(category or "label")
        self.name = str(name or "dataset")
        self.dataset_name = self.name
        self.train_set = train_dataset
        self.val_set = val_dataset or train_dataset
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()

class AnomalibEstimator(BaseEstimator, ABC):
    def __init__(
        self,
        checkpoint_path=None,
        threshold_percentile=95,
        batch_size=8,
        num_workers=0,
        random_state=20,
        default_root_dir=None,
        max_epochs=20,
    ):
        self.checkpoint_path = checkpoint_path
        self.threshold_percentile = threshold_percentile
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.default_root_dir = default_root_dir
        self.max_epochs = max_epochs

        self.engine = None
        self.model = None
        self.transforms = lambda x: x #  identity by default, overwrite for specific models 
        self._threshold = None
        self._tmp_root = None
        self._last_scores_key = None
        self._last_scores = None

    def __del__(self):
        if hasattr(self, "_tmp_root"):
            self._cleanup_tmp_root()

    @abstractmethod
    def _build_model(self, checkpoint_path=None):
        pass

    @abstractmethod
    def _build_callbacks(self):
        pass

    def _cleanup_tmp_root(self):
        if self._tmp_root is None:
            return
        try:
            shutil.rmtree(self._tmp_root, ignore_errors=True)
        finally:
            self._tmp_root = None

    def _ensure_tmp_root(self):
        if self._tmp_root is None:
            self._tmp_root = Path(tempfile.mkdtemp(prefix="estimator_tmp_"))

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
        # Anomalib models expect file paths for their datasets for visualization and logging purposes, so writing normalized temp images during pipeline execution
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
            dataset=self._prepare_dataset(X, split_name="predict"),
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
        callbacks = self._build_callbacks()
        self.engine = Engine(
            accelerator="gpu",
            max_epochs=self.max_epochs,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            default_root_dir=self.default_root_dir,
            callbacks=callbacks)

        if self.checkpoint_path is not None:
            self.model = self._build_model(self.checkpoint_path)

        else:
            if X is None or len(X) == 0:
                raise ValueError("X must contain at least one normal training image.")

            train_ds = self._prepare_dataset(X, split_name="train", is_anomaly=False)
            self.model = self._build_model()
            datamodule = CustomDataModule(
                train_dataset=train_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=_collate_skip_none_anomalib)

            self.engine.fit(model=self.model, datamodule=datamodule)

        train_scores = self._get_scores(X)
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



class PatchcoreEstimator(AnomalibEstimator):
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
        random_state=20,
        default_root_dir=None,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            threshold_percentile=threshold_percentile,
            batch_size=batch_size,
            num_workers=num_workers,
            random_state=random_state,
            default_root_dir=default_root_dir,
        )
        self.backbone = backbone
        self.layers = layers
        self.pre_trained = pre_trained
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.transforms = transforms.ToTensor()
        self.num_neighbors = num_neighbors
        self.tile_size = tile_size
    
    def _build_model(self, checkpoint_path=None):
        if self.checkpoint_path is not None:
            return Patchcore.load_from_checkpoint(self.checkpoint_path)
        return Patchcore(
                backbone=self.backbone,
                layers=list(self.layers),
                pre_trained=self.pre_trained,
                coreset_sampling_ratio=self.coreset_sampling_ratio,
                num_neighbors=self.num_neighbors,
            )
    
    def _build_callbacks(self):
        return [TilerConfigurationCallback(tile_size=self.tile_size)]


class DinomalyEstimator(AnomalibEstimator):
    def __init__(
        self,
        checkpoint_path=None,
        encoder_name="dinov2reg_vit_base_14",
        bottleneck_dropout=0.2,
        decoder_depth=8,
        remove_class_token=False,
        threshold_percentile=95,
        max_epochs=20,
        batch_size=8,
        num_workers=0,
        random_state=20,
        default_root_dir=None,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            threshold_percentile=threshold_percentile,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            random_state=random_state,
            default_root_dir=default_root_dir,
        )
        self.encoder_name=encoder_name
        self.bottleneck_dropout=bottleneck_dropout
        self.decoder_depth=decoder_depth
        self.remove_class_token=remove_class_token
        self.transforms = transforms.ToTensor()
    
    def _build_model(self, checkpoint_path=None):
        if checkpoint_path is not None:
            return Dinomaly.load_from_checkpoint(checkpoint_path)

        model = Dinomaly(
            encoder_name=self.encoder_name,
            bottleneck_dropout=self.bottleneck_dropout,
            decoder_depth=self.decoder_depth,
            remove_class_token=self.remove_class_token,
        )
        model.configure_pre_processor(crop_size=448)
        return model
    
    def _build_callbacks(self):
        return []
