from sklearn.base import TransformerMixin, BaseEstimator
from anomaly_detection.ImageDataset import ImageDataset, _collate_no_skip_none
import numpy as np
import timm
from tqdm import tqdm
import torch
import open_clip
from PIL import Image
import skimage.feature as skif
import cv2

class ORBFeatureExtractor(TransformerMixin, BaseEstimator):
    def __init__(self,
                 n_features=250
                 ):
        self.n_features = n_features

    def fit(self, X, y=None):
        return self

    def _orb_to_fixed_vector(self, descriptors):
        descriptor_len = 32
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_features * descriptor_len, dtype=np.float32)

        descriptors = descriptors[:self.n_features]
        if descriptors.shape[0] < self.n_features:
            pad_rows = self.n_features - descriptors.shape[0]
            padding = np.zeros((pad_rows, descriptor_len), dtype=descriptors.dtype)
            descriptors = np.vstack((descriptors, padding))

        return descriptors.reshape(-1).astype(np.float32)
    
    def transform(self, X):
        feature_vectors = []
        orb = cv2.ORB_create(nfeatures=self.n_features) 

        for image in tqdm(X, desc=f"Extracting ORB features"):
            keypoints, features = orb.detectAndCompute(image, None)
            feature_vectors.append(self._orb_to_fixed_vector(features))

        return np.asarray(feature_vectors, dtype=np.float32)
            
class HogFeatureExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, 
                 orientations=9,
                 pixels_per_cell=(8,8),
                 cells_per_block=(3,3),
                 block_norm='L2-Hys',
                 ):
        self.orientations=orientations
        self.pixels_per_cell=pixels_per_cell
        self.cells_per_block=cells_per_block
        self.block_norm=block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feature_vectors = []

        for image in tqdm(X, desc=f"Extracting HOG features"):
            features = skif.hog(image,
                                    orientations=self.orientations,
                                    pixels_per_cell=self.pixels_per_cell,
                                    cells_per_block=self.cells_per_block,
                                    block_norm=self.block_norm)
            feature_vectors.append(features)

        return np.asarray(feature_vectors, dtype=np.float32)
    
class ClipFeatureExtractor(TransformerMixin, BaseEstimator):

    def __init__(self, batch_size=16, model_name="ViT-B-32", pretrained="openai", device="cuda"):
        self.batch_size = batch_size
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device if(torch.cuda.is_available() and device == "cuda") else "cpu"
        
    def _load_openclip(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.model.eval()

    def fit(self, X, y=None):
        self._load_openclip()
        return self

    @torch.no_grad()
    def transform(self, X):
        if not hasattr(self, "model") or not hasattr(self, "preprocess"):
            self._load_openclip()
        
        if hasattr(self.model.visual, "output_dim"):
            output_dim = self.model.visual.output_dim
        elif hasattr(self.model.visual.trunk, "embed_dim"):
            output_dim = self.model.visual.trunk.embed_dim
        else:
            raise RuntimeError("Unable to infer CLIP output dimension from model visual backbone")
        

        feature_vectors = np.zeros((len(X), output_dim), dtype=np.float32)
        n_valid = 0
        data_loader = torch.utils.data.DataLoader(ImageDataset(X, self.preprocess),
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   pin_memory=(self.device == "cuda"),
                                                   collate_fn=_collate_no_skip_none)

        for idxs, batch in tqdm((data_loader), desc="Extracting CNN features"):
            if idxs is None:
                continue

            batch = batch.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type=self.device, enabled=(self.device == "cuda")):
                batch_features = self.model.encode_image(batch)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feature_vectors[idxs.numpy()] = batch_features.cpu().float().numpy()
            n_valid += len(idxs)
        
        print(f"Computed embeddings, {n_valid} valid out of {len(X)} images.")

        if n_valid == 0:
            raise RuntimeError("No embeddings computed; all images failed to load?")

        return feature_vectors

class CNNFeatureExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, model_name="resnet18", batch_size=16, pretrained=True, device="cuda"):
        self.batch_size = batch_size
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device if(torch.cuda.is_available() and device == "cuda") else "cpu"

    def fit(self, X, y=None):
        self.model = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=0).to(self.device)
        config = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.img_transform = timm.data.create_transform(**config)
        self.model.eval()
        return self

    @torch.no_grad()
    def transform(self, X):
        feature_vectors = np.zeros((len(X), self.model.num_features), dtype=np.float32)
        n_valid = 0

        data_loader = torch.utils.data.DataLoader(ImageDataset(X, self.img_transform),
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   pin_memory=(self.device == "cuda"),
                                                   collate_fn=_collate_no_skip_none)

        for idxs, batch in tqdm((data_loader), desc="Extracting CNN features"):
            if idxs is None:
                continue

            batch = batch.to(self.device, non_blocking=True)
            batch_features = self.model(batch)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feature_vectors[idxs.numpy()] = batch_features.cpu().float().numpy()
            n_valid += len(idxs)

        print(f"Extracted CNN features, {n_valid} valid out of {len(X)} images.")

        if n_valid == 0:
            raise RuntimeError("No features extracted; all images failed to load?")

        return feature_vectors