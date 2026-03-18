import torch
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image
import open_clip

class ClipFeatureExtractor(TransformerMixin, BaseEstimator):

    def __init__(self, batch_size=64, model_name="ViT-B-32", pretrained="openai", device="cuda"):
        self.batch_size = batch_size
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device if(torch.cuda.is_available() and device == "cuda") else "cpu"
        
    def _load_openclip(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.model.eval()

    def _to_pil_rgb(self, image):
        # clip preprocess expects PIL RGB images, so convert if needed
        if image is None:
            return None
        if not isinstance(image, np.ndarray):
            return None
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")

        return None
    
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
        

        feature_vectors = np.zeros((len(X), output_dim), dtype=np.float32)
        n_valid = 0

        for i in tqdm(range(0, len(X), self.batch_size), desc="Embedding"):
            batch = []
            batch_indices = []

            for j in range(i, min(i + self.batch_size, len(X))):
                pil_image = self._to_pil_rgb(X[j])
                if pil_image is None:
                    continue
                batch.append(self.preprocess(pil_image))
                batch_indices.append(j)

            if not batch:
                continue

            batch = torch.stack(batch).to(self.device)
            batch_feature_v = self.model.encode_image(batch)
            batch_feature_v = batch_feature_v / batch_feature_v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feature_vectors[batch_indices] = batch_feature_v.cpu().float().numpy()
            n_valid += len(batch_indices)

        # Keep zero-vector placeholders for invalid images so sample count stays aligned.
        if n_valid == 0:
            raise RuntimeError("No embeddings computed; all images failed to load?")

        return feature_vectors
