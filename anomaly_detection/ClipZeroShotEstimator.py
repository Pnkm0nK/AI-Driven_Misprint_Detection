from sklearn.base import BaseEstimator
from anomaly_detection.ImageDataset import ImageDataset, _collate_no_skip_none
from anomalib.models.image.winclip.torch_model import WinClipModel
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import open_clip
import numpy as np


class WinClipZeroShotEstimator(BaseEstimator):
    def __init__(self, batch_size=16, class_name="Spine implant label", threshold=0.5, device="cuda"):
        self.batch_size = batch_size
        self.class_name = class_name
        self.threshold = threshold
        self.device = device if(torch.cuda.is_available() and device == "cuda") else "cpu"

    def _load_winclip(self):
        self.model = WinClipModel(class_name=self.class_name)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.img_transform = transforms.Compose([
            # winclip expects 240x240 input
            transforms.Resize((240, 240)),
            transforms.ToTensor()
            ])


    @torch.no_grad()
    def fit(self, X, y=None):
        self._load_winclip()
        self.is_fitted_ = True
        return self

    @torch.no_grad()
    def score_samples(self, X):
        if not hasattr(self, "model") or not hasattr(self, "img_transform"):
            self._load_winclip()

        total_scores = np.zeros(len(X), dtype=np.float32)
        n_valid = 0
        
        data_loader = torch.utils.data.DataLoader(ImageDataset(X,
                                                               self.img_transform),
                                                batch_size=self.batch_size, shuffle=False,
                                                num_workers=0,
                                                pin_memory=(self.device == "cuda"),
                                                collate_fn=_collate_no_skip_none)

        for idxs, batch in tqdm(data_loader, desc="WinClip scoring"):

            if idxs is None:
                continue

            batch = batch.to(self.device, non_blocking=True)
            if len(batch) <= 0:
                continue

            with torch.amp.autocast(device_type=self.device, enabled=(self.device == "cuda")):
                batch_scores = self.model(batch)
            total_scores[idxs.numpy()] = batch_scores.pred_score.cpu().numpy()
            n_valid += len(idxs)
        print(f"Computed WinClip scores, {n_valid} valid out of {len(X)} images.")

        if n_valid == 0:
            raise RuntimeError("No WinClip scores computed; all images failed to load?")

        return total_scores

    def predict(self, X):
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(int)

class ClipZeroShotEstimator(BaseEstimator):
    def __init__(self, batch_size=16, model_name="ViT-B-32", pretrained="openai",prompts=None, device="cuda"):
        self.batch_size = batch_size
        self.model_name = model_name
        self.pretrained = pretrained
        if not prompts:
            prompts = ["normal label of a spine implant, adhering to all standards with no misprints", "anomalous label with misprints, smudges, or other defects that deviate from the normal appearance"]
        self.prompts = prompts
        self.device = device if(torch.cuda.is_available() and device == "cuda") else "cpu"

    def _load_openclip(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

        return None
    def fit(self, X, y=None):
        self._load_openclip()
        self.is_fitted_ = True
        return self

    @torch.no_grad()
    def score_samples(self, X):
        if not hasattr(self, "model") or not hasattr(self, "preprocess"):
            self._load_openclip()
        
        text= self.tokenizer(self.prompts)
        text_features = self.model.encode_text(text.to(self.device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


        total_probabilities = np.zeros((len(X), 2), dtype=np.float32)
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
                text_probs = (100.0 * batch_features @ text_features.T).softmax(dim=-1)
                total_probabilities[idxs.numpy()] = text_probs.cpu().float().numpy()

            if not batch:
                continue
            n_valid += len(idxs)

        print(f"Computed probabilities, {n_valid} valid out of {len(X)} images.")

        if n_valid == 0:
            raise RuntimeError("No embeddings computed; all images failed to load?")

        return total_probabilities[:, 1]
    
    def predict(self, X):
        probabilities = self.score_samples(X)
        return (probabilities > 0.5).astype(int)