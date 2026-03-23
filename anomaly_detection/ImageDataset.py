import torch
import numpy as np
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, preprocess):
        self.images = images
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pil_image = self._to_pil_rgb(image)
        if pil_image is None:
            return idx, None
        return idx, self.preprocess(pil_image)

    def _to_pil_rgb(self, image):
        if image is None:
            return None
        if not isinstance(image, np.ndarray):
            return None
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")

        return None

def _collate_no_skip_none(batch):
    batch = [(idx, img) for idx, img in batch if img is not None]
    
    if not batch:
        return None, None
        
    idxs, images = zip(*batch)
    return torch.tensor(idxs), torch.stack(images)