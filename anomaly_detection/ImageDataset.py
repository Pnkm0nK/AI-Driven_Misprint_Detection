import torch
import numpy as np
from PIL import Image
from anomalib.data import ImageItem, ImageBatch

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, preprocess):
        self.images = images
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pil_image = to_pil_rgb(image)
        if pil_image is None:
            return idx, None
        return idx, self.preprocess(pil_image)

class AnomalibImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, preprocess, is_anomaly=None, image_paths=None):
        self.images = images
        self.preprocess = preprocess
        self.is_anomaly = is_anomaly
        self.image_paths = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pil_image = to_pil_rgb(image)
        if pil_image is None:
            return None
        tensor_image = self.preprocess(pil_image)
        class_label = 1 if self.is_anomaly else 0
        image_path = None
        if self.image_paths is not None:
            image_path = str(self.image_paths[idx])
        item = ImageItem(image=tensor_image, gt_label=class_label, image_path=image_path)
        return item 

def _collate_skip_none_anomalib(batch):
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
    images = torch.stack([item.image for item in batch])
    gt_labels = torch.tensor([item.gt_label for item in batch], dtype=torch.int64)
    image_paths = [item.image_path for item in batch]
    
    # Return ImageBatch
    return ImageBatch(
        image=images,
        gt_label=gt_labels,
        image_path=image_paths,
    )     


def _collate_no_skip_none(batch):
    batch = [(idx, img) for idx, img in batch if img is not None]
    
    if not batch:
        return None, None
        
    idxs, images = zip(*batch)
    return torch.tensor(idxs), torch.stack(images)

def to_pil_rgb(image):
    if image is None:
        return None
    if not isinstance(image, np.ndarray):
        return None
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")

    return None