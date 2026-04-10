import torch
from tqdm import tqdm
import numpy as np
import timm
import faiss
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
import cv2
from anomaly_detection.ImageDataset import ImageDataset, _collate_no_skip_none

class AnomalyDinoEstimator(BaseEstimator):
    def __init__(self, backbone='vit_small_patch14_dinov2',
                  batch_size=16,
                  threshold_percentile=95,
                  n_neighbors=5,

                  half_precision=False):
        self._threshold = None
        self._last_key = None
        self._last_scores = None
        self.backbone = backbone
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.n_neighbors = n_neighbors
        self.half_precision = half_precision

        self.model = None
        self.patch_index = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _cache_key(self, X):
        # create a cache key based on the id and length of data to avoid recomputing scores
        return (id(X), len(X))

    def _extract_features(self, batch_tensor):
        with torch.inference_mode():
            batch_tensor = batch_tensor.to(self.device) 
            if self.half_precision:
                batch_tensor = batch_tensor.half()

            tokens = self.model.get_intermediate_layers(batch_tensor, n=1)[0]
            tokens = tokens.cpu().numpy()  
            self.is_fitted_ = True
        return tokens


    def _get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens

    def _compute_background_mask(self, img_features, grid_size, threshold = 10, kernel_size = 3, border = 0.2):
        # Kernel size for morphological operations should be odd
        pca = PCA(n_components=1, svd_solver='randomized')
        first_pc = pca.fit_transform(img_features.astype(np.float32))
        mask = first_pc > threshold
        # test whether the center crop of the images is kept (adaptive masking), adapt if your objects of interest are not centered!
        m = mask.reshape(grid_size)[int(grid_size[0] * border):int(grid_size[0] * (1-border)), int(grid_size[1] * border):int(grid_size[1] * (1-border))]
        if m.sum() <=  m.size * 0.35:
            mask = - first_pc > threshold
        # postprocess mask, fill small holes in the mask, enlarge slightly
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        return mask.squeeze()
    
    def _calculate_batch_scores(self, batch_patches):
        '''Calculate anomaly scores for a batch of images with a single FAISS search.'''
        if len(batch_patches) == 0:
            return []

        patch_counts = []
        valid_patch_groups = []
        for patches in batch_patches:
            if patches.shape[0] == 0:
                patch_counts.append(0)
            else:
                patch_counts.append(patches.shape[0])
                valid_patch_groups.append(patches.astype('float32'))

        if len(valid_patch_groups) == 0:
            return [0.0] * len(batch_patches)

        all_patches = np.vstack(valid_patch_groups)
        dists, _ = self.patch_index.search(all_patches, self.n_neighbors)
        patch_distances = np.sqrt(np.maximum(dists, 0)).mean(axis=1)

        scores = []
        ptr = 0
        for count in patch_counts:
            if count == 0:
                scores.append(0.0)
                continue
            img_patch_distances = patch_distances[ptr:ptr + count]
            ptr += count
            scores.append(float(np.percentile(img_patch_distances, 99)))

        return scores


    def fit(self, X, y=None):
        self.model = timm.create_model(self.backbone, pretrained=True).to(self.device)
        config = timm.data.resolve_data_config({}, model=self.model)
        res = config['input_size'][1]
        self.grid_size = (res // 14, res // 14)
        self.transform = timm.data.create_transform(**config)
        self.model.eval()

        dataset = ImageDataset(X, preprocess=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=self.batch_size, 
                                                 shuffle=False, 
                                                 collate_fn=_collate_no_skip_none)
        
        img_masked_patches = [] 

        for _, batch in tqdm(dataloader, desc="Extracting features"):
            tokens_batch = self._extract_features(batch) # [B, N_patches, Dim]
            for img_tokens in tokens_batch:
                mask = self._compute_background_mask(img_tokens, self.grid_size)
                active_tokens = img_tokens[mask.ravel()]
                if active_tokens.shape[0] == 0:
                    active_tokens = img_tokens
                img_masked_patches.append(active_tokens)

        # build index
        dim = img_masked_patches[0].shape[1]
        faiss.omp_set_num_threads(15)
        self.patch_index = faiss.IndexHNSWFlat(dim, 16)

        self.patch_index.add(np.vstack(img_masked_patches).astype('float32'))


        # threshold estimation
        train_scores = []
        for start_idx in tqdm(range(0, len(img_masked_patches), self.batch_size), desc="Calculating training scores"):
            batch_patches = img_masked_patches[start_idx:start_idx + self.batch_size]
            train_scores.extend(self._calculate_batch_scores(batch_patches))
        self._last_key = self._cache_key(X)
        self._last_scores = np.array(train_scores)
        self._threshold = np.percentile(train_scores, self.threshold_percentile)
        self.is_fitted_ = True
        
        return self

    def score_samples(self, X):
        if self._last_key == self._cache_key(X) and self._last_scores is not None:
            return self._last_scores

        dataset = ImageDataset(X, preprocess=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=self.batch_size, 
                                                 shuffle=False, 
                                                 collate_fn=_collate_no_skip_none)
        scores = []

        for _, batch in tqdm(dataloader, desc="Scoring"):
            tokens_batch = self._extract_features(batch) # [B, N, Dim]

            batch_active_tokens = []
            for i in range(tokens_batch.shape[0]):
                img_tokens = tokens_batch[i]
                mask = self._compute_background_mask(img_tokens, self.grid_size)
                active_tokens = img_tokens[mask.ravel()]
                
                if active_tokens.shape[0] == 0:
                    active_tokens = img_tokens

                batch_active_tokens.append(active_tokens)

            scores.extend(self._calculate_batch_scores(batch_active_tokens))

        self._last_key = self._cache_key(X)
        self._last_scores = np.array(scores)
        return self._last_scores

    def predict(self, X):
        scores = self.score_samples(X)
        return (scores > self._threshold).astype(int) 