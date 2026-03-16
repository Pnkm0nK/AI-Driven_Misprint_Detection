from sklearn.base import TransformerMixin, BaseEstimator 
import numpy as np
from tqdm import tqdm
import cv2
import modules.image_processing_functions as ipf

class LabelNormalizer(TransformerMixin, BaseEstimator):
    def __init__(self,
                 template_type: str = "151",
                 downsample_factor: float = 0.3,
                 strategy: str = "orb",
                 n_features: int = 100,
                 max_matches: int = 15):
        if strategy not in ["orb", "template"]:
            raise ValueError(f"Invalid alignment strategy: {strategy}. Supported strategies are 'orb' and 'template'.")

        self.template_type = template_type
        self.downsample_factor = downsample_factor
        self.strategy = strategy
        self.n_features = n_features
        self.max_matches = max_matches

    def fit(self, X, y=None):
        # if self.strategy == "orb":
        #     self._templates = config.TEMPLATES
        # elif self.strategy == "template":
        #     self._templates = config.LOGO_TEMPLATES
    
        # for template_type, template_img_path in self._templates.items():
        #     if not template_img_path.exists():
        #         raise FileNotFoundError(f"Template image not found at {template_img_path}")

        #     self._templates[template_type] = cv2.imread(str(template_img_path), cv2.IMREAD_GRAYSCALE) 

        return self

    def transform(self, X):
        aligned_images = []

        for image in tqdm(X, desc=f"Normalizing labels using {self.strategy} alignment"):
            if self.strategy == "template":
                aligned = ipf.align_image(image, self.template_img_path)
            elif self.strategy == "orb":
                aligned = ipf.orb_align(image, self.template_type, self.n_features, self.max_matches)
            

            aligned = ipf.convert_to_greyscale(aligned)
            aligned = cv2.resize(aligned, (0,0), fx=self.downsample_factor, fy=self.downsample_factor)

            aligned_images.append(aligned)
        return np.array(aligned_images)