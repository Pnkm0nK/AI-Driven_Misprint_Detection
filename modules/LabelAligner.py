from sklearn import TransformerMixin, BaseEstimator 
import numpy as np
import config
import cv2
import image_processing_functions as ipf

class LabelAligner(TransformerMixin, BaseEstimator):
    def __init__(self,
                 strategy: str = "orb",
                 n_features: int = 100,
                 max_matches: int = 15):
        if strategy not in ["orb", "template"]:
            raise ValueError(f"Invalid alignment strategy: {strategy}. Supported strategies are 'orb' and 'template'.")
        self.strategy = strategy
        self.n_features = n_features
        self.max_matches = max_matches

    def fit(self, X, y=None):
        if self.strategy == "template":
            self._templates = config.TEMPLATES.items()
        else:
            self._templates = config.LOGO_TEMPLATES.items()
    
        for template_type, template_img_path in self._templates:
            if not template_img_path.exists():
                raise FileNotFoundError(f"Template image not found at {template_img_path}")

            self._templates[template_type] = cv2.imread(str(template_img_path), cv2.IMREAD_GRAYSCALE) 

        return self

    def transform(self, X):
        aligned_images = np.array([])

        for image in X:
            if self.strategy == "template":
                aligned = ipf.align_image(image, self.template_img_path)
            elif self.strategy == "orb":
                aligned = ipf.orb_align(image, self.template_type, self.n_features, self.max_matches)
                aligned_images.append(aligned)
        return aligned_images
    