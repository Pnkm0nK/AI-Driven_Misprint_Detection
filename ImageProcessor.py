import os
import cv2
import dotenv
import pdf2image
import numpy as np

class ImageProcessor():
    def __init__(self):
        dotenv.load_dotenv()
        self.poppler_path = os.getenv("POPPLER_PATH")

    def convert_pdf_to_image(self, pdf_path: str, dpi: int = 300)-> np.ndarray:
        # used only for single-page PDFs
        images = pdf2image.convert_from_path(pdf_path= pdf_path, dpi=dpi,
                                            poppler_path=self.poppler_path)
        return cv2.cvtColor(
            np.array(images[0]), cv2.COLOR_RGB2BGR
        ) 

    def preprocess_image_general(self, image: np.ndarray) -> np.ndarray:
        # General preprocessing: convert to grayscale and apply binary thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return gray
    
    def preprocess_top_label(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_image_general(image)
    
    def preprocess_large_label(self, image: np.ndarray) -> np.ndarray:
        # Rotate counterclockwise and do general preprocessing
        turned = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return self.preprocess_image_general(turned)
    
    def preprocess_small_label(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_image_general(image)
    
    def preprocess_patient_label(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_image_general(image)
    
    def get_suitable_preprocessing_method(self, roi_name: str):
        if "top" in roi_name:
            return self.preprocess_top_label
        elif "large" in roi_name:
            return self.preprocess_large_label
        elif "small" in roi_name:
            return self.preprocess_small_label
        elif "patient" in roi_name:
            return self.preprocess_patient_label
        else:
            return self.preprocess_image_general

class Type151ImageProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()
    def preprocess_small_label(self, image: np.ndarray) -> np.ndarray:
        image = cv2.rotate(image, cv2.ROTATE_180)
        return self.preprocess_image_general(image)
    
        