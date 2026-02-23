import os
import cv2
import dotenv
import pdf2image
from deskew import determine_skew
from utils import get_template_matching_results, convert_to_greyscale
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
    
    def convert_multipage_pdf_to_image(self, pdf_path: str, dpi: int = 300) -> np.ndarray:
        images = pdf2image.convert_from_path(pdf_path= pdf_path, dpi=dpi,
                                            poppler_path=self.poppler_path)
        cv_images = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]
        return cv_images
    
    def deskew_image(self, image:np.ndarray) -> np.ndarray:
        skew_angle = determine_skew(image, max_angle=30)
        rot_mat = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), skew_angle, 1)
        image = cv2.warpAffine(image, rot_mat, (image.shape[1],image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        return image
    
    def align_image(self, image: np.ndarray, template_image_path: str, padding=20) -> np.ndarray:
        '''
        Align the input image to the template image using deskewing and template matching. Returns the aligned image.
        '''
        deskewed = self.deskew_image(image)

        _, loc = get_template_matching_results(deskewed, template_image_path)

        x_start, y_start = loc 
        x_start -= padding
        if x_start < 0:
            deskewed = cv2.copyMakeBorder(deskewed, padding-x_start, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
            x_start = 0
        return deskewed[y_start:, x_start:]
    
    def unsharp(self, image: np.ndarray, kernel_size=(1,1), sigma=2, amount=2.0, threshold=0) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, threshold)
        return sharpened

    def preprocess_image_general(self, image: np.ndarray) -> np.ndarray:
        # General preprocessing: convert to grayscale 
        gray = convert_to_greyscale(image) 
        resized = cv2.resize(gray, (0,0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC) 
        unsharp_image = self.unsharp(resized)
        
        # padded = cv2.copyMakeBorder(unsharp_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255,255,255])
        # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return unsharp_image
    
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

class Type146ImageProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()

class Type063ImageProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()
    def ensure_correct_orientation(self, image, template_image_path):
        height, width = image.shape[:2]
        if height > width:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
        min_val_trial_1, loc = get_template_matching_results(image, template_image_path)
        upside_down_image = cv2.rotate(image, cv2.ROTATE_180)
        min_val_trial_2, loc = get_template_matching_results(upside_down_image, template_image_path)
        if min_val_trial_1 < min_val_trial_2:
            return image

        return upside_down_image

    def preprocess_large_label(self, image):
        return self.preprocess_image_general(image)
        
if __name__ == "__main__": 
    processor = ImageProcessor()
    # images = processor.convert_multipage_pdf_to_image("../label_scans/0400063.pdf")
    # for i, img in enumerate(images):
    #     cv2.imwrite(f"./images/063_page_{i}.jpg", img)
