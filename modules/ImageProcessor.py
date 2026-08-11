import cv2
import modules.image_processing_functions as ipf
import numpy as np

class ImageProcessor():
    def __init__(self):
        pass

    def preprocess_image_general(self, image: np.ndarray, barcode=False) -> np.ndarray:
        img = ipf.convert_to_greyscale(image) 
        
        if barcode:
            img = cv2.resize(img, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        else:
            if img.shape[0] < 100:
                scale = 100 / img.shape[0]
                new_width = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_width, 100), interpolation=cv2.INTER_CUBIC)
        
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])
        return img

    # --- Text Label Methods ---
    def preprocess_top_label(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_image_general(image)

    def preprocess_large_label(self, image: np.ndarray) -> np.ndarray:
        turned = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return self.preprocess_image_general(turned)

    def preprocess_small_label(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_image_general(image)

    def preprocess_patient_label(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_image_general(image)

    # --- Barcode Specific Methods ---
    def preprocess_top_barcode(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_image_general(image, barcode=True)

    def preprocess_large_barcode(self, image: np.ndarray) -> np.ndarray:
        turned = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return self.preprocess_image_general(turned, barcode=True)

    def preprocess_small_barcode(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_image_general(image, barcode=True)

    def preprocess_patient_barcode(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_image_general(image, barcode=True)

    def get_suitable_preprocessing_method(self, roi_name: str):
        roi_name = roi_name.lower()
        
        # Picking logic for Barcodes
        if "code128" in roi_name or "ecc200" in roi_name:
            if "top" in roi_name:
                return self.preprocess_top_barcode
            elif "large" in roi_name:
                return self.preprocess_large_barcode
            elif "small" in roi_name:
                return self.preprocess_small_barcode
            elif "patient" in roi_name:
                return self.preprocess_patient_barcode
            return lambda img: self.preprocess_image_general(img, barcode=True)

        # Picking logic for Labels
        if "top" in roi_name:
            return self.preprocess_top_label
        elif "large" in roi_name:
            return self.preprocess_large_label
        elif "small" in roi_name:
            return self.preprocess_small_label
        elif "patient" in roi_name:
            return self.preprocess_patient_label
        
        return self.preprocess_image_general
    
    def preprocess_region_image(self, roi_name: str, image: np.ndarray) -> np.ndarray:
        '''
        Applies suitable preprocessing to the input region image based on the region type

        :param roi_name: Name of the region of interest, used to determine the suitable preprocessing method 
        :type roi_name: str
        :param image: The region image to be preprocessed
        :type image: np.ndarray
        :return: The preprocessed region image
        :rtype: ndarray[_AnyShape, dtype[Any]]
        '''
        preprocess_method = self.get_suitable_preprocessing_method(roi_name)
        preprocessed_image = preprocess_method(image)
        return preprocessed_image

    @staticmethod
    def get_suitable_image_processor(template_type: str) -> "ImageProcessor":
        '''Returns an instance of the suitable ImageProcessor subclass based on the template name.
        If no specific processor is found for the template, returns a default ImageProcessor
        instance.

        :param template_type: Name of the template
        :type template_type: str
        '''
        template_type = template_type.lower()[0:3]

        cls = PROCESSOR_CLASSES.get(template_type, ImageProcessor)
        return cls() 
    
class Type151ImageProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()
    def preprocess_small_label(self, image: np.ndarray) -> np.ndarray:
        image = cv2.rotate(image, cv2.ROTATE_180)
        return self.preprocess_image_general(image)
    def preprocess_patient_label(self, image):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return super().preprocess_image_general(image)
    def preprocess_small_barcode(self, image):
        image = cv2.rotate(image, cv2.ROTATE_180)
        return super().preprocess_image_general(image, barcode=True)
    def preprocess_patient_barcode(self, image):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return super().preprocess_image_general(image, barcode=True)

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
            
        min_val_trial_1, loc = ipf.get_template_matching_results(image, template_image_path)
        upside_down_image = cv2.rotate(image, cv2.ROTATE_180)
        min_val_trial_2, loc = ipf.get_template_matching_results(upside_down_image, template_image_path)
        if min_val_trial_1 < min_val_trial_2:
            return image

        return upside_down_image

    def preprocess_large_label(self, image):
        return self.preprocess_image_general(image)


PROCESSOR_CLASSES: dict[str, type[ImageProcessor]] = {
    "151": Type151ImageProcessor,
    "146": Type146ImageProcessor,
    "063": Type063ImageProcessor,
}