import numpy as np
import cv2
from utils import display_region_image
from PIL import Image, ImageDraw, ImageFont
from custom_types import ROICollection

class LabelResult:
    def __init__(self, template_type: str,
                 roi_coordinates: ROICollection,
                 region_texts: dict[str,str],
                 region_barcodes: dict[str,str],
                 text_region_images: dict[str,np.ndarray],
                 barcode_images: dict[str,np.ndarray],
                 aligned_image: np.ndarray):
        '''
        Docstring for __init__

        :param template_type: Name of the template(e.g. "151", "146", "063")
        :type template_type: str
        :param roi_coordinates: Dictionary mapping each ROI category to its corresponding ROIs and their coordinates.
        :type roi_coordinates: ROICollection
        :param region_texts: Dictionary mapping each text region name to its extracted text.
        :type region_texts: dict[str, str]
        :param region_barcodes: Dictionary mapping each barcode region name to its extracted barcode value.
        :type region_barcodes: dict[str, str]
        :param text_region_images: Dictionary mapping each text region name to its image.
        :type text_region_images: dict[str, np.ndarray]
        :param barcode_images: Dictionary mapping each barcode region name to its image.
        :type barcode_images: dict[str, np.ndarray]
        :param aligned_image: The aligned full label image.
        :type aligned_image: np.ndarray
        '''
        self.template_type = template_type
        self.region_texts = region_texts
        self.region_barcodes = region_barcodes
        self.aligned_image = aligned_image
        self.roi_coordinates = roi_coordinates
        self._text_region_images = text_region_images
        self._barcode_images = barcode_images

    def get_extracted_texts(self) -> dict[str, str]:
        return self.region_texts
    
    def get_extracted_barcodes(self) -> dict[str, str]:
        return self.region_barcodes

    def display_all_region_images(self):
        for roi_name, image in self._text_region_images.items():
            if roi_name in self.region_texts:
                self.display_region_image(roi_name, image, result=self.region_texts[roi_name])
        for roi_name, image in self._barcode_images.items():
            if roi_name in self.region_barcodes:
                self.display_region_image(roi_name, image, result=self.region_barcodes[roi_name])
        cv2.waitKey(0)
        cv2.destroyAllWindows()