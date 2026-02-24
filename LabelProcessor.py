import cv2
import json
import numpy as np
import os
import pytesseract
import dotenv
import zxingcpp as zxing
from PIL import Image, ImageDraw, ImageFont

from ImageProcessor import ImageProcessor
from LabelResult import LabelResult
from ROIStorage import ROIStorage
from ResultStorage import ResultStorage
import config

class LabelProcessor:
    '''
    Class for e2e processing of label scans.
    Use process_label() to run the full pipeline on a given label scan(pdf or image file).
    After processing, use get_extracted_texts() and get_extracted_barcodes() to retrieve
    results after processing.
    display_all_region_images() can be used to visualize the extracted region images and their OCR results.
    '''
    def __init__(self):
        '''
        Class for e2e processing of label scans.
        '''
        dotenv.load_dotenv()
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")
        
        with open(config.TESSERACT_CFG, 'r') as f:
            self.tesseract_config = json.load(f)
        
        self.image_processor = ImageProcessor()
    
    def process_label(self, scan: str | np.ndarray) -> LabelResult:
        '''
        Performs the full processing pipeline of a label scan. This includes:
        1. using ORB to align the label scan to a template and classify it to a template type. 
        2. getting appropriate ROIs for the label type and extracting region images from the aligned label image.
        3. performing suitable preprocessing on the region images based on the label type and region type (text or barcode).
        4. performing OCR on the text region images and barcode reading on the barcode region images.

        :param scan: Path to the label scan (pdf or image file) or an image numpy array
        :type scan: str | np.ndarray
        :return: Returns LabelResult object containing all extracted information and images from the label.
        :rtype: LabelResult
        '''
        # get image from scan(pdf or image file)
        if isinstance(scan, np.ndarray):
            self.full_label_image = scan
        elif scan.lower().endswith(".pdf"):
            self.full_label_image = self.image_processor.convert_pdf_to_image(scan)
        elif scan.lower().endswith((".jpg", ".jpeg", ".png")):
            self.full_label_image = cv2.imread(scan)
        

        # use orb to align the image and classify it to a template type. This will help us select the suitable image processor and ROIs for the label.
        template_name, self.full_label_image = self.image_processor.orb_align_and_clasify(self.full_label_image)

        self.image_processor = self.image_processor.get_suitable_image_processor(template_name)
        roi_json_path = config.ROI_FILES[template_name]

        roi_storage = ROIStorage(img_h=self.full_label_image.shape[0],
                                      img_w=self.full_label_image.shape[1],
                                      roi_json_path=roi_json_path)
        roi_coordinates = roi_storage.load_roi_json_data()

        self.text_region_images: dict[str, np.ndarray] = self._extract_region_images(roi_coordinates["text_regions"])
        region_texts: dict[str, str] = self._extract_all_region_texts()

        self.barcode_images: dict[str, np.ndarray] = self._extract_region_images(roi_coordinates["barcode_regions"])
        region_barcodes: dict[str, str] = self._extract_all_barcodes()

        return LabelResult(template_name=template_name,
                           region_texts=region_texts,
                            region_barcodes=region_barcodes,
                            text_region_images=self.text_region_images,
                            barcode_images=self.barcode_images)

    def _extract_roi(self, roi_name: str, coordinates: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = coordinates
        image = self.full_label_image[y0:y1, x0:x1]
        preprocess_method = self.image_processor.get_suitable_preprocessing_method(roi_name)
        preprocessed_image = preprocess_method(image)
        return preprocessed_image

    def _extract_text_from_region_image(self, region_image: np.ndarray, config) -> str:
        # Perform OCR using pytesseract
        text = pytesseract.image_to_string(region_image, config=config)
        return text.strip()
    
    def _select_tesseract_config_for_roi(self, roi_name: str) -> str:
        for key in self.tesseract_config.keys():
            if key in roi_name:
                return self.tesseract_config[key]
        return self.tesseract_config["default"]

    def _extract_region_images(self, roi_coordinates) -> dict[str, np.ndarray]:
        region_images = {}
        for roi_name,(x0, y0, x1, y1) in roi_coordinates.items():
            image = self._extract_roi(roi_name, (x0, y0, x1, y1))
            region_images[roi_name] = image
        return region_images
    
    def _extract_all_region_texts(self) -> dict[str, str]:
        assert self.text_region_images, "Text region images have not been extracted."

        region_texts = {}
        for roi_name, img in self.text_region_images.items():
            config = self._select_tesseract_config_for_roi(roi_name)
            region_texts[roi_name] = self._extract_text_from_region_image(img, config=config)
        return region_texts
    
    def _extract_all_barcodes(self) -> dict[str, list]:
        assert self.barcode_images, "Barcode images have not been extracted."

        barcode_results = {}
        for roi_name, img in self.barcode_images.items():
            barcodes = zxing.read_barcodes(img,
                                           formats=zxing.BarcodeFormat.LinearCodes | zxing.BarcodeFormat.DataMatrix,
                                           return_errors=True,
                                           try_rotate=False)
            barcode_results[roi_name] = barcodes[0].text if barcodes else str("")
        return barcode_results


if __name__ == "__main__":
    image_name = "W151.jpg"
    processor = LabelProcessor()
    results = processor.process_label(str(config.IMAGES_DIR / image_name))
    results = ResultStorage(results)
    results.generate_summary(f"W151_unsharp+resized_results", str(config.RESULTS_DIR))