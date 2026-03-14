import cv2
import json
import numpy as np
import os
import pytesseract
import dotenv
from ultralytics import YOLO
import zxingcpp as zxing
import time
from modules.PDFConverter import PDFConverter

from modules.ImageProcessor import ImageProcessor
from modules.LabelResult import LabelResult
from modules.ROIStorage import ROIStorage
from custom_types import ROICollection
import config

class LabelProcessor:
    '''
    Class for e2e processing of label scans.
    Use process_label() to run the full pipeline on a given label scan(pdf or image file).
    After processing, use get_extracted_texts() and get_extracted_barcodes() to retrieve
    import PDFConverter
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
        self.label_classifier = YOLO("label_classifier.pt")
        self.run_times = {}
    
    def process_label(self, scan: str | np.ndarray, template_type: str = None) -> LabelResult:
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
        start_time = time.time()
        self.full_label_image = self._handle_scan_file(scan)
        self.run_times["Image loading"] = time.time() - start_time

        # use orb to align the image and classify it to a template type. This will help us select the suitable image processor and ROIs for the label.
        if template_type is None:
            start_time = time.time()
            classifier_results = self.label_classifier(self.full_label_image)
            most_probable_class = classifier_results[0].probs.top1
            template_type = classifier_results[0].names[most_probable_class]
            self.run_times["YOLO classification"] = time.time() - start_time

        start_time = time.time()
        self.full_label_image =self.image_processor.orb_align(self.full_label_image, template_type, n_features=200, max_matches=30)
        self.run_times["ORB alignment"] = time.time() - start_time

        # specialize image processor to the template
        self.image_processor = self.image_processor.get_suitable_image_processor(template_type)

        start_time = time.time()
        roi_storage = ROIStorage(img_h=self.full_label_image.shape[0],
                                      img_w=self.full_label_image.shape[1],
                                      template_type=template_type)
        roi_coordinates = roi_storage.load_roi_json_data()
        self.run_times["ROI loading"] = time.time() - start_time

        start_time = time.time()
        self.text_region_images: dict[str, np.ndarray] = self._extract_preprocessed_region_images(roi_coordinates["text_regions"])
        self.run_times["Text region extraction"] = time.time() - start_time
        
        start_time = time.time()
        region_texts: dict[str, str] = self._extract_all_region_texts()
        self.run_times["OCR text extraction"] = time.time() - start_time

        start_time = time.time()
        self.barcode_images: dict[str, np.ndarray] = self._extract_preprocessed_region_images(roi_coordinates["barcode_regions"])
        self.run_times["Barcode region extraction"] = time.time() - start_time
        
        start_time = time.time()
        region_barcodes: dict[str, str] = self._extract_all_barcodes()
        self.run_times["Barcode reading"] = time.time() - start_time

        # do not apply preprocessing to symbol regions, as here preprocessing is
        # specialized for differencing
        start_time = time.time()
        symbol_images = self._extract_region_images(roi_coordinates["symbol_regions"])
        self.run_times["Symbol region extraction"] = time.time() - start_time


        return LabelResult(template_type=template_type,
                           roi_coordinates=roi_coordinates,
                           region_texts=region_texts,
                           region_barcodes=region_barcodes,
                           text_region_images=self.text_region_images,
                           barcode_images=self.barcode_images,
                           symbol_images=symbol_images,
                           aligned_image=self.full_label_image,
                           run_times=self.run_times)
        
    def _postprocess_text(self, text: str) -> str:
        '''
        General postprocessing of extracted text to normalize text format 
        '''
        text = text.replace("\n", " ").strip()
        return text
    
    def _handle_scan_file(self, scan: str | np.ndarray) -> np.ndarray:
        '''Handles the input scan file, which can be a path to a pdf or image file,
           or an already loaded image as a numpy array. It returns the image as a numpy array for further processing.

           :param scan: Path to the label scan (pdf or image file) or an image numpy array
           :type scan: str | np.ndarray
        '''
        converter = PDFConverter()
        if isinstance(scan, np.ndarray):
            return scan
        elif scan.lower().endswith(".pdf"):
            return converter.convert_pdf_to_image(scan)
        elif scan.lower().endswith((".jpg", ".jpeg", ".png")):
            return cv2.imread(scan)

        raise ValueError(f"Unsupported scan file type: {scan}")


    def _remove_variable_info_from_image(self, image:np.ndarray, roi_coordinates: ROICollection ) -> np.ndarray:
        '''
        Removes variable information from the image that increases differnce between template and alligned image.
        Preprocessing step to find defects by differencing template and aligned image.
        '''
        cleaned_image = image.copy()
        for roi_type in roi_coordinates:
            for roi in roi_coordinates[roi_type].values():
                x0, y0, x1, y1 = roi
                cleaned_image[y0:y1, x0:x1] = 255
        return cleaned_image
    
    def _extract_text_from_region_image(self, region_image: np.ndarray, config) -> str:
        # Perform OCR using pytesseract
        text = pytesseract.image_to_string(region_image, config=config)
        return self._postprocess_text(text)
    
    def _select_tesseract_config_for_roi(self, roi_name: str) -> str:
        for key in self.tesseract_config.keys():
            if key in roi_name:
                return self.tesseract_config[key]
        return self.tesseract_config["default"]

    def _extract_preprocessed_region_images(self, roi_coordinates) -> dict[str, np.ndarray]:
        region_images = {}
        for roi_name, coords in roi_coordinates.items():
            image = self.image_processor.extract_roi(self.full_label_image, coords)
            image = self.image_processor.preprocess_region_image(roi_name, image)
            region_images[roi_name] = image
        return region_images
    
    def _extract_region_images(self, roi_coordinates) -> dict[str, np.ndarray]:
        region_images = {}
        for roi_name, coords in roi_coordinates.items():
            image = self.image_processor.extract_roi(self.full_label_image, coords)
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