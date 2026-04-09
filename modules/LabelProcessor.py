import cv2
import json
import joblib
import numpy as np
import os
import dotenv
import sklearn
from ultralytics import YOLO
import zxingcpp as zxing
import tesserocr
import threading
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import time

from modules.PDFConverter import PDFConverter
from modules.ImageProcessor import ImageProcessor
from modules.LabelResult import LabelResult
from modules.ROIStorage import ROIStorage
import modules.image_processing_functions as ipf
from utilities.custom_types import ROICollection
from utilities.tesserocr_config import tesserocr_config
import utilities.config as config

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

        self.tessdata_dir = os.getenv("TESSDATA_DIR")
        self.tesserocr_config = tesserocr_config
        self.thread_local = threading.local()
        
        self.image_processor = ImageProcessor()
        self.label_classifier = YOLO(config.LABEL_CLASSIFIER_MODEL_PATH)
        self.anomaly_detection_model = self._load_anomaly_detection_model(config.ANOMALY_DETECTION_MODEL_PATH)
        self.run_times = {}
    
    def process_label(self, scan: str | np.ndarray, template_type: str = None) -> LabelResult:
        '''
        Performs the full processing pipeline of a label scan. This includes:
        1. using ORB to align the label scan to a template and classify it to a template type. 
        2. getting appropriate ROIs for the label type and extracting region images from the aligned label image.
        3. performing suitable preprocessing on the region images based on the label type and region type (text or barcode).
        4. performing OCR on the text region images and barcode reading on the barcode region images.
        5. perform anomaly detection on label image using the loaded anomaly detection model

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
        self.full_label_image =ipf.orb_align(self.full_label_image, template_type, n_features=250, max_matches=50)
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
        region_texts: dict[str, str] = self._extract_all_region_text_parallel()
        self.run_times["OCR text extraction"] = time.time() - start_time

        start_time = time.time()
        self.barcode_images: dict[str, np.ndarray] = self._extract_preprocessed_region_images(roi_coordinates["barcode_regions"])
        self.run_times["Barcode region extraction"] = time.time() - start_time
        
        start_time = time.time()
        region_barcodes: dict[str, str] = self._extract_all_barcodes()
        self.run_times["Barcode reading"] = time.time() - start_time

        start_time = time.time()
        normalized_image = ipf.convert_to_greyscale(self.full_label_image)
        is_anomaly: np.ndarray = self._perform_anomaly_detection(np.array([normalized_image]), binary=True)
        is_anomaly = bool(is_anomaly[0])
        self.run_times["Anomaly detection"] = time.time() - start_time

        # do not apply preprocessing to symbol regions, as here preprocessing is
        # specialized for differencing
        # start_time = time.time()
        # symbol_images = self._extract_region_images(roi_coordinates["symbol_regions"])
        # self.run_times["Symbol region extraction"] = time.time() - start_time


        return LabelResult(template_type=template_type,
                           roi_coordinates=roi_coordinates,
                           region_texts=region_texts,
                           region_barcodes=region_barcodes,
                           is_anomaly=is_anomaly,
                           text_region_images=self.text_region_images,
                           barcode_images=self.barcode_images,
                           aligned_image=self.full_label_image,
                           run_times=self.run_times)
    
    def _load_anomaly_detection_model(self, model_path: str):
        '''
        Loads the panomaly detection model to be used for symbol region differencing.
        '''
        anomaly_model: sklearn.pipeline.Pipeline = joblib.load(model_path)
        # disable normalizer as label is normalized as part of the whole label processing pipeline 
        anomaly_model.set_params(normalizer="passthrough")
        return anomaly_model

    def _perform_anomaly_detection(self, normalized_image, binary=True):
        '''
        Performs anomaly detection on the normalized (greyscaled, aligned)
        label image using the loaded anomaly detection model.
        '''
        if not hasattr(self, "anomaly_detection_model"):
            self.anomaly_detection_model = self._load_anomaly_detection_model(config.ANOMALY_DETECTION_MODEL_PATH)
        if binary:
            return self.anomaly_detection_model.predict(normalized_image)
        else:
            return self.anomaly_detection_model.score_samples(normalized_image)


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
    
    def _extract_preprocessed_region_images(self, roi_coordinates) -> dict[str, np.ndarray]:
        region_images = {}
        for roi_name, coords in roi_coordinates.items():
            image = ipf.extract_roi(self.full_label_image, coords)
            image = self.image_processor.preprocess_region_image(roi_name, image)
            region_images[roi_name] = image
        return region_images
    
    def _extract_region_images(self, roi_coordinates) -> dict[str, np.ndarray]:
        region_images = {}
        for roi_name, coords in roi_coordinates.items():
            image = ipf.extract_roi(self.full_label_image, coords)
            region_images[roi_name] = image
        return region_images
    
    def _extract_all_region_texts(self) -> dict[str, str]:
        assert self.text_region_images, "Text region images have not been extracted."

        region_texts = {}
        for roi_name, img in self.text_region_images.items():
            img = Image.fromarray(img)
            config = self.tesserocr_config[self._get_suitable_config_key(roi_name)]
            with tesserocr.PyTessBaseAPI(path=self.tessdata_dir,lang=config["lang"], oem=config["oem"]) as api:
                api.SetPageSegMode(config["psm"])
                for k, v in config["vars"].items():
                    api.SetVariable(k, v)
                
                api.SetImage(img)
                region_texts[roi_name] = self._postprocess_text(api.GetUTF8Text().strip())
        return region_texts
    
    def _get_suitable_config_key(self, roi_name: str) -> str:
        for key in self.tesserocr_config.keys():
            if key in roi_name:
                return key
        return "default"
    
    def _init_thread_apis(self):
        """Initializes a tesserocr API instance for each thread in the ThreadPoolExecutor and stores them in thread-local storage to not reinitialize possibly saving time"""
        if not hasattr(self.thread_local, "api_cache"):
            self.thread_local.api_cache = {}
            for name, cfg in self.tesserocr_config.items():
                api = tesserocr.PyTessBaseAPI(path=self.tessdata_dir, lang=cfg["lang"], oem=cfg["oem"])
                api.SetPageSegMode(cfg["psm"])
                for k, v in cfg["vars"].items():
                    api.SetVariable(k, v)
                self.thread_local.api_cache[name] = api

    def _ocr_task(self, item):
        roi_name, img_array = item
        cfg_key = self._get_suitable_config_key(roi_name)
        
        cfg_key = cfg_key if cfg_key in self.thread_local.api_cache else "default"
        
        api = self.thread_local.api_cache[cfg_key]
        h, w  = img_array.shape
        
        api.SetImageBytes(img_array.tobytes(), width=w, height=h, bytes_per_pixel=1, bytes_per_line=w)
        text = api.GetUTF8Text().strip()
        api.Clear()  
        
        return roi_name, self._postprocess_text(text)

    def _extract_all_region_text_parallel(self):
        assert self.text_region_images, "Text region images have not been extracted."

        tasks = list(self.text_region_images.items())

        with ThreadPoolExecutor(max_workers=8, initializer=self._init_thread_apis) as executor:
            results = list(executor.map(self._ocr_task, tasks))

        return dict(results)


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