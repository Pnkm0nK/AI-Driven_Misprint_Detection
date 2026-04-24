import cv2
import joblib
import numpy as np
import os
import dotenv
from typing import Optional
import sklearn
from ultralytics import YOLO
import zxingcpp as zxing
import tesserocr
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from modules.PDFConverter import PDFConverter
from modules.ImageProcessor import ImageProcessor
from modules.LabelResult import LabelResult
from modules.ROIStorage import ROIStorage
import modules.image_processing_functions as ipf
from utilities.custom_types import ROICollection
from utilities.tesserocr_config import tesserocr_config
import utilities.config as config

class PipelineState:
    def __init__(self):
        self.image: Optional[np.ndarray] = None
        self.aligned_image: Optional[np.ndarray] = None
        self.template_type: str = None
        self.roi_coordinates: dict = {}
        self.text_region_images: dict = {}
        self.barcode_images: dict = {}
        self.region_texts: dict = {}
        self.region_barcodes: dict = {}
        self.is_anomaly: Optional[bool] = None
        self.run_times: dict = {}

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
        self.state: Optional[PipelineState] = None
        
        self.label_classifier = YOLO(config.LABEL_CLASSIFIER_MODEL_PATH)
        self.anomaly_detection_model = self._load_anomaly_detection_model(config.ANOMALY_DETECTION_MODEL_PATH)
    
    def process_label(self, scan: str | np.ndarray,
                      template_type: str = None,
                      run_ocr: bool = True,
                      run_anomaly_detection: bool = True,
                      postprocess_text: bool = True,
                      ocr_workers: int = 8) -> LabelResult:
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
        state = PipelineState()
        self.state = state

        # get image from scan(pdf or image file)
        start_time = time.time()
        state.image = self._handle_scan_file(scan)
        state.run_times["Image loading"] = time.time() - start_time

        # use orb to align the image and classify it to a template type. This will help us select the suitable image processor and ROIs for the label.
        if template_type is None:
            start_time = time.time()
            classifier_results = self.label_classifier(state.image)
            most_probable_class = classifier_results[0].probs.top1
            template_type = classifier_results[0].names[most_probable_class]
            state.run_times["YOLO classification"] = time.time() - start_time

        state.template_type = template_type

        start_time = time.time()
        state.aligned_image = ipf.orb_align(state.image, state.template_type, n_features=250, max_matches=50)
        state.run_times["ORB alignment"] = time.time() - start_time

        # specialize image processor to the template
        image_processor = ImageProcessor.get_suitable_image_processor(state.template_type)

        start_time = time.time()
        roi_storage = ROIStorage(img_h=state.aligned_image.shape[0],
                                 img_w=state.aligned_image.shape[1],
                                 template_type=state.template_type)
        state.roi_coordinates = roi_storage.load_roi_json_data()
        state.run_times["ROI loading"] = time.time() - start_time

        start_time = time.time()
        state.text_region_images = self._extract_preprocessed_region_images(
            state.aligned_image,
            state.roi_coordinates["text_regions"],
            image_processor,
        )
        state.run_times["Text region extraction"] = time.time() - start_time
        
        if run_ocr:
            start_time = time.time()
            state.region_texts = self._extract_all_region_text_parallel(
                state.text_region_images,
                postprocess=postprocess_text,
                max_workers=ocr_workers,
            )
            state.run_times["OCR text extraction"] = time.time() - start_time

        start_time = time.time()
        state.barcode_images = self._extract_preprocessed_region_images(
            state.aligned_image,
            state.roi_coordinates["barcode_regions"],
            image_processor,
        )
        state.run_times["Barcode region extraction"] = time.time() - start_time
        
        start_time = time.time()
        state.region_barcodes = self._extract_all_barcodes(state.barcode_images)
        state.run_times["Barcode reading"] = time.time() - start_time

        if run_anomaly_detection:
            start_time = time.time()
            normalized_image = ipf.convert_to_greyscale(state.aligned_image)
            is_anomaly: np.ndarray = self._perform_anomaly_detection(np.array([normalized_image]), binary=True)
            state.is_anomaly = bool(is_anomaly[0])
            state.run_times["Anomaly detection"] = time.time() - start_time

        # do not apply preprocessing to symbol regions, as here preprocessing is
        # specialized for differencing
        # start_time = time.time()
        # symbol_images = self._extract_region_images(state.aligned_image, state.roi_coordinates["symbol_regions"])
        # state.run_times["Symbol region extraction"] = time.time() - start_time


        return LabelResult(template_type=state.template_type,
                           roi_coordinates=state.roi_coordinates,
                           region_texts=state.region_texts,
                           region_barcodes=state.region_barcodes,
                           is_anomaly=state.is_anomaly,
                           text_region_images=state.text_region_images,
                           barcode_images=state.barcode_images,
                           aligned_image=state.aligned_image,
                           run_times=state.run_times)
    
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
        text = text.replace("\n", " ").strip().lower()
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
    
    def _extract_preprocessed_region_images(self,
                                            source_image: np.ndarray,
                                            roi_coordinates,
                                            image_processor: ImageProcessor) -> dict[str, np.ndarray]:
        region_images = {}
        for roi_name, coords in roi_coordinates.items():
            image = ipf.extract_roi(source_image, coords)
            image = image_processor.preprocess_region_image(roi_name, image)
            region_images[roi_name] = image
        return region_images
    
    def _extract_region_images(self, source_image: np.ndarray, roi_coordinates) -> dict[str, np.ndarray]:
        region_images = {}
        for roi_name, coords in roi_coordinates.items():
            image = ipf.extract_roi(source_image, coords)
            region_images[roi_name] = image
        return region_images
    
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
        return roi_name, text

    def _extract_all_region_text_parallel(self,
                                          text_region_images: dict[str, np.ndarray],
                                          postprocess: bool = True,
                                          max_workers: int = 8) -> dict[str, str]:
        assert text_region_images, "Text region images have not been extracted."

        tasks = list(text_region_images.items())

        with ThreadPoolExecutor(max_workers=max_workers, initializer=self._init_thread_apis) as executor:
            results = list(executor.map(self._ocr_task, tasks))
        
        if postprocess:
            results = {roi_name: self._postprocess_text(text) for roi_name, text in results}
        else:
            results = dict(results)
        return results


    def _extract_all_barcodes(self, barcode_images: dict[str, np.ndarray]) -> dict[str, list]:
        assert barcode_images, "Barcode images have not been extracted."

        barcode_results = {}
        for roi_name, img in barcode_images.items():
            barcodes = zxing.read_barcodes(img,
                                           formats=zxing.BarcodeFormat.LinearCodes | zxing.BarcodeFormat.DataMatrix,
                                           return_errors=True,
                                           try_rotate=False)
            barcode_results[roi_name] = barcodes[0].text if barcodes else str("")
        return barcode_results