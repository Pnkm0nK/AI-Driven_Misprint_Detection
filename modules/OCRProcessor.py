from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import os
import dotenv
from paddleocr import PaddleOCRVL
import numpy as np
import tesserocr
import easyocr

from utilities.tesserocr_config import tesserocr_config

class OCRProcessor(ABC):
    def __init__(self):
        pass

    @staticmethod
    def _postprocess_text(text: str) -> str:
        '''
        General postprocessing of extracted text to normalize text format 
        '''
        text = text.replace("\n", " ").strip().lower()
        return text

    @abstractmethod
    def extract_all_region_text_parallel(text_region_images: dict[str, np.ndarray],
                                         postprocess: bool):
        pass

class TesserocrProcessor(OCRProcessor):
    def __init__(self, max_workers: int = 8):
        super().__init__()
        self.max_workers = max_workers
        dotenv.load_dotenv()

        self.thread_local = threading.local()
    
        self.tessdata_dir = os.getenv("TESSDATA_DIR")
        self.tesserocr_config = tesserocr_config

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

    def _get_suitable_config_key(self, roi_name: str) -> str:
        for key in self.tesserocr_config.keys():
            if key in roi_name:
                return key
        return "default"
    
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
    
    def extract_all_region_text_parallel(self,
                                          text_region_images: dict[str, np.ndarray],
                                          postprocess: bool = True) -> dict[str, str]:
        assert text_region_images, "Text region images have not been extracted."

        tasks = list(text_region_images.items())

        with ThreadPoolExecutor(max_workers=self.max_workers, initializer=self._init_thread_apis) as executor:
            results = list(executor.map(self._ocr_task, tasks))
        
        if postprocess:
            results = {roi_name: self._postprocess_text(text) for roi_name, text in results}
        else:
            results = dict(results)
        return results

class EasyOCRProcessor(OCRProcessor):
    def __init__(self, max_workers: int = 8, batch_size: int = 32):
        super().__init__()
        self.tesserocr_config = tesserocr_config
        self.max_workers = max_workers
        self.batch_size = batch_size
    
    def _get_suitable_config_key(self, roi_name: str) -> str:
        for key in self.tesserocr_config.keys():
            if key in roi_name:
                return key
        return "default"

    def _get_or_create_easyocr_reader(self):
        if hasattr(self, "_easyocr_reader") and self._easyocr_reader is not None:
            return self._easyocr_reader

        try:
            import torch
            use_gpu = bool(torch.cuda.is_available())
            if use_gpu:
                torch.backends.cudnn.benchmark = True
        except Exception:
            use_gpu = False

        default_lang = "en"

        self._easyocr_reader = easyocr.Reader(
            lang_list=[default_lang],
            gpu=use_gpu,
            verbose=False,
        )
        print(f"Initialized EasyOCR reader with GPU={use_gpu}")
        return self._easyocr_reader

    def _prepare_easyocr_item(self, item):
        roi_name, img_array = item
        cfg_key = self._get_suitable_config_key(roi_name)

        if img_array is None or getattr(img_array, "size", 0) == 0:
            return roi_name, cfg_key, None

        # Keep image untouched; preprocessing is handled upstream.
        return roi_name, cfg_key, img_array
    
    def _transfer_tessconfig_to_easyocr(self, cfg_key: str) -> dict:
        cfg = self.tesserocr_config.get(cfg_key, self.tesserocr_config["default"])
        vars_cfg = cfg.get("vars", {})
        allowlist = vars_cfg.get("tessedit_char_whitelist")
        blocklist = vars_cfg.get("tessedit_char_blacklist")
        return allowlist, blocklist

    def _run_easyocr_batch(self,
                           reader,
                           roi_batch: list[tuple[str, np.ndarray]],
                           cfg_key: str,
                           batch_size: int) -> dict[str, str]:
        
        allowlist, blocklist = self._transfer_tessconfig_to_easyocr(cfg_key)

        roi_names = [name for name, _ in roi_batch]
        images = [img for _, img in roi_batch]

        # EasyOCR batched path requires homogeneous image shapes
        same_shape = all(image.shape == images[0].shape for image in images)
        if not same_shape:
            text_by_roi: dict[str, str] = {}
            for roi_name, image in roi_batch:
                output = reader.readtext(
                    image,
                    detail=0,
                    paragraph=False,
                    decoder="greedy",
                    allowlist=allowlist,
                    blocklist=blocklist,
                )
                text = " ".join(str(token) for token in output).strip() if isinstance(output, list) else str(output or "").strip()
                text_by_roi[roi_name] = text
            return text_by_roi

        raw_outputs = reader.readtext_batched(
            images,
            detail=0,
            paragraph=False,
            batch_size=max(1, min(batch_size, len(images))),
            workers=0,
            decoder="greedy",
            allowlist=allowlist,
            blocklist=blocklist,
        )

        text_by_roi: dict[str, str] = {}
        for roi_name, output in zip(roi_names, raw_outputs):
            if isinstance(output, list):
                text = " ".join(str(token) for token in output).strip()
            elif output is None:
                text = ""
            else:
                text = str(output).strip()

            text_by_roi[roi_name] = text

        return text_by_roi

    def extract_all_region_text_parallel(self,
                                         text_region_images: dict[str, np.ndarray],
                                         postprocess: bool = True) -> dict[str, str]:
        reader = self._get_or_create_easyocr_reader()
        tasks = list(text_region_images.items())

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            prepared = list(executor.map(self._prepare_easyocr_item, tasks))

        results: dict[str, str] = {}
        grouped: dict[tuple[str, tuple[int, ...]], list[tuple[str, np.ndarray]]] = {}
        for roi_name, cfg_key, image in prepared:
            if image is None:
                results[roi_name] = ""
                continue
            grouped.setdefault((cfg_key, image.shape), []).append((roi_name, image))

        for (cfg_key, _shape), roi_group in grouped.items():
            for i in range(0, len(roi_group), self.batch_size):
                batch = roi_group[i:i + self.batch_size]
                batch_results = self._run_easyocr_batch(
                    reader=reader,
                    roi_batch=batch,
                    cfg_key=cfg_key,
                    batch_size=self.batch_size,
                )
                results.update(batch_results)

        if postprocess:
            return {roi_name: self._postprocess_text(text) for roi_name, text in results.items()}
        return results

class PaddleOCRProcessor(OCRProcessor):
    def __init__(self, max_workers=4, batch_size=16):
        super().__init__()
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.ocr = PaddleOCRVL(
            # use_gpu=True,
            # use_layout_detection=False,
            # layout_detection_model_name=None,
            # use_chart_recognition=False,
            # use_seal_recognition=False, 
            # use_ocr_for_image_block=False,
            # use_doc_orientation_classify=False,
            # use_doc_unwarping=False
        )

    def _postprocess_text(self, text: str, region: str) -> str:
        text = OCRProcessor._postprocess_text(text)
        if "size" in region:
            text = text.replace("×", "x")
        elif "subscript" in region:
            text = text.replace("{", "(").replace("}", ")")
        elif "ref" in region:
            text = text.replace(":", "")
        return text

    def extract_all_region_text_parallel(self, text_region_images: dict[str, np.ndarray], postprocess: bool = True) -> dict[str, str]:
        region_keys = list(text_region_images.keys())
        region_images = list(text_region_images.values())
        result_dict: dict[str, str] = {}
        results = self.ocr.predict(
                region_images,
                use_doc_orientation_classify=False,
                use_chart_recognition=False,
                use_seal_recognition=False,
                use_doc_unwarping=False,
                use_layout_detection=False,
                return_json=False,
                return_markdown=False,
                max_new_tokens=64 
            )

        for region_key, result in zip(region_keys, results):
            try:
                text = result.str['res']['parsing_res_list'][0]['block_content']
            except (KeyError, IndexError, TypeError):
                text = ""
            
            if postprocess:
                text = self._postprocess_text(text, region_key)
            
            result_dict[region_key] = text
                
        return result_dict



    