import cv2
import json
import numpy as np
import os
import pytesseract
import dotenv
import zxingcpp as zxing
from PIL import Image, ImageDraw, ImageFont

from ImageProcessor import ImageProcessor, Type151ImageProcessor, Type146ImageProcessor, Type063ImageProcessor
from ROIStorage import ROIStorage
from ResultStorage import ResultStorage



class LabelProcessor:
    '''
    Class for e2e processing of label scans.
    Use get_extracted_texts() and get_extracted_barcodes() to retrieve results after processing.
    display_all_region_images() can be used to visualize the extracted region images and their OCR results.
    '''
    def __init__(self, scan_path: str,
                roi_json_path: str,
                image_processor: ImageProcessor = ImageProcessor()):
        '''
        Class for e2e processing of label scans.
        :param scan_path: path to pdf or image file of label scan 
        :type scan_path: str
        :param roi_json_path: path to json file containing roi coordinates
        :type roi_json_path: str
        :param image_processor: instance of ImageProcessor to use for image processing tasks. If not provided, a default ImageProcessor will be used.
        :type image_processor: ImageProcessor
        '''
        dotenv.load_dotenv()
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")
        
        with open("./tesseract_config.json", 'r') as f:
            self.tesseract_config = json.load(f)
        
        self.image_processor = image_processor
        template_image_path = "./images/logo_template.jpg"

        # get image from scan
        if scan_path.lower().endswith(".pdf"):
            self.full_label_image = self.image_processor.convert_pdf_to_image(scan_path)
        elif scan_path.lower().endswith((".jpg", ".jpeg", ".png")):
            self.full_label_image = cv2.imread(scan_path)

        # Align image to template
        PADDING = int(os.getenv("PADDING"))
        self.full_label_image = self.image_processor.align_image(self.full_label_image, template_image_path, padding=PADDING)

        self.roi_storage = ROIStorage(img_h=self.full_label_image.shape[0],
                                      img_w=self.full_label_image.shape[1],
                                      roi_json_path=roi_json_path)

        # crop image to start from anchor point

        self.roi_coordinates = self.roi_storage.load_roi_json_data()

        self.text_region_images: dict[str, np.ndarray] = self._extract_region_images(self.roi_coordinates["text_regions"])
        self.region_texts: dict[str, str] = self._extract_all_region_texts()

        self.barcode_images: dict[str, np.ndarray] = self._extract_region_images(self.roi_coordinates["barcode_regions"])
        self.region_barcodes: dict[str, str] = self._extract_all_barcodes()

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

    @staticmethod 
    def _display_region_image(roi_name: str, image: np.ndarray, result: str):
            image = Image.fromarray(image)
            image = image.convert("RGB")

            padding_y = 10
            padding_x = 5
            
            display_text_region = f"Region: {roi_name}"
            display_text_ocr_result = f"OCR Result: {result}"
            
            try:
                font_title = ImageFont.truetype("AvenirNextWorld-Regular.ttf", 21)
                font_text = ImageFont.truetype("AvenirNextWorld-Bold.ttf", 32)
            except:
                font_title = ImageFont.load_default()
                font_text = ImageFont.load_default()
            
            # use text bboxes to expand canvas size if text doesn't fit
            draw = ImageDraw.Draw(image)

            title_bbox = draw.textbbox((0, 0), display_text_region, font=font_title)
            text_bbox = draw.textbbox((0, 0), display_text_ocr_result, font=font_text)

            title_width = title_bbox[2] - title_bbox[0]
            title_height = title_bbox[3] - title_bbox[1]

            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            max_text_width = max(title_width, text_width)
            
            required_height = title_height + text_height + image.height + 4 * padding_y

            required_width = max_text_width + padding_x
            if required_width > image.width:
                new_image = Image.new("RGB", (required_width, required_height), (255, 255, 255))
                new_image.paste(image, (0, title_height + 2 * padding_y))
            else:
                new_image = Image.new("RGB", (image.width, required_height), (255, 255, 255))
                new_image.paste(image, (0, title_height + 2 * padding_y))
            
            draw = ImageDraw.Draw(new_image)
            draw.text((padding_x, padding_y), display_text_region, fill=(255, 100, 0), font=font_title)
            draw.text((padding_x, image.height + title_height + 2 * padding_y), display_text_ocr_result, fill=(0, 100, 255), font=font_text)
            
            # Convert back to numpy for cv2.imshow
            image_np = np.array(new_image)
            cv2.imshow(roi_name, image_np)

    def get_extracted_texts(self) -> dict[str, str]:
        return self.region_texts
    
    def get_extracted_barcodes(self) -> dict[str, str]:
        return self.region_barcodes

    def display_all_region_images(self):
        for roi_name, image in self.text_region_images.items():
            if roi_name in self.region_texts:
                self._display_region_image(roi_name, image, result=self.region_texts[roi_name])
        for roi_name, image in self.barcode_images.items():
            if roi_name in self.region_barcodes:
                self._display_region_image(roi_name, image, result=self.region_barcodes[roi_name])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class LabelFactory:
    @staticmethod
    def create_label_processor(type: str, scan_path: str) -> LabelProcessor:
        if type == "151":
            image_processor = Type151ImageProcessor()
            roi_json_path = "./roi_data/label_151_rois.json"
        elif type == "146":
            image_processor = Type146ImageProcessor()
            roi_json_path = "./roi_data/label_146_rois.json"
        elif type == "063":
            image_processor = Type063ImageProcessor()
            roi_json_path = "./roi_data/label_063_rois.json"
        else:
            raise ValueError(f"Unsupported label type: {type}")

        
        return LabelProcessor(scan_path, roi_json_path, image_processor=image_processor)
    
if __name__ == "__main__":
    label_type = "146"
    processor = LabelFactory.create_label_processor(type=label_type, scan_path="./images/W146.jpg")
    processor.display_all_region_images()
    results = ResultStorage(extracted_texts=processor.get_extracted_texts(),
                            gt_file_path=f"./ground_truth/label_{label_type}_gt.json",
                            extracted_barcodes=processor.get_extracted_barcodes()
                            )
    results.generate_summary(f"{label_type}_unsharp+resized_results", "./results")