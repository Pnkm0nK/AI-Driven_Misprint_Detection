import cv2
import pdf2image
import numpy as np
import os
import pytesseract
from ROI_draw import create_roi_gui
import dotenv
import json
import matplotlib.pyplot as plt

class LabelProcessor:
    def __init__(self, label_type: str,
                 label_scan_pdf_path: str | None = None,
                 full_label_image = None):

        dotenv.load_dotenv()
        self.poppler_path = os.getenv("POPPLER_PATH")
        self.tesseract_cmd = os.getenv("TESSERACT_PATH")

        self.label_type: str = label_type
        self.label_scan_pdf_path: str | None = label_scan_pdf_path

        if not full_label_image and self.label_scan_pdf_path:
            self.convert_pdf_to_image()
        else:
            self.full_label_image = full_label_image 
        
        self.full_label_image_greyscale = cv2.cvtColor(
            self.full_label_image, cv2.COLOR_BGR2GRAY)
        
        # ROI coordinates stored as (x_top, y_top, x_bottom, y_bottom)
        self.roi_coordinates: dict[str, tuple[int, int, int, int]] = {}

        self.region_texts: dict[str, str] = {}

        self.region_images: dict[str, np.ndarray] = {}

    def convert_pdf_to_image(self, dpi: int = 300):
        images = pdf2image.convert_from_path(pdf_path= self.label_scan_pdf_path, dpi=dpi,
                                            poppler_path=self.poppler_path)
        self.full_label_image = cv2.cvtColor(
            np.array(images[0]), cv2.COLOR_RGB2BGR
        )  # assuming single-page PDF
    
    def add_roi(self, roi_name: str,
                coordinates: tuple[int, int, int, int]):
        self.roi_coordinates[roi_name] = coordinates
    
    def add_roi_dict(self, roi_dict: dict[str, tuple[int, int, int, int]]):
        self.roi_coordinates.update(roi_dict)

    def normalize_roi_coordinates(self,
                                   roi_coordinates: dict[str, tuple[int, int, int, int]],
                                   img_w, img_h) -> dict[str, tuple[float, float, float, float]]:
        normalized_coordinates = {}
        for roi_name, (x_top, y_top, x_bottom, y_bottom) in roi_coordinates.items():
            norm_x_top = x_top / img_w
            norm_y_top = y_top / img_h
            norm_x_bottom = x_bottom / img_w
            norm_y_bottom = y_bottom / img_h
            normalized_coordinates[roi_name] = (norm_x_top, norm_y_top, norm_x_bottom, norm_y_bottom)
        return normalized_coordinates
    
    def denormalize_roi_coordinates(self,
                                     normalized_coordinates: dict[str, tuple[float, float, float, float]],
                                     img_w, img_h) -> dict[str, tuple[int, int, int, int]]:
                            
        denormalized_coordinates = {}
        for roi_name, (norm_x_top, norm_y_top, norm_x_bottom, norm_y_bottom) in normalized_coordinates.items():
            x_top = int(norm_x_top * img_w)
            y_top = int(norm_y_top * img_h)
            x_bottom = int(norm_x_bottom * img_w)
            y_bottom = int(norm_y_bottom * img_h)
            denormalized_coordinates[roi_name] = (x_top, y_top, x_bottom, y_bottom)
        return denormalized_coordinates
        
    def save_roi_json_data(self, output_json_path: str):
        assert self.full_label_image is not None, "Full label image is not loaded."

        # Normalize coordinates before saving
        img_h, img_w = self.full_label_image.shape[:2]
        normalized_coordinates = self.normalize_roi_coordinates(self.roi_coordinates, img_w, img_h)

        with open(output_json_path, 'w') as f:
            json.dump(normalized_coordinates, f, indent=4)

    def load_roi_json_data(self, input_json_path: str):
        assert self.full_label_image is not None, "Full label image is not loaded."
        # Load and denormalize coordinates after loading

        with open(input_json_path, 'r') as f:
            normalized_coordinates = json.load(f)

        img_h, img_w = self.full_label_image.shape[:2]
        self.roi_coordinates = self.denormalize_roi_coordinates(normalized_coordinates, img_w, img_h)

    def establish_starting_position(self, template_image_path: str)-> tuple[int, int]:
        # Use template matching to find starting position
        # reference for ROIs
        template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
        assert template_image is not None, "Template image not found or could not be loaded."
 
        # find normalized least square difference between template and full image
        res = cv2.matchTemplate(self.full_label_image_greyscale,template_image,cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return min_loc  # top-left corner of matched region
    
    def preprocess_region(self, region_image: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
        # Apply binary thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def extract_text_from_region_image(self, region_image: np.ndarray) -> str:
        # Preprocess the image
        preprocessed_image = self.preprocess_region(region_image)
        # Perform OCR using pytesseract
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        custom_config = str(r'--oem 3 --psm 6')  # OEM 3: Default, PSM 6: Assume a single uniform block of text
        text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
        return text.strip()
    
    def extract_all_region_texts(self) -> dict[str, str]:
        assert self.region_images, "Region images have not been extracted."

        for roi_name, img in self.region_images.items():
            self.region_texts[roi_name] = self.extract_text_from_region_image(img)

    def extract_region_images(self):
        assert self.full_label_image is not None, "Full label image is not loaded."
        
        for roi_name,(x0, y0, x1, y1) in self.roi_coordinates.items():
            self.region_images[roi_name] = self.full_label_image[y0:y1, x0:x1]
    
    def display_region_images(self):
        excluded_regions = ["top_label"]
        for roi_name, image in self.region_images.items():
            if roi_name in self.region_texts and roi_name not in excluded_regions:
                image = image.copy()
                image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2), interpolation=cv2.INTER_LINEAR)
                cv2.putText(image, self.region_texts[roi_name], (5, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1)
            cv2.imshow(roi_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class LabelProcessorDirector:
    def __init__(self):
        pass

    def construct_type_146_label_processor(self,
                                       label_scan_pdf_path: str) -> LabelProcessor:
        processor = LabelProcessor(
            label_type="146",
            label_scan_pdf_path=label_scan_pdf_path
        )
        start_x, start_y = processor.establish_starting_position(
            template_image_path="./images/logo_template.jpg"
        )

        processor.full_label_image = processor.full_label_image[start_y:, start_x:] 

        processor.load_roi_json_data(
            input_json_path="./roi_data/label_146_rois.json")
        return processor

    def add_rois_to_type_146_label_processor(self,
                                       processor: LabelProcessor):
        new_rois =create_roi_gui(
            full_image=processor.full_label_image,
            roi_coordinates=processor.roi_coordinates
        )

        processor.add_roi_dict(new_rois)
        processor.extract_region_images()
        processor.display_region_images()
        processor.save_roi_json_data(
            output_json_path="./roi_data/label_146_rois.json"
        )

if __name__ == "__main__":
    processor = LabelProcessorDirector().construct_type_146_label_processor(
        label_scan_pdf_path="../label_scans/M333023W146.pdf")
    processor.extract_region_images()
    processor.extract_all_region_texts()
    processor.display_region_images()