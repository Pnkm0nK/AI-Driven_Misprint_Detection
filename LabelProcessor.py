import cv2
from ImageProcessor import ImageProcessor
from ROIStorage import ROIStorage
import numpy as np
import os
import pytesseract
import dotenv

class LabelProcessor:
    def __init__(self, label_scan_pdf_path: str, roi_json_path: str):
        dotenv.load_dotenv()
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")

        self.image_processor = ImageProcessor()
        self.full_label_image = self.image_processor.convert_pdf_to_image(label_scan_pdf_path)

        PADDING = int(os.getenv("PADDING"))
        self.roi_storage = ROIStorage(self.full_label_image, roi_json_path=roi_json_path)
        start_x, start_y =self.roi_storage.establish_roi_starting_position(
            template_image_path="./images/logo_template.jpg", padding_x=PADDING)
        self.full_label_image = self.full_label_image[start_y:, start_x:]

        self.roi_coordinates = self.roi_storage.load_roi_json_data()

        self.region_images: dict[str, np.ndarray] = self._extract_region_images()

        self.region_texts: dict[str, str] = self._extract_all_region_texts()

    def _extract_text_from_region_image(self, region_image: np.ndarray) -> str:
        # Perform OCR using pytesseract
        custom_config = str(r'--oem 3 --psm 6')  # OEM 3: Default, PSM 6: Assume a single uniform block of text
        text = pytesseract.image_to_string(region_image, config=custom_config)
        return text.strip()


    def _extract_region_images(self) -> dict[str, np.ndarray]:
        region_images = {}
        for roi_name,(x0, y0, x1, y1) in self.roi_coordinates.items():
            image = self.full_label_image[y0:y1, x0:x1]
            preprocess_method = self.image_processor.get_suitable_preprocessing_method(roi_name)
            preprocessed_image = preprocess_method(image)
            region_images[roi_name] = preprocessed_image
        return region_images

    def _extract_all_region_texts(self) -> dict[str, str]:
        assert self.region_images, "Region images have not been extracted."

        region_texts = {}
        for roi_name, img in self.region_images.items():
            region_texts[roi_name] = self._extract_text_from_region_image(img)
        return region_texts

    def display_region_images(self):
        for roi_name, image in self.region_images.items():
            if roi_name in self.region_texts:
                image = image.copy()
                image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2), interpolation=cv2.INTER_LINEAR)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = cv2.copyMakeBorder(image, 50, 50, 0, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                display_text_region = f"Region: {roi_name}"
                display_text_ocr_result = f"OCR Result: {self.region_texts[roi_name]}"
                cv2.putText(image, display_text_ocr_result, (5, image.shape[0]- 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 56, 255),2)
                cv2.putText(image, display_text_region, (5, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 56, 0),2)
            cv2.imshow(roi_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save_extracted_texts_to_txt(self, output_txt_path: str):
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for roi_name, text in self.region_texts.items():
                f.write(f"--- Region: {roi_name} ---\n")
                f.write(text + "\n\n")
    
    def save_extracted_texts_to_json(self, output_json_path: str):
        import json
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.region_texts, f, indent=4, ensure_ascii=False)
        


    
if __name__ == "__main__":
    processor = LabelProcessor("../label_scans/M333023W146.pdf", "./roi_data/label_146_rois.json")
    processor.save_extracted_texts_to_txt("./extracted_info/label_146_texts.txt")
    processor.save_extracted_texts_to_json("./extracted_info/label_146_texts.json")
    processor.display_region_images()