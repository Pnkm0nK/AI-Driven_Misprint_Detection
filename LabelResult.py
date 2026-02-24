import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

class LabelResult:
    def __init__(self, template_name: str, region_texts: dict[str,str], region_barcodes: dict[str,str],
                 text_region_images: dict[str,np.ndarray], barcode_images: dict[str,np.ndarray]):
        '''
        Docstring for __init__

        :param template_name: Name of the template(e.g. "151", "146", "063")
        :type template_name: str
        :param region_texts: Dictionary mapping each text region name to its extracted text.
        :type region_texts: dict[str, str]
        :param region_barcodes: Dictionary mapping each barcode region name to its extracted barcode value.
        :type region_barcodes: dict[str, str]
        :param text_region_images: Dictionary mapping each text region name to its image.
        :type text_region_images: dict[str, np.ndarray]
        :param barcode_images: Dictionary mapping each barcode region name to its image.
        :type barcode_images: dict[str, np.ndarray]
        '''
        self.template_name = template_name
        self.region_texts = region_texts
        self.region_barcodes = region_barcodes
        self._text_region_images = text_region_images
        self._barcode_images = barcode_images

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
        for roi_name, image in self._text_region_images.items():
            if roi_name in self.region_texts:
                self._display_region_image(roi_name, image, result=self.region_texts[roi_name])
        for roi_name, image in self._barcode_images.items():
            if roi_name in self.region_barcodes:
                self._display_region_image(roi_name, image, result=self.region_barcodes[roi_name])
        cv2.waitKey(0)
        cv2.destroyAllWindows()