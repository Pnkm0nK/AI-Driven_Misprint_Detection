import pdf2image
import os
import cv2
import numpy as np

class PDFConverter:
    def __init__(self, poppler_path: str | None=None):
        if poppler_path is None:
            poppler_path = os.getenv("POPPLER_PATH", "")
        self.poppler_path = poppler_path

    def convert_pdf_to_image(self, pdf_path: str, dpi: int = 300)-> np.ndarray:
        '''
        Convert a single-page PDF to an image using pdf2image. Returns the image as a numpy array in BGR format.
        '''
        images = pdf2image.convert_from_path(pdf_path= pdf_path, dpi=dpi,
                                            poppler_path=self.poppler_path)
        return cv2.cvtColor(
            np.array(images[0]), cv2.COLOR_RGB2BGR
        ) 
    
    def convert_multipage_pdf_to_image(self, pdf_path: str, dpi: int = 300) -> np.ndarray:
        images = pdf2image.convert_from_path(pdf_path= pdf_path, dpi=dpi,
                                            poppler_path=self.poppler_path)
        cv_images = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]
        return cv_images