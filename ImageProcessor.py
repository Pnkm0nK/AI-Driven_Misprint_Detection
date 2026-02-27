import os
import time
import cv2
import dotenv
import pdf2image
from deskew import determine_skew
from utils import get_template_matching_results
import numpy as np
import config

class ImageProcessor():
    def __init__(self):
        dotenv.load_dotenv()
        self.poppler_path = os.getenv("POPPLER_PATH")
    
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
    
    def convert_to_greyscale(self, img: np.ndarray) -> np.ndarray:
        '''
        Convert an image to grayscale if it is in color. If the image is already in grayscale, return it as is
        '''
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return img
    
    def threshold_image(self, image: np.ndarray) -> np.ndarray:
        '''
        Apply Otsu's thresholding to binarize the input image. Returns the thresholded image.
        '''
        _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def extract_roi(self, image: np.ndarray, coordinates: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = coordinates
        roi_image = image[y0:y1, x0:x1]
        return roi_image

    def deskew_image(self, image:np.ndarray) -> np.ndarray:
        skew_angle = determine_skew(image, max_angle=30)
        rot_mat = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), skew_angle, 1)
        image = cv2.warpAffine(image, rot_mat, (image.shape[1],image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        return image
    
    def align_image(self, image: np.ndarray, template_image_path: str) -> np.ndarray:
        '''
        Align the input image to the template image using deskewing and template matching. Returns the aligned image.
        '''
        deskewed = self.deskew_image(image)

        _, loc = get_template_matching_results(deskewed, template_image_path)

        padding = config.PADDING

        x_start, y_start = loc 
        x_start -= padding
        if x_start < 0:
            deskewed = cv2.copyMakeBorder(deskewed, padding-x_start, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
            x_start = 0
        return deskewed[y_start:, x_start:]
    
    def unsharp(self, image: np.ndarray, kernel_size=(1,1), sigma=2, amount=2.0, threshold=0) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, threshold)
        return sharpened
    
    def orb_align_and_clasify(self, image: np.ndarray, n_features:int=250, max_matches: int = 15, visualize=False) -> tuple[str, np.ndarray]:
        '''
        Classify and align the input image to the best matching template using ORB feature matching.
        Returns the estimated template name and the aligned image.

        on my pc shows 0.24 seconds computation time for 30 features and 10 matches per image on average
        
        :param image: Input image to be aligned 
        :type image: np.ndarray
        :param n_features: Number of ORB features to detect
        :type n_features: int
        :param max_matches: Maximum number of ORB matches to consider for alignment
        :type max_matches: int
        :param visualize: Whether to visualize the ORB matches and alignment results
        :return: Estimated template name and aligned image in a tuple
        :rtype: tuple[str, np.ndarray]
        '''
        def align_using_orb_matches(matches, dst_kps, template):
            if len(matches) >= 4:
                src_pts = np.float32([src_kps[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                dst_pts = np.float32([dst_kps[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                
                # No perspective change, using affine transform for deskewing and translation correction
                M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
                
                return cv2.warpAffine(original_image, M, (template.shape[1], template.shape[0]),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255,255,255))

        
        original_image = image.copy()
        image = self.convert_to_greyscale(image)

        orb = cv2.ORB_create(nfeatures=n_features)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        src_kps, query_descrs = orb.detectAndCompute(image, None)

        # go through templates and find best match based on total distance of top matches
        # save intermediate results to reuse in alignment step and visualization
        least_distance = float("inf") 
        estimated_template_type = None
        best_matches = []
        best_dst_kps = []

        for template_type, template_img_path in config.TEMPLATES.items():
            template = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
            assert template is not None, f"Failed to load template image at {template_img_path}"
            dst_kps, target_descrs = orb.detectAndCompute(template, None)
            # maybe try Knn match and Lowe's ratio test if too many false matches with crossCheck
            matches = bf.match(query_descrs, target_descrs)
            matches = sorted(matches, key=lambda x: x.distance)
            top = matches[:min(max_matches, len(matches))]
            total_distance = sum(m.distance for m in top) if top else float("inf")

            if total_distance < least_distance:
                least_distance = total_distance
                estimated_template_type = template_type
                best_dst_kps = dst_kps
                best_matches = top

        best_template = cv2.imread(config.TEMPLATES[estimated_template_type], cv2.IMREAD_GRAYSCALE) 
        aligned_image = align_using_orb_matches(best_matches, best_dst_kps, best_template)

        if visualize:
            # reload template in color for visualization only
            img_match = cv2.drawMatches(image, src_kps, best_template, best_dst_kps, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            height, width = img_match.shape[:2]
            aspect_ratio = width / height
            width = 800
            height = int(width / aspect_ratio)
            img_match = cv2.resize(img_match, (width, height))

            cv2.imshow("Matches", img_match)
            cv2.waitKey(0)
        return estimated_template_type, aligned_image
    
    def preprocess_image_general(self, image: np.ndarray) -> np.ndarray:
        # General preprocessing: convert to grayscale 
        gray = self.convert_to_greyscale(image) 
        resized = cv2.resize(gray, (0,0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC) 
        # unsharp_image = self.unsharp(resized)
        
        # padded = cv2.copyMakeBorder(unsharp_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255,255,255])
        # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return resized
    
    def preprocess_top_label(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_image_general(image)
    
    def preprocess_large_label(self, image: np.ndarray) -> np.ndarray:
        # Rotate counterclockwise and do general preprocessing
        turned = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return self.preprocess_image_general(turned)
    
    def preprocess_small_label(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_image_general(image)
    
    def preprocess_patient_label(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess_image_general(image)
    
    def get_suitable_preprocessing_method(self, roi_name: str):
        if "top" in roi_name:
            return self.preprocess_top_label
        elif "large" in roi_name:
            return self.preprocess_large_label
        elif "small" in roi_name:
            return self.preprocess_small_label
        elif "patient" in roi_name:
            return self.preprocess_patient_label
        else:
            return self.preprocess_image_general
    
    def preprocess_region_image(self, roi_name: str, image: np.ndarray) -> np.ndarray:
        '''
        Applies suitable preprocessing to the input region image based on the region type

        :param roi_name: Name of the region of interest, used to determine the suitable preprocessing method 
        :type roi_name: str
        :param image: The region image to be preprocessed
        :type image: np.ndarray
        :return: The preprocessed region image
        :rtype: ndarray[_AnyShape, dtype[Any]]
        '''
        preprocess_method = self.get_suitable_preprocessing_method(roi_name)
        preprocessed_image = preprocess_method(image)
        return preprocessed_image

    def get_suitable_image_processor(self, template_type: str):
        '''Returns an instance of the suitable ImageProcessor subclass based on the template name.
        If no specific processor is found for the template, returns a default ImageProcessor
        instance.

        :param template_type: Name of the template
        :type template_type: str
        '''

        cls = PROCESSOR_CLASSES.get(template_type, ImageProcessor)
        return cls() 
    
class Type151ImageProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()
    def preprocess_small_label(self, image: np.ndarray) -> np.ndarray:
        image = cv2.rotate(image, cv2.ROTATE_180)
        return self.preprocess_image_general(image)
    def preprocess_patient_label(self, image):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return super().preprocess_image_general(image)

class Type146ImageProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()

class Type063ImageProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()
    def ensure_correct_orientation(self, image, template_image_path):
        height, width = image.shape[:2]
        if height > width:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
        min_val_trial_1, loc = get_template_matching_results(image, template_image_path)
        upside_down_image = cv2.rotate(image, cv2.ROTATE_180)
        min_val_trial_2, loc = get_template_matching_results(upside_down_image, template_image_path)
        if min_val_trial_1 < min_val_trial_2:
            return image

        return upside_down_image

    def preprocess_large_label(self, image):
        return self.preprocess_image_general(image)


PROCESSOR_CLASSES: dict[str, type[ImageProcessor]] = {
    "151": Type151ImageProcessor,
    "146": Type146ImageProcessor,
    "063": Type063ImageProcessor,
}

if __name__ == "__main__": 
    processor = ImageProcessor()
    # images = processor.convert_multipage_pdf_to_image("../label_scans/0400063.pdf")
    # for i, img in enumerate(images):
    #     cv2.imwrite(f"./images/063_page_{i}.jpg", img)
    image_path = [str(config.IMAGES_DIR / "W151.jpg"), str(config.IMAGES_DIR / "W146.jpg"), str(config.IMAGES_DIR / "063_page_0.jpg")]
    cumm_time = 0
    cnt = 0
    for path in image_path:
        image = cv2.imread(path)
        time_start = time.time()
        name, aligned = processor.orb_align_and_clasify(image, n_features=30, max_matches=10)
        cumm_time += time.time() - time_start
        cnt += 1
    print(f"Alignment of {cnt} images took {cumm_time:.3f} seconds")