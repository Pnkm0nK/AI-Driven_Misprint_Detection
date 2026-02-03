import cv2
import json
import numpy as np

class ROIStorage:
    def __init__(self, full_label_image: np.ndarray, roi_json_path: str = None):
        assert full_label_image is not None, "Full label image must be provided."
        self.full_label_image: np.ndarray = full_label_image

        self.img_h, self.img_w = full_label_image.shape[:2]

        self.roi_json_path = roi_json_path

    @staticmethod
    def normalize_roi_coordinates(
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

    @staticmethod    
    def denormalize_roi_coordinates(normalized_coordinates: dict[str, tuple[float, float, float, float]],
                                     img_w, img_h) -> dict[str, tuple[int, int, int, int]]:
                            
        denormalized_coordinates = {}
        for roi_name, (norm_x_top, norm_y_top, norm_x_bottom, norm_y_bottom) in normalized_coordinates.items():
            x_top = int(norm_x_top * img_w)
            y_top = int(norm_y_top * img_h)
            x_bottom = int(norm_x_bottom * img_w)
            y_bottom = int(norm_y_bottom * img_h)
            denormalized_coordinates[roi_name] = (x_top, y_top, x_bottom, y_bottom)
        return denormalized_coordinates
        
    def save_roi_json_data(self, roi_coordinates: dict[str, tuple[int, int, int, int]]):
        # Normalize coordinates before saving
        normalized_coordinates = self.normalize_roi_coordinates(roi_coordinates, self.img_w, self.img_h)

        with open(self.roi_json_path, 'w') as f:
            json.dump(normalized_coordinates, f, indent=4)
    
    def load_roi_json_data(self):
        # Load and denormalize coordinates after loading

        try:
            with open(self.roi_json_path, 'r') as f:
                normalized_coordinates = json.load(f)
        except:
            return {}

        return self.denormalize_roi_coordinates(normalized_coordinates, self.img_w, self.img_h)

    def establish_roi_starting_position(self, template_image_path: str, padding_x:int)-> tuple[int, int]:
        # Use template matching to find starting position
        # reference for ROIs
        image = cv2.cvtColor(self.full_label_image, cv2.COLOR_BGR2GRAY)
        template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
        assert template_image is not None, "Template image not found or could not be loaded."
 
        # find normalized least square difference between template and full image
        res = cv2.matchTemplate(image,template_image,cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return (min_loc[0] - padding_x, min_loc[1])  # top-left corner of matched region