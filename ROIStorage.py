import json
from config import ROI_FILES
from custom_types import ROICollection, ROIObject

class ROIStorage:
    def __init__(self, img_w: int, img_h: int, template_type: str):
        self.img_h = img_h
        self.img_w = img_w

        self.roi_json_path = ROI_FILES[template_type]

        self.roi_data = self.load_roi_json_data()

    @staticmethod
    def normalize_roi_coordinates(roi_coordinates: ROIObject,
                                  img_w, img_h) -> ROIObject: 
        normalized_coordinates = {}
        for roi_name, (x_top, y_top, x_bottom, y_bottom) in roi_coordinates.items():
            norm_x_top = x_top / img_w
            norm_y_top = y_top / img_h
            norm_x_bottom = x_bottom / img_w
            norm_y_bottom = y_bottom / img_h
            normalized_coordinates[roi_name] = (norm_x_top, norm_y_top, norm_x_bottom, norm_y_bottom)
        return normalized_coordinates

    @staticmethod    
    def denormalize_roi_coordinates(normalized_coordinates: ROIObject,
                                     img_w, img_h) -> ROIObject:
                            
        denormalized_coordinates = {}
        for roi_name, (norm_x_top, norm_y_top, norm_x_bottom, norm_y_bottom) in normalized_coordinates.items():
            x_top = int(norm_x_top * img_w)
            y_top = int(norm_y_top * img_h)
            x_bottom = int(norm_x_bottom * img_w)
            y_bottom = int(norm_y_bottom * img_h)
            denormalized_coordinates[roi_name] = (x_top, y_top, x_bottom, y_bottom)
        return denormalized_coordinates
        
    def save_roi_json_data(self, rois: ROICollection):
        # Normalize coordinates before saving
        for roi_category, roi in rois.items():
            rois[roi_category] = self.normalize_roi_coordinates(roi, self.img_w, self.img_h)

        with open(self.roi_json_path, 'w') as f:
            json.dump(rois, f, indent=4)
    

    def load_roi_json_data(self) -> ROICollection:
        # Load and denormalize coordinates after loading
        try:
            with open(self.roi_json_path, 'r') as f:
                rois = json.load(f)
        except FileNotFoundError:
            print(f"ROI JSON file not found at {self.roi_json_path}. Returning empty ROI collection.")
            return {}
        

        for roi_category, roi in rois.items():
            rois[roi_category] = self.denormalize_roi_coordinates(roi, self.img_w, self.img_h)
 
        return rois
