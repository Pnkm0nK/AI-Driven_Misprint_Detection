import config
import cv2
from pathlib import Path
from LabelProcessor import LabelProcessor
from ImageProcessor import ImageProcessor
from ResultStorage import ResultStorage

def main():
    image_name = "W151.jpg"
    processor = LabelProcessor()
    results = processor.process_label(str(config.IMAGES_DIR / image_name))
    results = ResultStorage(results)
    results.generate_summary(f"W151_result", str(config.RESULTS_DIR))
    results.display_regions_with_mismatches()

def deskew_and_match_template():
    image_name = "W151.jpg"
    image_path = str(config.IMAGES_DIR / image_name)
    template_path = str(config.TEMPLATE_DIR / "label_151_template.jpg")
    full_label_image = cv2.imread(image_path)  
    image_processor = ImageProcessor()
    image_processor.align_image(full_label_image,)

def save_aligned_image():
    image_name = "W146.jpg"
    image_path = str(config.IMAGES_DIR / image_name)
    full_label_image = cv2.imread(image_path)  
    image_processor = ImageProcessor()
    template_type, full_label_image = image_processor.orb_align_and_clasify(full_label_image)

    # specialize image processor to the template

    _,_,img_w,img_h = config.LABEL_DIMENSIONS[template_type]
    cropped_image = full_label_image[0:img_h, 0:img_w] 
    cv2.imwrite(str(config.IMAGES_DIR / f"W{template_type}_template.jpg"), cropped_image)

def try_removing_variable_info():
    image_name = "W151.jpg"
    processor = LabelProcessor()
    results = processor.process_label(str(config.IMAGES_DIR / image_name))
    cleaned_image = processor._remove_variable_info_from_image(results.aligned_image, results.roi_coordinates)
    cv2.imwrite(str(config.RESULTS_DIR / "cleaned_image.jpg"), cleaned_image)

def denormalize_coords_for_full_label():
    image_name = "W151.jpg"
    img =cv2.imread(str(config.IMAGES_DIR / image_name))
    h, w = img.shape[:2]

    from ROIStorage import ROIStorage
    object ={
        "full_label": [
            0.0,
            0.0,
            0.4440282979608822,
            0.8830562846310878
        ]
    }
    print(ROIStorage.denormalize_roi_coordinates(object, w,h))

def transfer_roi_coordinater():
    from ROIStorage import ROIStorage
    import json
    image_name = "W151.jpg"
    img =cv2.imread(str(config.IMAGES_DIR / image_name))
    h, w = img.shape[:2]
    storage = ROIStorage(w, h, "151")
    _,_,new_w, new_h = config.LABEL_151_DIMENSIONS
    rois = storage.load_roi_json_data()
    coords = storage.transfer_roi_coordinates(rois, w, h, new_w, new_h)
    normed = {}
    for category, roi in coords.items():
        normed[category] = storage.normalize_roi_coordinates(roi, new_w, new_h)
    open(config.ROI_DIR / "label_151_rois_transfered.json", 'w').write(json.dumps(normed, indent=4))


if __name__ == "__main__":
    save_aligned_image()