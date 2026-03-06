import config
import cv2
from pathlib import Path
from LabelProcessor import LabelProcessor
from ImageProcessor import ImageProcessor
from ResultStorage import ResultStorage

def main():
    image_name = "W151_many_1.jpg"
    processor = LabelProcessor()
    results = processor.process_label(str(config.IMAGES_DIR / image_name))
    results.display_text_region_images()
    results = ResultStorage(results)
    results.generate_summary(f"W151_many_1_result_retrain", str(config.RESULTS_DIR))

def perform_symbol_image_differencing(querry_image_path):
    querry_image = cv2.imread(querry_image_path)
    processor = LabelProcessor()
    results = processor.process_label(querry_image)
    results.display_symbol_region_images()
    image_processor: ImageProcessor = ImageProcessor.get_suitable_image_processor(results.template_type)
    template_image = cv2.imread(str(config.TEMPLATES[results.template_type]))

    symbol_rois = results.roi_coordinates["symbol_regions"]
    for roi_name, coords in symbol_rois.items():
        querry_crop = results._symbol_images[roi_name]
        template_crop = image_processor.extract_roi(template_image, coords )
        diff = image_processor.calculate_image_difference(template_crop, querry_crop)
        print(f"Difference for {roi_name}: {diff}")

def save_image_from_scan(label_scan_path, multipage=False):
    image_processor = ImageProcessor()
    image_name = Path(label_scan_path).name.replace(".pdf", ".jpg")
    if multipage:
        images = image_processor.convert_multipage_pdf_to_image(label_scan_path)
        for idx, img in enumerate(images):
            output_image_path = config.IMAGES_DIR / f"{image_name.replace('.jpg', '')}_{idx+1}.jpg"
            cv2.imwrite(str(output_image_path), img)
    else:
        output_image_path = config.IMAGES_DIR / image_name
        cv2.imwrite(str(output_image_path), image_processor.convert_pdf_to_image(label_scan_path))

def save_deskewed_aligned_and_cropped_image(image_name, template_type):
    image_path = str(config.IMAGES_DIR / image_name)
    template_path = str(config.LOGO_TEMPLATES[template_type])
    full_label_image = cv2.imread(image_path)  
    image_processor = ImageProcessor()
    processed_image = image_processor.align_image(full_label_image, template_path)
    processed_image = image_processor.extract_roi(processed_image, config.LABEL_DIMENSIONS[template_type])
    cv2.imwrite(str(config.IMAGES_DIR / f"{image_name.replace('.jpg', '')}_aligned_cropped.jpg"), processed_image)


def save_orb_aligned_image():
    image_name = "W151_2_1.jpg"
    image_path = str(config.IMAGES_DIR / image_name)
    full_label_image = cv2.imread(image_path)  
    image_processor = ImageProcessor()
    template_type, full_label_image = image_processor.orb_align_and_clasify(full_label_image, visualize=True)

    # specialize image processor to the template

    _,_,img_w,img_h = config.LABEL_DIMENSIONS[template_type]
    cropped_image = full_label_image[0:img_h, 0:img_w] 
    cv2.imwrite(str(config.IMAGES_DIR / f"W{template_type}_template.jpg"), cropped_image)

def remove_variable_info_from_template(template_type):
    processor = LabelProcessor()
    results = processor.process_label(str(config.TEMPLATES[template_type]))
    cleaned_image = processor._remove_variable_info_from_image(results.aligned_image, results.roi_coordinates)
    cv2.imwrite(str(config.TEMPLATE_DIR / f"{template_type}_cleaned.jpg"), cleaned_image)

def remove_variable_info_from_aligned_image(image_name, template_type):
    processor = LabelProcessor()
    results = processor.process_label(str(config.IMAGES_DIR / image_name))
    cleaned_image = processor._remove_variable_info_from_image(results.aligned_image, results.roi_coordinates)
    cv2.imwrite(str(config.IMAGES_DIR / f"{template_type}_cleaned.jpg"), cleaned_image)

def denormalize_coords_for_full_label():
    image_name = "W151.jpg"
    img =cv2.imread(str(config.IMAGES_DIR / image_name))
    h, w = img.shape[:2]

def test_image_differencing():
    clean_image = cv2.imread(str(config.TEMPLATE_DIR / "151_cleaned.jpg"))
    querry_image = cv2.imread(str(config.IMAGES_DIR / "151_cleaned.jpg"))
    diff = LabelProcessor()._calculate_image_difference(clean_image, querry_image)
    ssim_diff = LabelProcessor().calculate_ssim(clean_image, querry_image)
    print(f"Image difference: {diff}")
    print(f"SSIM: {ssim_diff:.4f}")

def transfer_roi_coordinates():
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
    # perform_symbol_image_differencing(str(config.IMAGES_DIR / "W151_2_gs.jpg"))
    # save_deskewed_aligned_and_cropped_image("W151_2_gs.jpg", "151")
    # save_orb_aligned_image()
    main()