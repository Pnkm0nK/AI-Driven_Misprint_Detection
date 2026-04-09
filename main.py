import utilities.config as config
import time
import os
import cv2
from pathlib import Path
from modules.LabelProcessor import LabelProcessor
from modules.PDFConverter import PDFConverter
from modules.ImageProcessor import ImageProcessor
import modules.image_processing_functions as ipf
from modules.ResultPostprocessor import ResultPostprocessor
from utilities.data_parser import parse_data_from_loftware
from utilities.ROI_draw import annotate_template_rois, annotate_label_types
import utilities.utils as utils


def main():
    image_name = "W151_many_1.jpg"
    gt_name = "W151_many_1_gt.json"
    gt_path = config.GT_DIR / gt_name
    processor = LabelProcessor()
    results = processor.process_label(str(config.IMAGES_DIR /"151" / image_name))
    results = ResultPostprocessor(results, gt_path)
    results.show_highlighted_mismatches()
    results.generate_summary(f"W151_many_1_e2e_tesserocr", str(config.RESULTS_DIR))

def test_augmentation(img_path):
    image = cv2.imread(str(img_path))
    augmented_image = ipf.add_binary_simplex_window(image, color=(255, 255, 255), threshold=0.7, scale=0.01)
    augmented_image = ipf.add_binary_simplex_window(augmented_image, color=(5, 10, 5),threshold=0.7, scale=0.04)
    augmented_image = ipf.add_motion_blur(augmented_image, (40, 20), (700, 80), kernel_size=7)
    augmented_image = ipf.add_streaks(augmented_image, num_streaks=3, color=(0, 0, 0), thickness=4)
    cv2.imshow("Simplex smudged Image", augmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def augment_images(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for image_path in Path(image_dir).glob("*.jpg"):
        image = cv2.imread(str(image_path))
        image = ipf.apply_random_augmentation(image)
        output_path = Path(output_dir) / image_path.name
        cv2.imwrite(str(output_path), image)

def check_regions(img_folder):
    processor = LabelProcessor()
    for img_path in os.listdir(img_folder):
        if img_path.endswith(".jpg") or img_path.endswith(".png"):
            img_path = os.path.join(img_folder, img_path)
            results = processor.process_label(str(img_path))
            results.display_text_region_images()

def generate_train_data(template_type):
    image_dir = config.IMAGES_DIR / template_type
    output_dir = config.BASE_DIR / "annomaly_detection" / "train_data"
    for image_path in image_dir.glob("*.jpg"):
        image = cv2.imread(str(image_path))
        aligned_image =ipf.orb_align(image, template_type, n_features=500, max_matches=100)
        cv2.imwrite(str(output_dir / image_path.name), aligned_image)

def remove_markup_from_images(image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(image_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(image_folder, file)
            image = cv2.imread(image_path)
            cleaned_image = utils.remove_markup_from_image(image)
            if file.endswith(".png"):
                file = file.replace(".png", ".jpg")
            output_path = os.path.join(output_folder, file)
            # write as jpg
            cv2.imwrite(output_path, cleaned_image)

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
        template_crop = ipf.extract_roi(template_image, coords )
        diff = ipf.calculate_image_difference(template_crop, querry_crop)
        print(f"Difference for {roi_name}: {diff}")

def save_image_from_scan(label_scan_path, multipage=False):
    processor = PDFConverter()
    image_name = Path(label_scan_path).name.replace(".pdf", ".jpg")
    if multipage:
        images = processor.convert_multipage_pdf_to_image(label_scan_path)
        for idx, img in enumerate(images):
            output_image_path = config.IMAGES_DIR / f"{image_name.replace('.jpg', '')}_{idx+1}.jpg"
            cv2.imwrite(str(output_image_path), img)
    else:
        output_image_path = config.IMAGES_DIR / image_name
        cv2.imwrite(str(output_image_path), processor.convert_pdf_to_image(label_scan_path))

def save_deskewed_aligned_and_cropped_image(image_name, template_type):
    image_path = str(config.IMAGES_DIR / image_name)
    template_path = str(config.LOGO_TEMPLATES[template_type])
    full_label_image = cv2.imread(image_path)  
    processed_image = ipf.align_image(full_label_image, template_path)
    processed_image = ipf.extract_roi(processed_image, config.LABEL_DIMENSIONS[template_type])
    cv2.imwrite(str(config.IMAGES_DIR / f"{image_name.replace('.jpg', '')}_aligned_cropped.jpg"), processed_image)

def save_orb_aligned_image():
    image_name = "W151_2_1.jpg"
    image_path = str(config.IMAGES_DIR / image_name)
    full_label_image = cv2.imread(image_path)  
    template_type, full_label_image = ipf.orb_align_and_classify(full_label_image, visualize=True)

    # specialize image processor to the template

    _,_,img_w,img_h = config.LABEL_DIMENSIONS[template_type]
    cropped_image = full_label_image[0:img_h, 0:img_w] 
    cv2.imwrite(str(config.IMAGES_DIR / f"W{template_type}_template.jpg"), cropped_image)

def save_orb_aligned_image_for_all(image_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_dir.glob("*.jpg"):
        full_label_image = cv2.imread(str(image_path))  
        template_type = "151"
        template_type, full_label_image = ipf.orb_align(full_label_image, template_type=template_type, n_features=200, max_matches=50, visualize=False)

        _,_,img_w,img_h = config.LABEL_DIMENSIONS[template_type]
        cropped_image = full_label_image[0:img_h, 0:img_w] 
        output_image_path = output_dir / f"{image_path.stem}_aligned.jpg"
        cv2.imwrite(str(output_image_path), cropped_image)

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

def find_not_included_files_in_folder(folder1, folder2, output_folder):
    files_in_folder1 = [os.path.splitext(f)[0] for f in os.listdir(folder1)]
    files_in_folder2 = [os.path.splitext(f)[0] for f in os.listdir(folder2)]
    not_included_files = set(files_in_folder1) - set(files_in_folder2)
    os.makedirs(output_folder, exist_ok=True)
    for file in not_included_files:
        file_path = os.path.join(folder1, file + ".png")
        if os.path.exists(file_path):
            output_path = os.path.join(output_folder, file + ".png")
            cv2.imwrite(output_path, cv2.imread(file_path))
    print(f"Files in {folder1} not in {folder2}: {not_included_files}")

def test_image_differencing():
    clean_image = cv2.imread(str(config.TEMPLATE_DIR / "151_cleaned.jpg"))
    querry_image = cv2.imread(str(config.IMAGES_DIR / "151_cleaned.jpg"))
    diff = LabelProcessor()._calculate_image_difference(clean_image, querry_image)
    ssim_diff = LabelProcessor().calculate_ssim(clean_image, querry_image)
    print(f"Image difference: {diff}")
    print(f"SSIM: {ssim_diff:.4f}")

def transfer_roi_coordinates():
    from modules.ROIStorage import ROIStorage
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

def perform_batch_label_analysis():
    processor = LabelProcessor()
    cnt = 0
    total_anomalies = 0
    start_time = time.time()
    for image_path in (config.IMAGES_DIR / "151" / "train_augmented").glob("*.jpg"):
        results = processor.process_label(str(image_path))
        total_anomalies += int(results.is_anomaly)
        cnt += 1
    end_time = time.time()
    print(f"Total images: {cnt}, Total anomalies detected: {total_anomalies}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # find_not_included_files_in_folder(config.BASE_DIR / "parsed_data" / "151", 
    #                                     config.IMAGES_DIR / "151" / "loftware", 
    #                                     config.IMAGES_DIR / "151" / "not_included"
    #                                 )
    # annotate_template_rois("146")
    # image_folder = config.BASE_DIR / "anomaly_detection" / "train_data" / "151"/"01SL.jpg"
    # image_folder = config.IMAGES_DIR / "151" / "loftware" / "01SL.png"
    # test_augmentation(image_folder)
    # parse_data_from_loftware(30)
    # remove_markup_from_images(config.IMAGES_DIR / "151" / "not_included", config.IMAGES_DIR / "151" / "test")
    # augment_images(config.IMAGES_DIR / "151" / "test", config.IMAGES_DIR / "151" / "train_augmented")
    perform_batch_label_analysis()