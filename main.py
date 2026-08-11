from modules.LabelResult import LabelResult
from paddleocr import PaddleOCRVL
import numpy as np
from utilities.prepare_tesstrain_data import generate_data
import utilities.config as config
import time
import os
import cv2
from pathlib import Path
import json
from modules.LabelProcessor import LabelProcessor
import modules.OCRProcessor as ocr
from modules.PDFConverter import PDFConverter
from modules.ImageProcessor import ImageProcessor
import modules.image_processing_functions as ipf
from modules.ResultPostprocessor import ResultPostprocessor
from utilities.data_parser import parse_data_from_loftware
from utilities.ROI_draw import annotate_template_rois, annotate_label_types
import utilities.utils as utils
import utilities.plots as plots


def main():
    image_name = "W151_many_2.jpg"
    gt_name = "60SL.json"
    gt_path = config.GT_DIR / gt_name
    processor = LabelProcessor()
    results = processor.process_label(run_anomaly_detection=True, scan=str(config.IMAGES_DIR /"151" / "test_alignment_normal" / image_name))
    results = ResultPostprocessor(results, gt_path)

    # results.generate_summary(f"", str(config.RESULTS_DIR))

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
    output_dir = config.BASE_DIR / "anomaly_detection" / "train_data"
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

def align_images_in_folder(input_dir, output_dir,template_type):
    os.makedirs(output_dir, exist_ok=True)
    for image_path in input_dir.glob("*.jpg"):
        full_label_image = cv2.imread(str(image_path))
        full_label_image = ipf.orb_align(full_label_image, template_type=template_type, n_features=200, max_matches=50, visualize=False)
        # Save the aligned image
        cv2.imwrite(str(output_dir / f"{image_path.stem}.jpg"), full_label_image)

def save_orb_aligned_image():
    image_name = "W151_2_1.jpg"
    image_path = str(config.IMAGES_DIR / image_name)
    full_label_image = cv2.imread(image_path)  
    template_type, full_label_image = ipf.orb_align_and_classify(full_label_image, visualize=True)

    # specialize image processor to the template

    _,_,img_w,img_h = config.LABEL_DIMENSIONS[template_type]
    cropped_image = full_label_image[0:img_h, 0:img_w] 
    cv2.imwrite(str(config.IMAGES_DIR / f"W{template_type}_template.jpg"), cropped_image)

def remove_variable_info_from_template(template_type):
    processor = LabelProcessor()
    results = processor.process_label(str(config.TEMPLATES[template_type]), run_ocr=False, read_barcodes=False, run_anomaly_detection=False)
    cleaned_image = processor._remove_variable_info_from_image(results.aligned_image, results.roi_coordinates)
    cv2.imwrite(str(config.TEMPLATE_DIR / f"{template_type}_cleaned.jpg"), cleaned_image)

def remove_variable_info_from_aligned_image(image_name, template_type):
    processor = LabelProcessor()
    results = processor.process_label(str(config.IMAGES_DIR / image_name), run_ocr=False, read_barcodes=False, run_anomaly_detection=False)
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
    diff = ipf.calculate_image_difference(clean_image, querry_image, visualize=True)
    print(f"Image difference: {diff}")

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

def get_label_gt_batch(input_dir,output_dir):
    processor = LabelProcessor()
    cnt = 0
    start_time = time.time()
    for image_path in (input_dir).glob("*.jpg"):
        results = processor.process_label(str(image_path), postprocess_text=False)
        results = ResultPostprocessor(results)
        json_gt_path = output_dir / f"{image_path.stem}_gt.json"
        results.save_extracted_texts_to_json(json_gt_path)
        cnt += 1
    end_time = time.time()
    print(f"Total images: {cnt}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")

def perform_batch_label_analysis():
    processor = LabelProcessor()
    cnt = 0
    total_anomalies = 0
    start_time = time.time()
    for image_path in (config.IMAGES_DIR / "151" / "test").glob("*.jpg"):
        results = processor.process_label(str(image_path))
        json_gt_path = config.RESULTS_DIR / f"{image_path.stem}.json"
        results = ResultPostprocessor(results, str(json_gt_path))
        if results.has_defect:
            total_anomalies += 1
            results.generate_summary(image_path.stem)
            
            print(f"Anomaly detected in {image_path.name}!")
            print(results.summary_text)
        cnt += 1
    end_time = time.time()
    print(f"Total images: {cnt}, Total anomalies detected: {total_anomalies}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")

def ocr_trials(engine="tesseract"):
    trial_results_dir = config.RESULTS_DIR / f"ocr_trials_{engine}"
    if engine == "paddleocr":
        engine = ocr.PaddleOCRProcessor()
    elif engine == "easyocr":
        engine = ocr.EasyOCRProcessor()
    else:
        engine = ocr.TesserocrProcessor() 
    processor = LabelProcessor(ocr_processor=engine)
    test_dir = config.IMAGES_DIR / "151" / "test_ocr"
    gt_dir =  config.BASE_DIR / "ground_truth"
    trial_results_dir.mkdir(exist_ok=True)
    cer_array = []
    time_array=[]
    img_cnt = 0
    total_time = 0.0
    for image_path in test_dir.glob("*.jpg"):
        results = processor.process_label(str(image_path),
                                          run_anomaly_detection=False,
                                          read_barcodes=False,
                                          )
        time_taken = results.run_times["OCR text extraction"]
        print(f"Extracted text for {image_path.name}:")
        gt_path = gt_dir / f"{image_path.stem}.json"
        result_processor = ResultPostprocessor(results, gt_path)
        # result_processor.display_regions_with_mismatches()
        cer = result_processor.add_cer_metric() * 100
        result_processor.generate_summary(image_path.stem, trial_results_dir, json_output=False)
        print(f"Character Error Rate for {image_path.name}: {cer:.2f}%")
        cer_array.append(cer)
        img_cnt += 1
        total_time += time_taken
        time_array.append(time_taken)
    time_array = np.array(time_array)
    avg_time = np.mean(time_array) if len(time_array) > 0 else 0.0
    min_time = np.min(time_array) if len(time_array) > 0 else 0.0
    max_time = np.max(time_array) if len(time_array) > 0 else 0.0
    avg_cer = sum(cer_array) / len(cer_array) if cer_array else 0.0
    max_cer = max(cer_array) if cer_array else 0.0
    min_cer = min(cer_array) if cer_array else 0.0
    res = {"avg_cer": avg_cer, "max_cer": max_cer, "min_cer": min_cer, "total_time": total_time, "avg_time": avg_time, "min_time": min_time, "max_time": max_time, "cer_array": cer_array}
    with open(trial_results_dir / "results.json", 'w') as f:
        json.dump(res, f, indent=4)


    print(f"\n \n OCR Trials completed for {img_cnt} images.")
    print(f"Average Character Error Rate across {img_cnt} images: {avg_cer:.2f}%")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average processing time per image: {total_time/img_cnt:.2f} seconds")

def save_otsu_thresholded_image(image_path, output_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(output_path), otsu_thresh)

def check_paddleocr_reader():
    reader = ocr.PaddleOCRProcessor() 
    test_image_path = config.IMAGES_DIR / "151" / "test_loftware_normal" / "01SF.jpg"
    gt_file_path = config.GT_DIR / "01SF.json"
    processor = LabelProcessor(ocr_processor=reader)
    result: LabelResult  = processor.process_label(str(test_image_path), postprocess_text=True, run_anomaly_detection=False, read_barcodes=False)
    result_processor = ResultPostprocessor(result, gt_file_path=str(gt_file_path))
    result_processor.add_run_times_to_summary()
    result_processor.generate_summary(f"results_{str(test_image_path.stem)}", str(config.RESULTS_DIR), json_output=False)

def generate_gts_ocr():
    reader = ocr.PaddleOCRProcessor() 
    test_images_path = config.IMAGES_DIR / "151" / "unprocessed_gts"
    processor = LabelProcessor(ocr_processor=reader)
    for test_image_path in test_images_path.glob("*.jpg"):
        gt_file = config.GT_DIR / f"{test_image_path.stem}.json"
        result: LabelResult = processor.process_label(str(test_image_path), run_ocr=True, postprocess_text=True, run_anomaly_detection=False, read_barcodes=True)
        result_processor = ResultPostprocessor(result)
        result_processor.save_extracted_texts_to_json(str(gt_file))
    

def check_speed_paddle():
    ocr = PaddleOCRVL(
        use_chart_recognition=False,
        use_seal_recognition=False, 
        use_doc_orientation_classify=False,
        use_doc_unwarping=False
    )
    test_image_path = config.IMAGES_DIR / "151" / "test_loftware_normal" / "01SF.jpg"
    img = cv2.imread(str(test_image_path))
    timer_start = time.time()
    result = ocr.predict(
        [img],
        return_json=False,
        return_markdown=False,
        max_new_tokens=64 
    )
    print(f"PaddleOCRVL processing time: {time.time() - timer_start:.2f} seconds")

def assess_alignment_performance():
    anomaly_samples = config.IMAGES_DIR / "151" / "test_alignment_anomalous"
    normal_samples = config.IMAGES_DIR / "151" / "test_alignment_normal"
    template_type = "151"

    parameter_sets = [
        {"n_features": 125, "max_matches": 25, "crosscheck":True},
        {"n_features": 250, "max_matches": 50, "crosscheck":True},
        {"n_features": 250, "max_matches": 50, "crosscheck":False},
        {"n_features": 500, "max_matches": 100, "crosscheck":True},
        {"n_features": 500, "max_matches": 100, "crosscheck":False},]
    total_stats = []
    for params in parameter_sets:
        template_img_path = config.TEMPLATES[template_type]
        template_img = cv2.imread(str(template_img_path), cv2.IMREAD_GRAYSCALE)

        orb = cv2.ORB_create(nfeatures=params['n_features'])
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=params['crosscheck'])
        dst_kps, target_descrs = orb.detectAndCompute(template_img, None)

        times = []

        for image_path in anomaly_samples.glob("*.jpg"):
            img = cv2.imread(str(image_path))

            timer_start = time.time()
            aligned_img = ipf.orb_align(img, template_type="151", dst_kps=dst_kps, target_descrs=target_descrs, **params)
            times.append(time.time() - timer_start)
            aligned_img = cv2.resize(aligned_img, (500,1400))
            print(f"Anomalous image: {image_path.name}, Params: {params}")
        anomalous_n = len(times)
        cv2.destroyAllWindows()

    
        for image_path in normal_samples.glob("*.jpg"):
            img = cv2.imread(str(image_path))

            timer_start = time.time()
            aligned_img = ipf.orb_align(img, template_type="151", dst_kps=dst_kps, target_descrs=target_descrs, **params)
            times.append(time.time() - timer_start)
            aligned_img = cv2.resize(aligned_img, (500,1400))
            print(f"Normal image: {image_path.name}, Params: {params}")
        cv2.destroyAllWindows()
        avg_time = sum(times) / len(times) if times else 0
        print(f"Total images processed: {len(times)}, anomalous: {anomalous_n}, normal: {len(times)-anomalous_n}")
        print(f"Average alignment time for params {params}: {avg_time:.2f} seconds")
        print("Max alignment time: {:.2f} seconds".format(max(times)))
        print("Min alignment time: {:.2f} seconds".format(min(times)))
        total_stats.append({"params": params, "avg_time": avg_time, "max_time": max(times), "min_time": min(times), "total_images": len(times), "anomalous_images": anomalous_n, "normal_images": len(times)-anomalous_n})
    print("\n=== Alignment Performance Summary ===")
    for stats in total_stats:
        print(f"Params: {stats['params']}, Avg Time: {stats['avg_time']*1000:.2f} ms, Max Time: {stats['max_time']*1000:.2f} ms, Min Time: {stats['min_time']*1000:.2f} ms, Total Images: {stats['total_images']}, Anomalous Images: {stats['anomalous_images']}, Normal Images: {stats['normal_images']}")

def assess_barcode_reading_performance():
    processor = LabelProcessor()
    test_dir = config.IMAGES_DIR / "151" / "test_ocr"
    gt_dir =  config.BASE_DIR / "ground_truth"
    output_dir = config.RESULTS_DIR / "barcode_metrics"
    total_barcodes = 0
    correct_barcodes = 0
    total_time = 0
    max_time = 0
    min_time = float('inf')
    for test_image_path in test_dir.glob("*.jpg"):
        results = processor.process_label(str(test_image_path), run_ocr=False, read_barcodes=True, run_anomaly_detection=False, postprocess_text=False)
        time_taken = results.run_times.get("Barcode reading", 0.0)
        total_time += time_taken
        max_time = max(max_time, time_taken)
        min_time = min(min_time, time_taken)
        gt_path = gt_dir / f"{test_image_path.stem}.json"
        result_processor = ResultPostprocessor(results, gt_path)
        correct_barcodes += result_processor.metrics["correct_barcode_readings_n"]
        total_barcodes += result_processor.metrics["total_barcodes"]
    accuracy = (correct_barcodes / total_barcodes) if total_barcodes > 0 else 0.0
    print(f"Total barcodes: {total_barcodes}, Correctly read: {correct_barcodes}, Accuracy: {accuracy*100:.2f}%, Total processing time: {total_time*1000:.2f} ms, Average time per image: {total_time*1000/len(list(test_dir.glob('*.jpg'))):.2f} ms, Max time: {max_time*1000:.2f} ms, Min time: {min_time*1000:.2f} ms")

def e2e_system_test():
    processor = LabelProcessor()
    normal_dir = config.IMAGES_DIR / "151" / "test_alignment_normal"
    anomaly_dir = config.IMAGES_DIR / "151" / "test_alignment_anomalous"
    gt_dir = config.BASE_DIR / "ground_truth"
    wrong_gt_dir = config.BASE_DIR / "wrong_ground_truth"

    from collections import defaultdict
    stats = defaultdict(int)
    time_stats = defaultdict(list)

    # Clean set lookup containing raw string names (e.g., 'label_01.json')
    wrong_gt_filenames = {p.name for p in wrong_gt_dir.glob("*.json")}

    # --- TRACK 1: EVALUATING NORMAL IMAGES (Expected to be clean) ---
    for normal_image_path in normal_dir.glob("*.jpg"):
        stats["total_normal_images"] += 1
        stem = normal_image_path.stem
        
        label_result = processor.process_label(str(normal_image_path), template_type="151")
        
        total_label_time = 0.0
        for step_name, run_time in label_result.run_times.items():
            time_stats[step_name].append(run_time)
            total_label_time += run_time
        time_stats["TOTAL_PIPELINE_EXECUTION"].append(total_label_time)

        is_wrong_gt = f"{stem}.json" in wrong_gt_filenames
        gt_path = wrong_gt_dir / f"{stem}.json" if is_wrong_gt else gt_dir / f"{stem}.json"
        
        result_processor = ResultPostprocessor(label_result, gt_path)

        visual_anomaly = getattr(label_result, "is_anomaly", False)
        text_anomaly = result_processor._text_mismatched or result_processor._barcode_mismatched

        if is_wrong_gt:
            stats["normal_images_with_simulated_erp_drift"] += 1
            
            # Since text is purposefully wrong, text detection is a SUCCESS (True Positive)
            if text_anomaly:
                stats["true_positives"] += 1
                if visual_anomaly:
                    stats["tp_dual_trigger_normal_wrong_gt"] += 1
                else:
                    stats["tp_caught_by_ocr_only_normal_wrong_gt"] += 1
            else:
                # OCR failed to see the injected text drift
                stats["false_negatives"] += 1
                stats["fn_ocr_missed_text_drift"] += 1
                if not visual_anomaly:
                    stats["fn_missed_by_both_systems"] += 1
                
                result_processor.generate_summary(f"{stem}_normal_wrong_gt_MISSED_BY_OCR", str(config.RESULTS_DIR), json_output=False)
        else:
            # Baseline Normal Label: Neither track should fire
            if visual_anomaly or text_anomaly:
                stats["false_positives"] += 1
                if visual_anomaly and text_anomaly:
                    stats["fp_caused_by_both"] += 1
                elif visual_anomaly:
                    stats["fp_caused_by_visual_model_only"] += 1
                elif text_anomaly:
                    stats["fp_caused_by_ocr_only"] += 1
                
                result_processor.generate_summary(f"{stem}_normal_correct_gt_FALSE_ALARM", str(config.RESULTS_DIR), json_output=False)
            else:
                stats["true_negatives"] += 1

    # --- TRACK 2: EVALUATING ANOMALOUS IMAGES (Expected to have physical defects) ---
    for anomaly_image_path in anomaly_dir.glob("*.jpg"):
        stats["total_anomalous_images"] += 1
        stem = anomaly_image_path.stem
        
        label_result = processor.process_label(str(anomaly_image_path), template_type="151")
        
        total_label_time = 0.0
        for step_name, run_time in label_result.run_times.items():
            time_stats[step_name].append(run_time)
            total_label_time += run_time
        time_stats["TOTAL_PIPELINE_EXECUTION"].append(total_label_time)

        is_wrong_gt = f"{stem}.json" in wrong_gt_filenames
        gt_path = wrong_gt_dir / f"{stem}.json" if is_wrong_gt else gt_dir / f"{stem}.json"
            
        result_processor = ResultPostprocessor(label_result, gt_path)

        visual_anomaly = getattr(label_result, "is_anomaly", False)
        text_anomaly = result_processor._text_mismatched or result_processor._barcode_mismatched

        if is_wrong_gt:
            stats["anomalous_images_with_simulated_erp_drift"] += 1
            
            # This is a DUAL defect image (Physical Anomaly + Injected Text Mismatch)
            if visual_anomaly or text_anomaly:
                stats["true_positives"] += 1
                if visual_anomaly and text_anomaly:
                    stats["tp_dual_defect_caught_by_both"] += 1
                elif visual_anomaly:
                    stats["tp_dual_defect_caught_by_visual_only"] += 1
                elif text_anomaly:
                    stats["tp_dual_defect_caught_by_ocr_only"] += 1
            else:
                stats["false_negatives"] += 1
                stats["fn_missed_by_both_systems"] += 1
                result_processor.generate_summary(f"{stem}_anomalous_wrong_gt_COMPLETE_MISS", str(config.RESULTS_DIR), json_output=False)
        else:
            # Standard Anomalous Label: Defect is purely physical (Visual track should catch it)
            if visual_anomaly or text_anomaly:
                stats["true_positives"] += 1
                if visual_anomaly and text_anomaly:
                    # Visual correctly caught it, but OCR generated an accidental text false alarm
                    stats["tp_physical_with_ocr_false_alarm"] += 1
                elif visual_anomaly:
                    stats["tp_physical_caught_by_visual_only"] += 1
                elif text_anomaly:
                    # Critical Edge Case: Visual model missed physical anomaly, but OCR saved it via a false alarm reading!
                    stats["tp_physical_saved_by_ocr_error"] += 1
            else:
                stats["false_negatives"] += 1
                stats["fn_visual_missed_physical_defect"] += 1
                stats["fn_missed_by_both_systems"] += 1
                result_processor.generate_summary(f"{stem}_anomalous_correct_gt_MISSED_PHYSICAL_DEFECT", str(config.RESULTS_DIR), json_output=False)

    # --- METRIC CALCULATIONS ---
    tp, fp, tn, fn = stats["true_positives"], stats["false_positives"], stats["true_negatives"], stats["false_negatives"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n=== End-to-End Cascade Performance Metrics ===")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"System Precision : {precision:.3f} | Recall (Sensitivity): {recall:.3f}")
    print(f"Specificity      : {specificity:.3f} | Global System Accuracy: {accuracy:.3f}")
    print(f"Combined F1-Score: {f1:.3f}")

    print("\n=== DETAILED TRUE POSITIVE BREAKDOWN (WHY DEFECTS WERE CAUGHT) ===")
    print(f"Total True Positives (TP): {tp}")
    print("  [On Simulated ERP Drift Normal Labels]:")
    print(f"    --> Successfully caught by the ERP mismatch: {stats['tp_caught_by_ocr_only_normal_wrong_gt']}")
    print(f"    --> Caught by OCR with a concurrent visual model hit : {stats['tp_dual_trigger_normal_wrong_gt']}")
    print("  [On Purely Physical Anomaly Labels]:")
    print(f"    --> Successfully caught strictly by Visual Model     : {stats['tp_physical_caught_by_visual_only']}")
    print(f"    --> Caught by Visual Model but with an OCR false alarm: {stats['tp_physical_with_ocr_false_alarm']}")
    print(f"    --> Visual model missed it, but saved by OCR mistake  : {stats['tp_physical_saved_by_ocr_error']}")
    print("  [On Dual Defect Labels (Physical + ERP Drift)]:")
    print(f"    --> Completely verified by both tracks simultaneously : {stats['tp_dual_defect_caught_by_both']}")
    print(f"    --> Slipped past OCR text but caught by Visual Model   : {stats['tp_dual_defect_caught_by_visual_only']}")
    print(f"    --> Slipped past Visual Model but caught by OCR text   : {stats['tp_dual_defect_caught_by_ocr_only']}")

    print("\n=== DETAILED FALSE POSITIVE BREAKDOWN (PURE FALSE ALARMS) ===")
    print(f"Total False Positives (FP): {fp}")
    print(f"  --> Triggered strictly by OCR parsing noise on clean labels  : {stats['fp_caused_by_ocr_only']}")
    print(f"  --> Triggered strictly by Visual Model score drift on clean : {stats['fp_caused_by_visual_model_only']}")
    print(f"  --> Concurrent phantom defect triggered across both tracks : {stats['fp_caused_by_both']}")

    print("\n=== CRITICAL MISSES BREAKDOWN (FALSE NEGATIVES) ===")
    print(f"Total Undetected System Defects (FN): {fn}")
    print(f"  --> OCR missed injected text discrepancies completely        : {stats['fn_ocr_missed_text_drift']}")
    print(f"  --> Visual model missed physical anomalies completely        : {stats['fn_visual_missed_physical_defect']}")
    print(f"  --> Absolute Blind Spot: Slipped past both systems entirely : {stats['fn_missed_by_both_systems']}")

    if time_stats:
        print("\n=== Pipeline Execution Timing Performance ===")
        sorted_steps = sorted(time_stats.keys(), key=lambda k: (k == "TOTAL_PIPELINE_EXECUTION", k))
        for step_name in sorted_steps:
            values = time_stats[step_name]
            if not values: continue
            avg_time = (sum(values) / len(values)) * 1000
            min_time = min(values) * 1000
            max_time = max(values) * 1000
            prefix = ">>> " if step_name == "TOTAL_PIPELINE_EXECUTION" else "  "
            print(f"{prefix}[{step_name:<25}] Avg: {avg_time:7.2f} ms | Min: {min_time:7.2f} ms | Max: {max_time:7.2f} ms (n={len(values)})")
        




if __name__ == "__main__":
    main()
    # find_not_included_files_in_folder(config.BASE_DIR / "parsed_data" / "151", 
    #                                     config.IMAGES_DIR / "151" / "loftware", 
    #                                     config.IMAGES_DIR / "151" / "not_included"
    #                                 )
    # annotate_template_rois("151")
    # image_folder = config.BASE_DIR / "anomaly_detection" / "train_data" / "151"/"01SL.jpg"
    # image_folder = config.IMAGES_DIR / "151" / "loftware" / "01SL.png"
    # test_augmentation(image_folder)
    # parse_data_from_loftware(30)
    # remove_markup_from_images(config.IMAGES_DIR / "151" / "not_included", config.IMAGES_DIR / "151" / "test")
    # augment_images(config.IMAGES_DIR / "151" / "test_loftware_normal", config.BASE_DIR / "anomaly_detection" / "anomalous_data" / "151" / "new_test_augmented")
    # input_dir = config.BASE_DIR / "anomaly_detection" / "train_data" / "151"
    # save_otsu_thresholded_image(config.IMAGES_DIR / "151" / "test" / "01SF.jpg", config.RESULTS_DIR / "otsu_showcase.jpg")
    # remove_variable_info_from_aligned_image("W151_many_3.jpg", "151")
    # remove_variable_info_from_template("151")
    # test_image_differencing()
    # plots.generate_roc_curves()

    # output_path = config.BASE_DIR / "models" / "orb_features"

    # image = cv2.imread(str(config.TEMPLATE_DIR / "W151_template.jpg"))
    # get_label_gt_batch(input_dir, config.BASE_DIR / "train_gts")   
    # ocr_trials(engine="paddleocr")
    # plots.generate_roc_curves_cv()
    # plots.generate_ocr_violin_plot()
    # assess_barcode_reading_performance()
    # generate_gts_ocr()
    # ipf.save_orb_features(image=image, output_path=output_path, n_features=500)

    # assess_alignment_performance()
    # check_annotations_with_paddleocr()
    # generate_data()
    # e2e_system_test()
    # annotate_template_rois("151")