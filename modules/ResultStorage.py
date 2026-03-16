import cv2
from utilities.utils import display_region_image
from utilities.metrics import calculate_character_error_rate
from modules.LabelResult import LabelResult
import utilities.config as config
import json

class ResultStorage:
    def __init__(self, results: LabelResult, gt_file_path: str = None):
        self.extracted_texts: dict[str, str] = {k: v.lower() for k, v in results.get_extracted_texts().items()}
        self.extracted_barcodes: dict[str, str] = results.get_extracted_barcodes()    
        self._text_region_images = results._text_region_images
        self._barcode_images = results._barcode_images
        self._run_times = results.run_times

        try:
            with open(gt_file_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
                self.gt_texts: dict[str, str] = {k: v.lower() for k, v in gt_data["text_regions"].items()}
                self.gt_barcodes: dict[str, str] = gt_data["barcode_regions"] 
        except (FileNotFoundError, TypeError):
            self.gt_texts = None
            self.gt_barcodes = None
        
        self.metrics: dict[str, str] = {}
        self.summary_text: str = f"Label {results.template_type} Summary"
    
    def save_summary_to_txt(self, output_txt_path: str):
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(self.summary_text)

    def save_extracted_texts_to_json(self, output_json_path: str):
        extracted_data = {}
        extracted_data["text_regions"]= self.extracted_texts
        extracted_data["barcode_regions"] = self.extracted_barcodes if self.extracted_barcodes else {}
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)
    
    def add_cer_metric(self):
        if not self.extracted_texts:
            print("No extracted texts to evaluate.")
            return
        if not self.gt_texts:
            print("Ground truth texts not provided. Cannot calculate CER.")
            return
        pred_texts = []
        gt_texts = []  
        for roi_name, gt_text in self.gt_texts.items():
            pred_texts.append(self.extracted_texts.get(roi_name, ""))
            gt_texts.append(gt_text)
        result = calculate_character_error_rate(pred_texts=pred_texts, gt_texts=gt_texts)
        formatted_result = f"{result*100:.1f}%"
        self.metrics['cer'] = formatted_result 
        return
    
    def add_barcode_reading_accuracy_metric(self):
        if not self.extracted_barcodes:
            print("No extracted barcodes to evaluate.")
            return
        if not self.gt_barcodes:
            print("Ground truth barcodes not provided. Cannot calculate barcode reading accuracy.")
            return
        correct_count = 0
        total_count = len(self.extracted_barcodes)
        for roi_name, barcode_text in self.extracted_barcodes.items():
            if roi_name in self.gt_barcodes and barcode_text == self.gt_barcodes[roi_name]:
                correct_count += 1
        accuracy = correct_count / total_count if total_count > 0 else 0
        formatted_accuracy = f"Correctly read {correct_count}/{total_count}; Accuracy:({accuracy*100:.1f}%)"
        self.metrics['barcode_reading_accuracy'] = formatted_accuracy
        return
    
    def add_extracted_texts_to_summary(self):
        self.summary_text += "\n\nExtracted Texts:\n"
        for roi_name, text in self.extracted_texts.items():
            self.summary_text += f"--- Region: {roi_name} ---\n"
            self.summary_text += text + "\n\n"
        if self.extracted_barcodes:
            self.summary_text += "\n\nExtracted Barcodes:\n"
            for roi_name, text in self.extracted_barcodes.items():
                self.summary_text += f"--- Barcode: {roi_name} ---\n"
                self.summary_text += text + "\n\n"
    
    def add_text_mismatches_to_summary(self):
        if not self.gt_texts:
            print("Ground truth texts not provided. Cannot add text mismatches to summary.")
            return
        self.summary_text += "\n\nText Mismatches:\n"
        for roi_name, gt_text in self.gt_texts.items():
            pred_text = self.extracted_texts.get(roi_name, "")
            if pred_text != gt_text:
                self.summary_text += f"--- Region: {roi_name} ---\n"
                self.summary_text += f"Ground Truth: {gt_text}\n"
                self.summary_text += f"Extracted Text: {pred_text}\n\n"
    
    def display_regions_with_mismatches(self):
        '''Displays region images where extracted text/barcode does not match the ground truth.'''
        if not self.gt_texts:
            print("Ground truth texts not provided. Cannot display regions with mismatches.")
            return
        for roi_name, gt_text in self.gt_texts.items():
            pred_text = self.extracted_texts.get(roi_name, "")
            if pred_text != gt_text:
                region_image = self._text_region_images.get(roi_name, None)
                if region_image is not None:
                    display_region_image(roi_name=f"Region: {roi_name}\nGT: {gt_text}",
                                         image=region_image,
                                         result=pred_text)
        for roi_name, gt_barcode in self.gt_barcodes.items():
            pred_barcode = self.extracted_barcodes.get(roi_name, "")
            if pred_barcode != gt_barcode:
                barcode_image = self._barcode_images.get(roi_name, None)
                if barcode_image is not None:
                    display_region_image(roi_name=f"Barcode: {roi_name}\nGT: {gt_barcode}",
                                         image=barcode_image,
                                         result=pred_barcode)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def add_run_times_to_summary(self):
        self.summary_text += "\nProcessing step execution times:\n"
        for step_name, run_time in self._run_times.items():
            self.summary_text += f"{step_name}: {run_time:.2f} seconds\n"
    
    def add_metric_info_to_summary(self):
        lines = [self.summary_text, "\n\nMetric summary:"]
        for metric_name, metric_value in self.metrics.items():
            lines.append(f"{metric_name}: {metric_value}")
        self.summary_text = "\n".join(lines)

    def generate_summary(self, summary_name: str, output_dir: str, verbose: bool = True, json_output: bool = True):
        print(f"Generating summary for {summary_name}...")
        self.add_cer_metric()
        self.add_barcode_reading_accuracy_metric()
        self.add_metric_info_to_summary()
        if verbose:
            self.add_text_mismatches_to_summary()
        self.add_run_times_to_summary()
        self.save_summary_to_txt(output_txt_path=f"{output_dir}/{summary_name}.txt")
        if json_output:
            self.save_extracted_texts_to_json(output_json_path=f"{output_dir}/{summary_name}.json")
    