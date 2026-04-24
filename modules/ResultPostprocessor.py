import cv2
from utilities.utils import display_region_image
from utilities.metrics import calculate_character_error_rate, calculate_levenshtein_distance

from modules.LabelResult import LabelResult
import utilities.config as config
import json

class ResultPostprocessor:
    def __init__(self, results: LabelResult, gt_file_path: str = None):
        self.extracted_texts: dict[str, str] = results.get_extracted_texts()
        self.extracted_barcodes: dict[str, str] = results.get_extracted_barcodes()
        self.roi_coordinates = results.roi_coordinates
        self.text_distance_threshold = 2
        self.false_barcode_threshold = 1
        self.full_label_image = results.aligned_image
        self._text_region_images = results._text_region_images
        self._barcode_images = results._barcode_images
        self._is_anomaly = results.is_anomaly
        self._run_times = results.run_times
        self.has_defect = None

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

        if not self.gt_texts and not self.gt_barcodes:
            print("No ground truth data available. Cannot conclude defect status.")
            return
        self.conclude_defect_status()
        

    
    def conclude_defect_status(self):
        if not "false_barcode_readings_n" in self.metrics:
            self.add_barcode_reading_accuracy_metric()
        
        self.conclusion_text = ""

        # anomaly detection decision
        if self._is_anomaly:
            self.conclusion_text += "Defect detected!"

        # text mismatch decision
        for roi_name, gt_text in self.gt_texts.items():
            pred_text = self.extracted_texts.get(roi_name, "")
            distance = calculate_levenshtein_distance(pred_text, gt_text)
            if distance > self.text_distance_threshold:
                self._text_mismatched = True
                self.conclusion_text = f"Text mismatch in region {roi_name} (Levenshtein distance: {distance})!"
            else:
                self._text_mismatched = False

        # barcode mismatch decision
        if (self.metrics["false_barcode_readings_n"] > 0 and self._is_anomaly==True or
           self.metrics["false_barcode_readings_n"] > self.false_barcode_threshold and self._is_anomaly==False):
            self._barcode_mismatched = True
        else: 
            self._barcode_mismatched = False

        if not self._is_anomaly and not self._text_mismatched and not self._barcode_mismatched:
            self.conclusion_text = "No defect detected."
            self.has_defect = False
            return 
        self.has_defect = True
        return
    
    def show_highlighted_mismatches(self):
        if not self.gt_texts:
            print("Ground truth texts not provided. Cannot show highlighted mismatches.")
            return
        img = self.full_label_image.copy()
        for roi_name, gt_text in self.gt_texts.items():
            pred_text = self.extracted_texts.get(roi_name, "")
            if pred_text != gt_text:
                x0, y0, x1, y1 = self.roi_coordinates["text_regions"][roi_name]
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
        for roi_name, gt_barcode in self.gt_barcodes.items():
            pred_barcode = self.extracted_barcodes.get(roi_name, "")
            if pred_barcode != gt_barcode:
                x0, y0, x1, y1 = self.roi_coordinates["barcode_regions"][roi_name]
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.imshow("Highlighted Mismatches (Red: Text, Blue: Barcode)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            

    def save_summary_to_txt(self, output_txt_path: str):
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(self.summary_text)

    def save_extracted_texts_to_json(self, output_json_path: str) -> None:
        '''Saves the extracted texts and barcodes to a JSON file for further analysis or record-keeping'''
        extracted_data = {}
        extracted_data["text_regions"]= self.extracted_texts
        extracted_data["barcode_regions"] = self.extracted_barcodes if self.extracted_barcodes else {}
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)
    
    def add_cer_metric(self) -> float:
        if not self.extracted_texts:
            print("No extracted texts to evaluate.")
            return 0.0 
        if not self.gt_texts:
            print("Ground truth texts not provided. Cannot calculate CER.")
            return 0.0
        pred_texts = []
        gt_texts = []  
        for roi_name, gt_text in self.gt_texts.items():
            pred_texts.append(self.extracted_texts.get(roi_name, ""))
            gt_texts.append(gt_text)
        result = calculate_character_error_rate(pred_texts=pred_texts, gt_texts=gt_texts)
        formatted_result = f"{result*100:.1f}%"
        self.metrics['cer'] = formatted_result 
        return result 
    
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
        self.metrics['false_barcode_readings_n'] = total_count - correct_count
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
        if self._run_times:
            self.summary_text += "\nProcessing step execution times:\n"
            for step_name, run_time in self._run_times.items():
                self.summary_text += f"{step_name}: {run_time:.2f} seconds\n"
    
    def add_metric_info_to_summary(self):
        lines = [self.summary_text, "\n\nMetric summary:"]
        for metric_name, metric_value in self.metrics.items():
            lines.append(f"{metric_name}: {metric_value}")
        self.summary_text = "\n".join(lines)
    
    def add_conclusion_to_summary(self):
        if not hasattr(self, "conclusion_text") and self.conclusion_text:
            self.conclude_defect_status()
        self.summary_text += f"\n\nCONCLUSION:\n{self.conclusion_text}"

    def generate_summary(self, summary_name: str, output_dir: str|None=None, verbose: bool = True, json_output: bool = True):
        '''
        Generates a summary of the results, including metrics and extracted texts, and saves it to a text file if output_dir is provided. Optionally saves extracted texts to a JSON file and includes detailed mismatch information in the summary if verbose is True
        '''
        print(f"Generating summary for {summary_name}...")
        self.add_cer_metric()
        self.add_barcode_reading_accuracy_metric()
        self.add_metric_info_to_summary()
        self.add_conclusion_to_summary()
        if verbose:
            self.add_text_mismatches_to_summary()
            self.add_run_times_to_summary()
        if output_dir:
            self.save_summary_to_txt(output_txt_path=f"{output_dir}/{summary_name}.txt")
        if json_output and output_dir:
            self.save_extracted_texts_to_json(output_json_path=f"{output_dir}/{summary_name}.json")
    