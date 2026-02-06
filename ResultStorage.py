from metrics import calculate_character_error_rate
import json

class ResultStorage:

    def __init__(self, extracted_texts: dict[str, str],
                gt_file_path: str = None,
                extracted_barcodes: dict[str, str]| None = None):
        self.extracted_texts = extracted_texts
        self.extracted_barcodes = extracted_barcodes    
        try:
            with open(gt_file_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
                self.gt_texts = gt_data["text_regions"]
                self.gt_barcodes = gt_data["barcode_regions"] 
        except (FileNotFoundError, TypeError):
            self.gt_texts = None
            self.gt_barcodes = None
        
        self.metrics: dict[str, str] = {}
        self.summary_text: str = ""
    
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
    
    def add_metric_info_to_summary(self):
        lines = ["Metric summary:"]
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
        self.save_summary_to_txt(output_txt_path=f"{output_dir}/{summary_name}.txt")
        if json_output:
            self.save_extracted_texts_to_json(output_json_path=f"{output_dir}/{summary_name}.json")
    