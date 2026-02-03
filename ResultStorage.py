from metrics import calculate_character_error_rate
import json

class ResultStorage:

    def __init__(self, extracted_texts: dict[str, str], gt_texts_path: str = None):
        self.extracted_texts = extracted_texts
        try:
            with open(gt_texts_path, 'r', encoding='utf-8') as f:
                gt_texts = json.load(f)
        except (FileNotFoundError, TypeError):
            gt_texts = None
        
        self.gt_texts = gt_texts
        self.metrics: dict[str, str] = {}
        self.summary_text: str = ""
    
    def save_summary_to_txt(self, output_txt_path: str):
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(self.summary_text)

    def save_extracted_texts_to_json(self, output_json_path: str):
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.extracted_texts, f, indent=4, ensure_ascii=False)
    
    def add_cer_metric(self):
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
    
    def add_extracted_texts_to_summary(self):
        self.summary_text += "\n\nExtracted Texts:\n"
        for roi_name, text in self.extracted_texts.items():
            self.summary_text += f"--- Region: {roi_name} ---\n"
            self.summary_text += text + "\n\n"
    
    def add_metric_info_to_summary(self):
        lines = ["OCR results metric summary:"]
        for metric_name, metric_value in self.metrics.items():
            lines.append(f"{metric_name}: {metric_value}")
        self.summary_text = "\n".join(lines)