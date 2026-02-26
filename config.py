from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

IMAGES_DIR   = BASE_DIR / "images"
ROI_DIR      = BASE_DIR / "roi_data"
GT_DIR       = BASE_DIR / "ground_truth"
RESULTS_DIR  = BASE_DIR / "results"
TEMPLATE_DIR = BASE_DIR / "templates"

TESSERACT_CFG = BASE_DIR / "tesseract_config.json"

GT_FILES = {
    "151": GT_DIR / "label_151_gt.json",
    "146": GT_DIR / "label_146_gt.json",
}

TEMPLATES = {
    "151": IMAGES_DIR / "W151_template.jpg",
    "146": IMAGES_DIR / "W146_aligned.jpg",
    "063": IMAGES_DIR / "063_page_0.jpg",
}

ROI_FILES = {
    "151": ROI_DIR / "label_151_rois.json",
    "146": ROI_DIR / "label_146_rois.json",
    "063": ROI_DIR / "label_063_rois.json",
}

LABEL_DIMENSIONS = {
    "151": (0, 0, 1102, 3099),
    "146": (0, 0, 1102, 3099),
    "063": (0, 0, 2550, 3300),
}

PADDING = 20
