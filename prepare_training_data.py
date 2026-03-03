import json
import os
import random
import subprocess
import sys
from pathlib import Path
from LabelProcessor import LabelProcessor
from ROIStorage import ROIStorage

import cv2
import dotenv
import numpy as np

import config

# -----config-----
IMAGES_TO_PREPARE = ["W151.jpg", "W146.jpg"]
GT_NAMES = ["W151_gt_tesstrain.json", "W146_gt_tessdata.json"]


EVAL_SPLIT   = 0.1  
RANDOM_SEED  = 333 
LANG         = "eng"
# ----------------


def load_single_line_gt(gt_json_path:str) -> dict[str, str]:
    """
    Loads text_regions from a GT JSON file.
    Returns only entries whose value contains no newline character.
    """
    try:
        with open(gt_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"GT JSON file not found: {gt_json_path}")

    single_line = {k: v for k, v in data["text_regions"].items() if "\n" not in v}
    skipped = len(data["text_regions"]) - len(single_line)
    if skipped:
        print(f"Skipped {skipped} multi-line entries from gt file: {Path(gt_json_path).name}")
    return single_line


def save_tiff_and_gt(image: np.ndarray, gt_text: str, out_stem: Path) -> Path:
    """
    Saves the preprocessed ROI image as a TIFF, writes the paired .gt.txt,
    and generates the WordStr .box file required by `tesseract lstm.train`.
    Returns the path to the saved TIFF.
    """
    tif_path = Path(str(out_stem) + ".tif")
    gt_path  = Path(str(out_stem) + ".gt.txt")
    cv2.imwrite(str(tif_path), image)
    gt_path.write_text(gt_text, encoding="utf-8")
    return tif_path

def write_list_file(path: Path, lstmf_paths: list[Path]) -> None:
    """Writes a Tesseract list file with one absolute path per line."""
    path.write_text(
        "\n".join(str(p.resolve()) for p in lstmf_paths) + "\n",
        encoding="utf-8",
    )


def generate_data() -> None:
    dotenv.load_dotenv()
    TRAINING_DIR = Path(os.getenv("TRAINING_DIR"))
    images_to_prepare: list[Path] = [config.IMAGES_DIR / img for img in IMAGES_TO_PREPARE]
    gt_paths: list[Path] = [config.GT_DIR / gt for gt in GT_NAMES]

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    total_skipped = 0
    processor = LabelProcessor()

    for image_path, gt_path in zip(images_to_prepare, gt_paths):
        template_type, image = processor.image_processor.orb_align_and_clasify(cv2.imread(str(image_path))) 
        processor.image_processor = processor.image_processor.get_suitable_image_processor(template_type)

        processor.full_label_image = image

        print(f"\nProcessing: {image_path.name}  (template type '{template_type}')")

        try:
            gt_map = load_single_line_gt(gt_json_path=str(gt_path))
        except FileNotFoundError as exc:
            print(f"Skipping: {exc}")
        
        roi_storage = ROIStorage(img_h=image.shape[0],
                                  img_w=image.shape[1],
                                    template_type=template_type)
        text_rois = roi_storage.load_roi_json_data().get("text_regions", {})

        region_images = processor._extract_preprocessed_region_images(text_rois)


        print(f"GT entries (single-line): {len(gt_map)}  |  "
              f"ROIs in JSON: {len(text_rois)}")

        for roi_name, gt_text in gt_map.items():
            if roi_name not in text_rois:
                print(f"Skipped {roi_name}: not in ROI data")
                total_skipped += 1
                continue
            roi_crop = region_images.get(roi_name, None)

            if roi_crop is None or roi_crop.size == 0:
                print(f"Skipped empty crop for {roi_name}")
                total_skipped += 1
                continue


            safe_roi = roi_name.replace(" ", "_")
            stem_name = f"{LANG}.{safe_roi}_{sample_idx}"
            out_stem  = TRAINING_DIR / stem_name

            tif_path = save_tiff_and_gt(roi_crop, gt_text, out_stem)
            print(f"  + {stem_name}  |  {repr(gt_text)}")

            sample_idx += 1

    print(f"\n{'─'*60}")
    print(f"Done.")
    print(f"  Total samples  : {sample_idx}")
    print(f"  Skipped/failed : {total_skipped}")

if __name__ == "__main__":
    generate_data()
