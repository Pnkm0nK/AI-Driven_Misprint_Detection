import os

import cv2
import subprocess
import shutil
import sys
from pathlib import Path
from modules.ROIStorage import ROIStorage, ROICollection
import json
import utilities.config as config

'''
Script for creating and editing ROIS using labelme lib
Provides conversion to and from labelme JSON format
'''


def _get_labelme_command() -> list[str]:
    '''
    Resolve a working labelme command, in case labelme is not on the PATH
    '''
    if shutil.which("labelme"):
        return ["labelme"]
    return [sys.executable, "-m", "labelme"]

def roi_collection_to_labelme(
    roi_collection: ROICollection,
    image_path: str | Path,
    img_w: int,
    img_h: int,
) -> dict:
    '''
    Convert denormalized pixel ROICollection to a labelme
    annotation dict. The returned dict can be written directly to a labelme JSON file.

    :param roi_collection: Denormalized ROI collection as returned by
                           ROIStorage.load_roi_json_data()
    :param image_path: Path to the image file (written into labelme JSON for
                       reference; labelme does NOT need to find it at runtime
                       when imageData is null)
    :param img_w: Image width in pixels
    :param img_h: Image height in pixels
    :return: labelme annotation dict
    '''
    shapes = []
    for category, rois in roi_collection.items():
        for roi_name, coords in rois.items():
            x0, y0, x1, y1 = coords
            shapes.append({
                "label": f"{category}/{roi_name}",
                "points": [[float(x0), float(y0)], [float(x1), float(y1)]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            })

    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": str(Path(image_path).name),
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def save_labelme_json(
    roi_collection: ROICollection,
    image_path: str | Path,
    img_w: int,
    img_h: int,
    output_path: str | Path,
) -> None:
    '''
    Convert ROICollection to labelme JSON and write to output_path.
    '''
    data = roi_collection_to_labelme(roi_collection, image_path, img_w, img_h)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved labelme JSON → {output_path}")


def labelme_to_roi(labelme_json_path: str | Path) -> ROICollection:
    '''
    Load a labelme JSON file and convert it back to a denormalized
    ROICollection pixel coords

    Labels are expected in the form category/roi_name. Any shape that
    does not follow this convention is placed under miscellaneous

    :param labelme_json_path: Path to the labelme JSON file
    :return: Denormalized ROICollection
    '''
    with open(labelme_json_path, "r") as f:
        data = json.load(f)

    roi_collection: ROICollection = {}

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue  # skip polygons or other types added labelme

        label: str = shape["label"]

        if "/" in label: # / is a way to categorize ROIs in labelme
            category, roi_name = label.split("/", maxsplit=1)
        else:
            category, roi_name = "miscellaneous", label

        (x0, y0), (x1, y1) = shape["points"]
        coords = (int(x0), int(y0), int(x1), int(y1))

        roi_collection.setdefault(category, {})[roi_name] = coords

    return roi_collection

def edit_rois_in_labelme(
    roi_collection: ROICollection,
    image_path: str | Path,
    img_w: int,
    img_h: int,
) -> ROICollection:
    '''
    Open labelme with the current ROI annotations pre-loaded, wait for the
    user to finish editing, then return the updated ROICollection

    :param roi_collection: Current denormalized ROICollection
    :param image_path: Path to the label image to annotate
    :param img_w: Image width in pixels
    :param img_h: Image height in pixels
    :return: Updated denormalized ROICollection after the user closes labelme
    '''
    image_path = Path(image_path)

    # temporary labelme JSON
    labelme_json_path = image_path.with_suffix(".json")

    save_labelme_json(roi_collection, image_path, img_w, img_h, labelme_json_path)

    print(f"Launching labelme for {image_path.name} …")
    print("Save and close labelme when done")

    labelme_command = _get_labelme_command()
    subprocess.run(
        [
            *labelme_command,
            str(image_path),
            "--labels",
            str(config.ROI_DIR / "labels.csv"),
            "--nodata",
        ],
        check=True,
    )

    if not labelme_json_path.exists():
        print("No labelme JSON found after editing – returning original ROIs.")
        return roi_collection

    updated = labelme_to_roi(labelme_json_path)
    os.remove(labelme_json_path)  # clean up temporary JSON file
    print(f"Loaded {sum(len(v) for v in updated.values())} ROIs from labelme.")
    return updated

def annotate_template_rois(template_type: str):
    image_path = config.TEMPLATES[template_type]

    template_image = cv2.imread(str(image_path))
    roi_storage = ROIStorage(img_h=template_image.shape[0],
                             img_w=template_image.shape[1],
                             template_type=template_type)
    roi_coordinates = roi_storage.load_roi_json_data()

    updated_rois = edit_rois_in_labelme(
        roi_coordinates,
        image_path,
        img_w=template_image.shape[1],
        img_h=template_image.shape[0],
    )
    roi_storage.save_roi_json_data(updated_rois)

def annotate_label_types(data_folder_path, output_path):
    labelme_command = _get_labelme_command()
    subprocess.run([*labelme_command, str(data_folder_path),
                    "--output", str(output_path),
                    "--nodata"],
                      check=True)