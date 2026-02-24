import config
from LabelProcessor import LabelProcessor
from ResultStorage import ResultStorage

if __name__ == "__main__":
    image_name = "W151.jpg"
    processor = LabelProcessor()
    results = processor.process_label(str(config.IMAGES_DIR / image_name))
    results.display_all_region_images()
    results = ResultStorage(results)
    results.generate_summary(f"W151_result", str(config.RESULTS_DIR))