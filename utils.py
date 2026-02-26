import numpy as np
import cv2

from PIL import Image, ImageDraw, ImageFont

def display_region_image(roi_name: str, image: np.ndarray, result: str):
        image = Image.fromarray(image)
        image = image.convert("RGB")

        padding_y = 10
        padding_x = 5
        
        display_text_region = f"Region: {roi_name}"
        display_text_ocr_result = f"OCR Result: {result}"
        
        try:
            font_title = ImageFont.truetype("AvenirNextWorld-Regular.ttf", 21)
            font_text = ImageFont.truetype("AvenirNextWorld-Bold.ttf", 32)
        except:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()
        
        # use text bboxes to expand canvas size if text doesn't fit
        draw = ImageDraw.Draw(image)

        title_bbox = draw.textbbox((0, 0), display_text_region, font=font_title)
        text_bbox = draw.textbbox((0, 0), display_text_ocr_result, font=font_text)

        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        max_text_width = max(title_width, text_width)
        
        required_height = title_height + text_height + image.height + 4 * padding_y

        required_width = max_text_width + padding_x
        if required_width > image.width:
            new_image = Image.new("RGB", (required_width, required_height), (255, 255, 255))
            new_image.paste(image, (0, title_height + 2 * padding_y))
        else:
            new_image = Image.new("RGB", (image.width, required_height), (255, 255, 255))
            new_image.paste(image, (0, title_height + 2 * padding_y))
        
        draw = ImageDraw.Draw(new_image)
        draw.text((padding_x, padding_y), display_text_region, fill=(255, 100, 0), font=font_title)
        draw.text((padding_x, image.height + title_height + 2 * padding_y), display_text_ocr_result, fill=(0, 100, 255), font=font_text)
        
        # Convert back to numpy for cv2.imshow
        image_np = np.array(new_image)
        cv2.imshow(roi_name, image_np)

def convert_to_greyscale(img: np.ndarray) -> np.ndarray:
    '''
    Convert an image to grayscale if it is in color. If the image is already in grayscale, return it as is
    '''
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return img

def get_template_matching_results(image: np.ndarray, template_image_path: str) -> tuple[float, tuple[int, int]]:
    '''
    Gets the template matching score and location for the given image and template. The score is the normalized least square difference.
    
    :param image: image to match against template
    :type image: np.ndarray
    :param template_image_path: Path to the template image to match
    :type template_image_path: str
    :return: A tuple containing the matching score and the top-left location of the best match
    :rtype: tuple[float, tuple[int, int]]
    '''
    image = convert_to_greyscale(image)
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    assert template_image is not None, "Template image not found or could not be loaded."

    res = cv2.matchTemplate(image,template_image,cv2.TM_SQDIFF_NORMED)
    min_val, _, min_loc, _ = cv2.minMaxLoc(res)
    return min_val, min_loc