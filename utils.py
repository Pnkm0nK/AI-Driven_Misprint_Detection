import numpy as np
import cv2

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