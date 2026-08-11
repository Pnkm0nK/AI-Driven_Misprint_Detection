import random

import numpy as np
import cv2
from deskew import determine_skew
import utilities.config as config
import opensimplex 
from pathlib import Path

def get_highlighted_anomalies(
    image: np.ndarray, 
    anomaly_map: np.ndarray, 
    threshold: float = 0.5, 
    alpha: float = 0.6
) -> np.ndarray:
    '''
    Overlays the anomaly map on the input image using a heatmap with a specified alpha transparency,
    highlighting areas where the anomaly score exceeds the specified threshold.
    
    :param image: The original input image (grayscale or color).
    :param anomaly_map: A 2D array of anomaly scores corresponding to the input image.
    :param threshold: The anomaly score threshold above which areas will be highlighted.
    :param alpha: Transparency factor for blending (0.0 = only original image, 1.0 = only heatmap).
    '''
    # 1. Ensure the base image is BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 2. Normalize and resize the anomaly map to fit the base image
    anomaly_map = anomaly_map - 0.2
    anomaly_map = np.clip(anomaly_map, 0, np.max(anomaly_map))  
    normalized_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    padded = cv2.copyMakeBorder(normalized_map,top=28, bottom=28, left=28, right=28, borderType=cv2.BORDER_CONSTANT, value=0)
    normalized_map = cv2.resize(padded, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 3. Create a continuous color heatmap from the normalized map
    heatmap = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)

    # 4. Create the binary mask based on the threshold rule
    _, binary_mask = cv2.threshold(normalized_map, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # 5. Blend the original image and the color heatmap together
    blended = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)

    # 6. Convert the 2D binary mask to 3D to match BGR shape for broadcasting
    mask_3d = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # 7. Apply the blended heatmap only where the mask is active
    highlighted_image = np.where(mask_3d == 255, blended, image)

    return highlighted_image

def convert_to_greyscale(img: np.ndarray) -> np.ndarray:
    '''
    Convert an image to grayscale if it is in color. If the image is already in grayscale, return it as is
    '''
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return img

def apply_random_augmentation(image: np.ndarray) -> np.ndarray:
    num_augmentations = random.randint(1, 3)
    augmentations = [add_binary_simplex_window, add_motion_blur, add_streaks]
    for _ in range(num_augmentations):
        augmentation = random.choice(augmentations)
        if augmentation == add_binary_simplex_window:
            color = random.choice([(255, 255, 255), (0, 0, 0)])
            image = augmentation(image, color=color, threshold=random.uniform(0.5, 0.8), scale=random.uniform(0.01, 0.05))
        elif augmentation == add_motion_blur:
            h, w = image.shape[:2]
            size_x = random.randint(10, w//4)
            size_y = random.randint(10, h//4)
            start = (random.randint(0,w-size_x), random.randint(0, h-size_y))
            end = (start[0]+size_x, start[1]+size_y)
            image = augmentation(image, start=start, end=end, feather=random.randint(10, 30), kernel_size=random.choice([9, 15, 21, 25]))
        elif augmentation == add_streaks:
            image = augmentation(image, num_streaks=random.randint(5, 15), color=(random.randint(200,255), random.randint(200,255), random.randint(200,255)), thickness=random.randint(1,4))
    return image

def add_streaks(image: np.ndarray, num_streaks: int = 5, 
                color: tuple = (255, 255, 255), thickness: int = 2) -> np.ndarray:
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)

    for _ in range(num_streaks):
        y_start = random.randint(0, h - thickness)
        
        streak_profile = np.random.rand(w).astype(np.float32)
        
        blur_size = random.choice([3, 5, 7, 11])
        streak_profile = cv2.GaussianBlur(streak_profile, (blur_size, 1), 0).flatten()
        
        streak_profile = np.clip((streak_profile - 0.3) * 2, 0, 1)
        
        for t in range(thickness):
            y = y_start + t
            if y >= h: break
            
            alpha = streak_profile
            
            if len(image.shape) == 3:
                alpha_3d = np.stack([alpha] * 3, axis=-1)
                result[y, :] = result[y, :] * (1 - alpha_3d) + color_arr * alpha_3d
            else:
                result[y, :] = result[y, :] * (1 - alpha) + color_arr[0] * alpha

    return result.astype(np.uint8)

def add_motion_blur(image: np.ndarray, start:tuple, end:tuple, feather: int=15, kernel_size: int = 15) -> np.ndarray:
    def get_angled_kernel(size: int, angle: float):
        kernel = np.zeros((size, size))
        center = size // 2
        kernel[center, :] = 1.0
        matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (size, size))
        return kernel / np.sum(kernel)
    
    x0, y0 = start
    x1, y1 = end
    h = y1 - y0
    w = x1 - x0

    kernel = get_angled_kernel(kernel_size, angle=45) 
    roi = image[y0:y1, x0:x1].copy()

    blurred = cv2.filter2D(roi, -1, kernel)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(mask, (feather, feather), (w - feather, h - feather), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), 0)
    
    if len(image.shape) == 3:
        mask = np.expand_dims(mask, axis=-1)

    res = image.copy()
    target_area = res[y0:y1, x0:x1]
    res[y0:y1, x0:x1] = (target_area * (1 - mask) + blurred * mask).astype(np.uint8)
    return res

def add_binary_simplex_window(image: np.ndarray, color: tuple = (255, 255, 255), threshold: float = 0.6, scale: float = 0.1, relative_max_window_size: float = 0.25, min_window_size: int = 50) -> np.ndarray:
    '''
    Adds a binary mask based on Simplex noise to a random window in the input image
    the mask is generated by thresholding the Simplex noise values,
    creating a binary pattern that simulates localized damage or occlusion
    the scale parameter controls the frequency of the noise, while the threshold determines
    the density of the masked area
    '''
    img_out = image.copy()
    h, w = image.shape[:2]
    opensimplex.random_seed()
    
    max_window_size = int(min(h, w) * relative_max_window_size)
    min_window_size = min_window_size if min_window_size < max_window_size else max_window_size // 2
    window_size = random.randint(min_window_size, max_window_size) 
    window_x = random.randint(0, w - window_size)
    window_y = random.randint(0, h - window_size)
    # generate coordinate arrays for the grid
    ix = np.linspace(0, window_size * scale, window_size)
    iy = np.linspace(0, window_size * scale, window_size)
    
    # generate 2D noise array (returns values in range [-1, 1])
    noise = opensimplex.noise2array(ix, iy)

    
    window_mask = (noise - noise.min()) / (noise.max() - noise.min())

    y_end = window_y + window_size
    x_end = window_x + window_size


    binary_mask = window_mask > threshold

    img_out[window_y:y_end, window_x:x_end][binary_mask] = color 
    return img_out

def extract_roi(image: np.ndarray, coordinates: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = coordinates
    roi_image = image[y0:y1, x0:x1]
    return roi_image


def deskew_image(image:np.ndarray) -> np.ndarray:
    '''
    Deskews input image using the deskew library and affine transform

    :param image: Input image to be deskewed
    :type image: np.ndarray

    :return: Deskewed image
    :rtype: np.ndarray
    '''
    skew_angle = determine_skew(image, max_angle=30)
    rot_mat = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), skew_angle, 1)
    image = cv2.warpAffine(image, rot_mat, (image.shape[1],image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return image


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

def crop_to_content(image: np.ndarray, template_type:str) -> np.ndarray:
    '''
    Crops the input image to the content area based on the template type. Uses predefined cropping coordinates for each template type.

    :param image: Input image to be cropped
    :type image: np.ndarray
    :param template_type: The type of template to determine cropping coordinates (e.g., "151", "146", "107")
    :type template_type: str

    :return: Cropped image containing only the content area
    :rtype: np.ndarray
    '''
    if template_type not in config.LABEL_DIMENSIONS:
        raise ValueError(f"Invalid template type: {template_type}. Supported types are {list(config.LABEL_DIMENSIONS.keys())}")
    
    coordinates = config.LABEL_DIMENSIONS[template_type]
    return extract_roi(image, coordinates)



def align_image(image: np.ndarray, template_image_path: str) -> np.ndarray:
    '''
    Align the input image to the template image using deskewing and template matching. Returns the aligned image.
    '''
    deskewed = deskew_image(image)

    _, loc = get_template_matching_results(deskewed, template_image_path)

    padding = config.PADDING

    x_start, y_start = loc 
    x_start -= padding
    if x_start < 0:
        deskewed = cv2.copyMakeBorder(deskewed, padding-x_start, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
        x_start = 0
    return deskewed[y_start:, x_start:]

def unsharp(image: np.ndarray, kernel_size=(1,1), sigma=2, amount=2.0, threshold=0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, threshold)
    return sharpened

def _align_using_orb_matches(matches, src_kps, dst_kps, template, original_image):
    if len(matches) < 4:
        raise ValueError(f"Not enough matches to estimate transform: {len(matches)} < 4")

    src_pts = np.float32([src_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([dst_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # No perspective change, using affine transform for deskewing and translation correction
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    if M is None:
        raise ValueError("estimateAffinePartial2D failed — not enough RANSAC inliers.")

    return cv2.warpAffine(original_image, M, (template.shape[1], template.shape[0]),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))

def orb_align(image: np.ndarray, template_type:str, n_features:int=500, max_matches: int = 100, crosscheck=True, dst_kps=None, target_descrs=None, visualize=False)-> np.ndarray:
    '''
    Align the input image to the specified template type using ORB feature matching. Returns the aligned image.
    :param image: Input image to be aligned
    :type image: np.ndarray 
    :param template_type: The type of template to align to (e.g., "151", "146", "107")
    :type template_type: str
    :param n_features: Number of ORB features to detect
    :type n_features: int
    :param max_matches: Maximum number of ORB matches to consider for alignment
    :type max_matches: int
    :param visualize: Whether to visualize the ORB matches and alignment results
    :type visualize: bool

    :return: Aligned image
    :rtype: np.ndarray
    '''
    original_image = image.copy()
    image = convert_to_greyscale(image)
    template_img_path = config.TEMPLATES[template_type]

    orb = cv2.ORB_create(nfeatures=n_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crosscheck)

    src_kps, query_descrs = orb.detectAndCompute(image, None)

    template = cv2.imread(str(template_img_path), cv2.IMREAD_GRAYSCALE)
    assert template is not None, f"Failed to load template image at {template_img_path}"

    if dst_kps is None or target_descrs is None:
        dst_kps, target_descrs = orb.detectAndCompute(template, None)

    if len(dst_kps) != n_features:
        raise ValueError("Number of keypoints in template does not match expected n_features. Ensure that the template image is processed with the same ORB settings and that dst_kps and target_descrs are correctly passed if precomputed.")

    matches = bf.match(query_descrs, target_descrs)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:min(max_matches, len(matches))]

    if visualize:
        # reload template in color for visualization only
        img_match = cv2.drawMatches(image, src_kps, template, dst_kps, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        height, width = img_match.shape[:2]
        aspect_ratio = width / height
        width = 800
        height = int(width / aspect_ratio)
        img_match = cv2.resize(img_match, (width, height))

        cv2.imshow("Matches", img_match)
        cv2.waitKey(0)
    return _align_using_orb_matches(matches, src_kps, dst_kps, template, original_image)

def orb_align_and_clasify( image: np.ndarray, n_features:int=200, max_matches: int = 50, visualize=False) -> tuple[str, np.ndarray]:
    '''
    Classify and align the input image to the best matching template using ORB feature matching.
    Returns the estimated template name and the aligned image.

    on my pc shows 0.24 seconds computation time for 30 features and 10 matches per image on average
    
    :param image: Input image to be aligned 
    :type image: np.ndarray
    :param n_features: Number of ORB features to detect
    :type n_features: int
    :param max_matches: Maximum number of ORB matches to consider for alignment
    :type max_matches: int
    :param visualize: Whether to visualize the ORB matches and alignment results
    :return: Estimated template name and aligned image in a tuple
    :rtype: tuple[str, np.ndarray]
    '''

    original_image = image.copy()
    image = convert_to_greyscale(image)

    orb = cv2.ORB_create(nfeatures=n_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    src_kps, query_descrs = orb.detectAndCompute(image, None)

    # go through templates and find best match based on total distance of top matches
    # save intermediate results to reuse in alignment step and visualization
    least_distance = float("inf") 
    estimated_template_type = None
    best_matches = []
    best_dst_kps = []

    for template_type, template_img_path in config.TEMPLATES.items():
        template = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
        assert template is not None, f"Failed to load template image at {template_img_path}"
        dst_kps, target_descrs = orb.detectAndCompute(template, None)
        # maybe try Knn match and Lowe's ratio test if too many false matches with crossCheck
        matches = bf.match(query_descrs, target_descrs)
        matches = sorted(matches, key=lambda x: x.distance)
        top = matches[:min(max_matches, len(matches))]
        total_distance = (sum(m.distance for m in top) / len(top)) if top else float("inf")

        if total_distance < least_distance:
            least_distance = total_distance
            estimated_template_type = template_type
            best_dst_kps = dst_kps
            best_matches = top

    best_template = cv2.imread(config.TEMPLATES[estimated_template_type], cv2.IMREAD_GRAYSCALE) 
    aligned_image = _align_using_orb_matches(best_matches, src_kps, best_dst_kps, best_template, original_image)

    if visualize:
        # reload template in color for visualization only
        cv2.imshow("aligned", aligned_image)
        cv2.waitKey(0)
        img_match = cv2.drawMatches(image, src_kps, best_template, best_dst_kps, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        height, width = img_match.shape[:2]
        aspect_ratio = width / height
        width = 800
        height = int(width / aspect_ratio)
        img_match = cv2.resize(img_match, (width, height))

        cv2.imshow("Matches", img_match)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return estimated_template_type, aligned_image


def save_orb_features(image: np.ndarray, output_path: str | Path, n_features: int = 500) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    image = convert_to_greyscale(image)
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)

    keypoint_array = np.array([
        (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in keypoints
    ], dtype=np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        keypoints=keypoint_array,
        descriptors=descriptors if descriptors is not None else np.empty((0, 32), dtype=np.uint8),
        n_features=np.array([n_features], dtype=np.int32)
    )

    return keypoints, descriptors

def load_orb_features(features_path: str | Path) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    data = np.load(str(features_path))
    keypoints_raw = data["keypoints"]
    descriptors = data["descriptors"]

    keypoints = [
        cv2.KeyPoint(
            x=float(kp[0]),
            y=float(kp[1]),
            size=float(kp[2]),
            angle=float(kp[3]),
            response=float(kp[4]),
            octave=int(kp[5]),
            class_id=int(kp[6])
        )
        for kp in keypoints_raw
    ]

    return keypoints, descriptors

def calculate_image_difference(image1: np.ndarray, image2: np.ndarray, visualize: bool = True) -> np.ndarray:
    # greyscale and threshold both images to reduce difference to variable info only
    # pixel intensities may differ slightly due to scanning differences

    image1 = convert_to_greyscale(image1)
    image2 = convert_to_greyscale(image2)

    # ecc may be helpful to lessen misalignment issues
    # in experiments lead to ~20% reduction in difference
    # which is not enough to make a sensitive enough detection

    _, image1 = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, image2 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Shape image 1: {image1.shape}, Shape image 2: {image2.shape}")

    diff_M = cv2.absdiff(image1, image2)
    _, diff_M = cv2.threshold(diff_M, 30, 255, cv2.THRESH_BINARY)

    # Morphological opening removes isolated noise specks without
    # expanding real differences (replaces the previous dilate)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diff_M = cv2.morphologyEx(diff_M, cv2.MORPH_OPEN, open_kernel)

    diff_sum = int(np.sum(diff_M))

    if visualize:
        print(f"Shape image 1: {image1.shape}, Shape image 2: {image2.shape}")
        for title, img in [("Image 1", image1), ("Image 2", image2), ("Difference Image", diff_M)]:
            resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(config.RESULTS_DIR / f"{title.replace(' ', '_').lower()}.jpg"), img)
            cv2.imshow(title, resized)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    return diff_sum

def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    # ensure images are greyscaled
    image1 = convert_to_greyscale(image1)
    image2 = convert_to_greyscale(image2)
    ssim_value = ssim(image1, image2)
    return ssim_value