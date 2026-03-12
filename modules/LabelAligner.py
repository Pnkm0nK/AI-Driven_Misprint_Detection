from sklearn import TransformerMixin, BaseEstimator 
import config
import cv2
import image_processing_functions as ipf

class LabelAligner(TransformerMixin, BaseEstimator):
    def __init__(self,
                 strategy: str = "orb",
                 n_features: int = 100,
                 max_matches: int = 15):
        if strategy not in ["orb", "template"]:
            raise ValueError(f"Invalid alignment strategy: {strategy}. Supported strategies are 'orb' and 'template'.")
        self.strategy = strategy
        self.n_features = n_features
        self.max_matches = max_matches

    def fit(self, X, y=None):
        self._templates = config.TEMPLATES.items()
        for template_type, template_img_path in self._templates:
            if not template_img_path.exists():
                raise FileNotFoundError(f"Template image not found at {template_img_path}")
            self._templates = {template_type: cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE) for template_type, template_img_path in self._templates}
        return self

    def transform(self, X):
        aligned_images = []

        for image in X:
            if self.strategy == "template":
                aligned = self.align_image(image, self.template_img_path)
            elif self.strategy == "orb":
                aligned = self.orb_align(image, self.template_type, self.n_features, self.max_matches)
                aligned_images.append(aligned)
        return aligned_images
    
    def align_image(self, image: np.ndarray, template_image_path: str) -> np.ndarray:
        '''
        Align the input image to the template image using deskewing and template matching. Returns the aligned image.
        '''
        deskewed = deskew_image(image)

        _, loc = ImageProcessor.get_template_matching_results(deskewed, template_image_path)

        padding = config.PADDING

        x_start, y_start = loc 
        x_start -= padding
        if x_start < 0:
            deskewed = cv2.copyMakeBorder(deskewed, padding-x_start, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
            x_start = 0
        return deskewed[y_start:, x_start:]

    def _align_using_orb_matches(self, matches, src_kps, dst_kps, template, original_image):
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
    
    def orb_align(self, image: np.ndarray, template_type:str, n_features:int=100, max_matches: int = 15, visualize=False)-> np.ndarray:
        '''
        Align the input image to the matching template using ORB feature matching.
        Returns the aligned image.

        on my pc shows 0.24 seconds computation time for 30 features and 10 matches per image on average
        
        :param image: Input image to be aligned 
        :type image: np.ndarray
        :param n_features: Number of ORB features to detect
        :type n_features: int
        :param max_matches: Maximum number of ORB matches to consider for alignment
        :type max_matches: int
        :param visualize: Whether to visualize the ORB matches and alignment results
        :return: Aligned image
        :rtype: np.ndarray
        '''

        original_image = image.copy()
        image = self.convert_to_greyscale(image)
        template_img_path = config.TEMPLATES[template_type]
        template = cv2.imread(str(template_img_path), cv2.IMREAD_GRAYSCALE)

        orb = cv2.ORB_create(nfeatures=n_features)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        src_kps, query_descrs = orb.detectAndCompute(image, None)
    
        assert template is not None, f"Failed to load template image at {template_img_path}"
        dst_kps, target_descrs = orb.detectAndCompute(template, None)
        # maybe try Knn match and Lowe's ratio test if too many false matches with crossCheck
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
        return self._align_using_orb_matches(matches, src_kps, dst_kps, template, original_image)

    def orb_align_and_clasify(self, image: np.ndarray, n_features:int=200, max_matches: int = 50, visualize=False) -> tuple[str, np.ndarray]:
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
        image = self.convert_to_greyscale(image)

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
        aligned_image = self._align_using_orb_matches(best_matches, src_kps, best_dst_kps, best_template, original_image)

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