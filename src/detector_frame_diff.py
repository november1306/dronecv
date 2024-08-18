import cv2
import numpy as np

from base_detector import BaseDetector
from utils import preprocess_image


class FrameDiffDetector(BaseDetector):
    def __init__(self, min_area=2, max_area=200, use_morphology=True):
        self.min_area = min_area
        self.max_area = max_area
        self.use_morphology = use_morphology

    def detect(self, prev_frame, current_frame, next_frame, scope):
        # Get ROIs using the scope
        prev_roi = scope.get_roi(prev_frame)
        current_roi = scope.get_roi(current_frame)
        next_roi = scope.get_roi(next_frame)

        # Preprocess frames using CLAHE and ensure grayscale
        prev_roi = preprocess_image(prev_roi)
        current_roi = preprocess_image(current_roi)
        next_roi = preprocess_image(next_roi)

        # Compute frame differences
        diff1 = cv2.absdiff(prev_roi, current_roi)
        diff2 = cv2.absdiff(current_roi, next_roi)

        # Combine the differences
        diff = cv2.bitwise_or(diff1, diff2)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)

        if self.use_morphology:
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [cnt for cnt in contours if self.min_area <= cv2.contourArea(cnt) <= self.max_area]

        debug_info = {
            'diff_image': diff,
            'threshold_image': thresh,
            'num_contours': len(contours),
            'num_valid_contours': len(valid_contours),
            'contour_areas': [cv2.contourArea(cnt) for cnt in valid_contours]
        }

        detection = None
        if valid_contours:
            max_intensity_diff = 0
            best_contour = None
            for cnt in valid_contours:
                mask = np.zeros(current_roi.shape, np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                mean_intensity = cv2.mean(diff, mask=mask)[0]
                if mean_intensity > max_intensity_diff:
                    max_intensity_diff = mean_intensity
                    best_contour = cnt

            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                local_detection = (x, y, w, h)
                detection = scope.local_to_global(local_detection)
                debug_info['max_intensity_diff'] = max_intensity_diff

        return detection, debug_info
