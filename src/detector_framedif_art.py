import cv2
import numpy as np
from base_detector import BaseDetector

class FrameDiffDetector(BaseDetector):
    def __init__(self):
        self.THRESHOLD_VALUE = 10
        self.MIN_CONTOUR_AREA = 20
        self.MORPH_KERNEL_SIZE = (5, 5)
        self.DILATION_ITERATIONS = 3
        self.EROSION_ITERATIONS = 1

    def detect(self, prev_frame, current_frame, next_frame, scope):
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference between consecutive frames
        diff = cv2.absdiff(prev_gray, curr_gray)

        # Apply a Gaussian blur to reduce noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        # Apply thresholding to detect significant changes
        _, thresh = cv2.threshold(diff, self.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to clean up the noise
        kernel = np.ones(self.MORPH_KERNEL_SIZE, np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=self.DILATION_ITERATIONS)
        thresh = cv2.erode(thresh, kernel, iterations=self.EROSION_ITERATIONS)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours of the moving object
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) >= self.MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(largest_contour)
                detection = (x, y, w, h)
                debug_info = {
                    "diff_image": diff,
                    "threshold_image": thresh,
                    "contour_area": cv2.contourArea(largest_contour)
                }
                return detection, debug_info

        return None, {"diff_image": diff, "threshold_image": thresh}