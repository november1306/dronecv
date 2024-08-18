import cv2
import numpy as np

from base_detector import BaseDetector


class MOG2Detector(BaseDetector):
    def __init__(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=1000,
            varThreshold=10,
            detectShadows=False
        )
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((11, 11), np.uint8)

    def detect(self, prev_frame, current_frame, next_frame, scope):
        current_roi = scope.get_roi(current_frame)

        # Apply MOG2
        fgmask = self.mog2.apply(current_roi)

        # Apply threshold to get rid of shadows
        _, fgmask = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)

        # Morphological operations to remove noise and fill gaps
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel_open)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel_close)

        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area and aspect ratio
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2:  # Aspect ratio constraint
                    valid_contours.append(contour)

        detection = None
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            local_detection = (x, y, w, h)
            detection = scope.local_to_global(local_detection)

        debug_info = {
            "mask": fgmask,
            "foreground_ratio": np.sum(fgmask > 0) / fgmask.size
        }

        return detection, debug_info
