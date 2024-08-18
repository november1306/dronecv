import cv2
import numpy as np

from base_detector import BaseDetector
from utils import preprocess_image


class OpticalFlowDetector(BaseDetector):
    def __init__(self):
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.movement_threshold = 0.3

    def detect(self, prev_frame, current_frame, next_frame, scope):
        prev_roi = scope.get_roi(prev_frame)
        current_roi = scope.get_roi(current_frame)

        prev_roi = preprocess_image(prev_roi)
        current_roi = preprocess_image(current_roi)

        prev_points = cv2.goodFeaturesToTrack(prev_roi, mask=None, **self.feature_params)

        debug_info = {
            "scope_size": prev_roi.shape,
            "num_features": 0 if prev_points is None else len(prev_points)
        }

        if prev_points is None:
            debug_info["error"] = "No features detected in the previous frame"
            return None, debug_info

        current_points, status, error = cv2.calcOpticalFlowPyrLK(prev_roi, current_roi, prev_points, None,
                                                                 **self.lk_params)

        good_new = current_points[status == 1]
        good_old = prev_points[status == 1]

        debug_info["num_good_points"] = len(good_new)

        if len(good_new) > 0 and len(good_old) > 0:
            movement = good_new - good_old

            # Estimate background motion using RANSAC
            if len(movement) >= 4:
                _, inliers = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
                inliers = inliers.ravel() == 1
                background_motion = np.mean(movement[inliers], axis=0) if np.any(inliers) else np.mean(movement, axis=0)
            else:
                background_motion = np.mean(movement, axis=0)

            # Calculate relative motion
            relative_motion = movement - background_motion
            relative_magnitude = np.linalg.norm(relative_motion, axis=1)
            max_magnitude = np.max(relative_magnitude)

            debug_info["background_motion"] = background_motion.tolist()
            debug_info["max_relative_magnitude"] = max_magnitude

            if max_magnitude > self.movement_threshold:
                max_movement_idx = np.argmax(relative_magnitude)
                max_point = good_new[max_movement_idx]

                box_size = max(int(max_magnitude * 2), 5)
                box_size = min(box_size, 30)

                x, y = max_point.ravel()
                local_detection = (int(x - box_size // 2), int(y - box_size // 2), box_size, box_size)
                global_detection = scope.local_to_global(local_detection)
                debug_info["detection"] = global_detection
                debug_info["max_movement"] = relative_magnitude[max_movement_idx]
                return global_detection, debug_info

        debug_info["error"] = "No significant movement detected"
        return None, debug_info
