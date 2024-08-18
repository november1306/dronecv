import cv2
import numpy as np

from src.base_tracker import BaseTracker
from src.enum_tracker import TrackerType
from utils import preprocess_image


class ArtTracker(BaseTracker):
    def __init__(self, learning_rate=0.125, psr_threshold=5.0):
        super().__init__(TrackerType.ART)
        self.learning_rate = learning_rate
        self.psr_threshold = psr_threshold
        self.current_bbox = None
        self.prev_frame = None
        self.lost_count = 0
        self.max_lost_frames = 10
        self._confidence = 1.0
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

    def _create_tracker(self):
        # ArtTracker is self-contained, so we return self
        return self

    def init(self, frame, bbox):
        self.current_bbox = bbox
        self.prev_frame = preprocess_image(frame)
        x, y, w, h = bbox
        center_x, center_y = x + w / 2, y + h / 2
        self.kalman.statePost = np.array([[center_x], [center_y], [0], [0]], dtype=np.float32)
        return True

    def update(self, frame):
        if self.current_bbox is None:
            return False, None

        # Predict next position using Kalman filter
        prediction = self.kalman.predict()

        # Perform detection using frame differencing
        detection = self._detect_object(frame)

        if detection is not None:
            x, y, w, h = detection
            center_x, center_y = x + w / 2, y + h / 2
            measurement = np.array([[center_x], [center_y]], dtype=np.float32)
            self.kalman.correct(measurement)
            self.current_bbox = detection
            self.lost_count = 0
            self._confidence = max(0, min(1, self._calculate_psr(frame[y:y + h, x:x + w]) / self.psr_threshold))
        else:
            self.lost_count += 1
            if self.lost_count > self.max_lost_frames:
                return False, None

            # Use Kalman filter prediction if no detection
            center_x, center_y = prediction[0, 0], prediction[1, 0]
            w, h = self.current_bbox[2:]
            self.current_bbox = (int(center_x - w / 2), int(center_y - h / 2), int(w), int(h))
            self._confidence *= 0.9

        self.prev_frame = preprocess_image(frame)
        return True, self.current_bbox

    def _detect_object(self, frame):
        if self.prev_frame is None:
            return None

        diff = cv2.absdiff(self.prev_frame, preprocess_image(frame))
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 20:
                return cv2.boundingRect(largest_contour)

        return None

    def _calculate_psr(self, roi):
        gray = preprocess_image(roi)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        peak = np.max(magnitude_spectrum)
        mask = np.ones_like(magnitude_spectrum)
        center = (mask.shape[1] // 2, mask.shape[0] // 2)
        cv2.circle(mask, center, 5, 0, -1)
        sidelobe = magnitude_spectrum * mask
        mean = np.mean(sidelobe[mask == 1])
        std = np.std(sidelobe[mask == 1])
        return (peak - mean) / std

    def get_bbox(self):
        return self.current_bbox

    @property
    def confidence(self):
        return self._confidence
