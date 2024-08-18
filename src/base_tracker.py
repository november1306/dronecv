import cv2
import numpy as np

from src import preprocess_image


class BaseTracker:
    def __init__(self, tracker_type):
        self.tracker_type = tracker_type
        self.tracker = self._create_tracker()

    def _create_tracker(self):
        raise NotImplementedError("Subclasses must implement this method")

    def init(self, frame, bbox):
        return self.tracker.init(frame, bbox)

    def update(self, frame):
        return self.tracker.update(frame)

    def get_bbox(self):
        return self.tracker.get_bbox() if hasattr(self.tracker, 'get_bbox') else None

    def _calculate_psr(self, roi):
        # Default implementation from tracker.py
        gray = preprocess_image(roi)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        peak = np.max(magnitude_spectrum)
        mask = np.ones_like(magnitude_spectrum)
        center = (mask.shape[1] // 2, mask.shape[0] // 2)
        cv2.circle(mask, center, 5, 0, -1)  # Exclude peak region
        sidelobe = magnitude_spectrum * mask
        mean = np.mean(sidelobe[mask == 1])
        std = np.std(sidelobe[mask == 1])
        return (peak - mean) / std

    @property
    def confidence(self):
        return self.tracker.confidence if hasattr(self.tracker, 'confidence') else 1.0
