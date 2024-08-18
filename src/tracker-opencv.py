import cv2

from src.base_tracker import BaseTracker
from src.enum_tracker import TrackerType


class TrackerOpenCV(BaseTracker):
    def __init__(self, tracker_type):
        super().__init__(tracker_type)
        self.opencv_tracker = self._create_tracker()

    def _create_tracker(self):
        if self.tracker_type == TrackerType.MIL:
            return cv2.TrackerMIL_create()
        elif self.tracker_type == TrackerType.KCF:
            return cv2.TrackerKCF_create()
        elif self.tracker_type == TrackerType.CSRT:
            return cv2.TrackerCSRT_create()
        else:
            raise ValueError(f"Unsupported OpenCV tracker type: {self.tracker_type}")

    def init(self, frame, bbox):
        return self.opencv_tracker.init(frame, bbox)

    def update(self, frame):
        ok, bbox = self.opencv_tracker.update(frame)
        return ok, tuple(map(int, bbox))

    def get_bbox(self):
        return self.opencv_tracker.get_bbox()

    @property
    def confidence(self):
        # OpenCV trackers don't provide a confidence score, so we return 1.0
        return 1.0
