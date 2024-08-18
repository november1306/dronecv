import cv2

from src.tracker_art import ArtTracker
from src.enum_tracker import TrackerType
from utils import preprocess_image




class ObjectTracker:
    def __init__(self, tracker_type='CSRT', psr_threshold=5.0):
        self.tracker_type = tracker_type
        self.tracker = create_tracker(tracker_type)
        self.current_bbox = None
        self.psr_threshold = psr_threshold
        self.prev_frame = None
        self.lost_count = 0
        self.max_lost_frames = 10
        self.confidence = 1.0

    def init(self, frame, bbox):
        self.current_bbox = bbox
        self.prev_frame = preprocess_image(frame)

        print(f"Initializing {self.tracker_type} tracker")
        print(f"Initial bounding box: {bbox}")
        print(f"Frame shape: {frame.shape}")

        # Debug: Save a cropped image of the initial bounding box
        x, y, w, h = [int(v) for v in bbox]
        crop = frame[y:y + h, x:x + w]
        cv2.imwrite(f"debug_{self.tracker_type}_init_crop.jpg", crop)

        ok = self.tracker.init(frame, bbox)
        if not ok:
            print(f"Failed to initialize {self.tracker_type} tracker")
        return ok

    def update(self, frame):
        ok, new_bbox = self.tracker.update(frame)

        if ok:
            x, y, w, h = [int(v) for v in new_bbox]
            if w > 0 and h > 0:
                self.current_bbox = (x, y, w, h)
                self.lost_count = 0

                # Calculate PSR (Peak to Sidelobe Ratio)
                roi = frame[y:y + h, x:x + w]
                psr = self._calculate_psr(roi)

                # Update confidence based on PSR
                self.confidence = max(0, min(1, psr / self.psr_threshold))

                if psr < self.psr_threshold:
                    print(f"Low PSR detected: {psr:.2f}. Confidence: {self.confidence:.2f}")
                    if self.confidence < 0.5:  # Only reinitialize if confidence is low
                        ok = self._reinitialize(frame)
            else:
                ok = False
                print("Invalid bounding box detected. Attempting reinitialization.")
        else:
            self.lost_count += 1
            print(f"Tracking failed. Lost count: {self.lost_count}")
            if self.lost_count > self.max_lost_frames / 2:
                ok = self._reinitialize(frame)

        self.prev_frame = preprocess_image(frame)
        return ok, self.current_bbox

    def _reinitialize(self, frame):
        print(f"{self.tracker_type} lost target, re-initializing")
        self.tracker = create_tracker(self.tracker_type)
        ok = self.tracker.init(frame, self.current_bbox)
        if ok:
            self.lost_count = 0
            self.confidence = 0.7  # Set confidence to moderate level after reinitialization
        else:
            print(f"Failed to re-initialize {self.tracker_type} tracker")
            self.confidence *= 0.8  # Reduce confidence when reinitialization fails
        return ok


    def get_bbox(self):
        return self.current_bbox
