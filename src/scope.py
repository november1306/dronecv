import cv2


class Scope:
    def __init__(self, top_left, size):
        self.top_left = top_left  # (x, y) of top-left corner
        self.size = size  # (width, height)

    def get_roi(self, frame):
        x, y = self.top_left
        w, h = self.size
        return frame[y:y + h, x:x + w]

    def global_to_local(self, global_coords):
        return (global_coords[0] - self.top_left[0], global_coords[1] - self.top_left[1])

    def local_to_global(self, local_coords):
        return (local_coords[0] + self.top_left[0], local_coords[1] + self.top_left[1])

    def draw(self, frame, color=(0, 255, 0)):
        x, y = self.top_left
        w, h = self.size
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    @property
    def bbox(self):
        return (*self.top_left, *self.size)
