import cv2
import os

class VideoProcessor:
    def __init__(self, frames_folder, start_frame=0):
        self.frames_folder = frames_folder
        self.start_frame = start_frame
        self.frame_count = len([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
        self.current_frame_num = start_frame

        # Initialize frame dimensions
        first_frame = self.get_frame(0)
        if first_frame is not None:
            self.frame_height, self.frame_width = first_frame.shape[:2]
        else:
            raise ValueError("Could not read the first frame")

    def process_frames(self, num_frames=None):
        if num_frames is None:
            num_frames = self.frame_count - self.start_frame
        else:
            num_frames = min(num_frames, self.frame_count - self.start_frame)

        for i in range(num_frames):
            self.current_frame_num = self.start_frame + i
            frame = self.get_frame(self.current_frame_num)
            if frame is None:
                print(f"Warning: Empty frame at {self.current_frame_num}")
                continue
            yield self.current_frame_num, frame

    def get_frame(self, frame_number):
        if 0 <= frame_number < self.frame_count:
            frame_path = os.path.join(self.frames_folder, f'frame_{frame_number:04d}.jpg')
            return cv2.imread(frame_path)
        return None