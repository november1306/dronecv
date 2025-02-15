import os
import time
import cv2
import numpy as np
from detector import create_detector
from detector_enum import DetectorType
from scope import Scope
from video_to_frames import video_to_frames
from video_processing import VideoProcessor
from visualization import visualize_results, draw_mog2_mask
from utils import log_debug_info, save_debug_image

print("All imports successful!")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}") 