import os

import cv2
import numpy as np
import json


def preprocess_image(image):
    # Check if the image is already grayscale
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def log_debug_info(frame_num, method, detection, debug_info, log_file=None):
    log_entry = {
        "frame": frame_num,
        "method": method,
        "debug_info": {}
    }

    if detection is not None:
        log_entry["detection"] = detection

    for key, value in debug_info.items():
        if isinstance(value, np.ndarray):
            log_entry["debug_info"][key] = f"array of shape {value.shape}"
        elif isinstance(value, list) and len(value) > 10:
            log_entry["debug_info"][key] = f"list of length {len(value)}"
        elif isinstance(value, (int, float, str, bool, type(None))):
            log_entry["debug_info"][key] = value
        else:
            log_entry["debug_info"][key] = str(value)

    log_string = json.dumps(log_entry, indent=2)
    print(log_string)

    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_string + "\n")

    return log_entry


def save_debug_image(output_dir, frame_num, method, image, suffix=''):
    filename = f"{output_dir}/debug_{method}_{frame_num:04d}{suffix}.png"
    cv2.imwrite(filename, image)


def log_detection(frame_num, method, detection, debug_info=None):
    if detection:
        print(f"Frame {frame_num}: Object detected ({method}) at: {detection}")
    else:
        print(f"Frame {frame_num}: No object detected using {method}")

    if debug_info:
        print(f"Debug info for {method}:")
        for key, value in debug_info.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: array of shape {value.shape}")
            elif isinstance(value, list) and len(value) > 10:
                print(f"  {key}: list of length {len(value)}")
            else:
                print(f"  {key}: {value}")

def process_initial_frame(frame, initial_bbox, output_dir):
    x, y, w, h = initial_bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    initial_frame_path = os.path.join(output_dir, 'initial_frame_with_detection.jpg')
    cv2.imwrite(initial_frame_path, frame)
    print(f"Saved initial frame with detection to: {initial_frame_path}")

def calculate_performance_metrics(frame_times):
    if frame_times:
        total_time = sum(frame_times)
        avg_frame_time = total_time / len(frame_times)
        fps = 1.0 / avg_frame_time
        return total_time, avg_frame_time, fps
    return None, None, None

def adaptive_scope_update(frame, detection, current_center, scope_size, max_shift=10):
    """
    Update the scope center based on the current detection.

    :param frame: The current frame (used for getting frame dimensions)
    :param detection: Tuple (x, y, w, h) representing the current detection
    :param current_center: Tuple (x, y) representing the current scope center
    :param scope_size: Tuple (width, height) representing the scope size
    :param max_shift: Maximum allowed shift in pixels per frame
    :return: New scope center (x, y)
    """
    frame_height, frame_width = frame.shape[:2]
    x, y, w, h = detection
    target_center = (x + w // 2, y + h // 2)

    dx = target_center[0] - current_center[0]
    dy = target_center[1] - current_center[1]

    # Limit the shift to max_shift
    dx = max(min(dx, max_shift), -max_shift)
    dy = max(min(dy, max_shift), -max_shift)

    new_center_x = max(scope_size[0] // 2, min(current_center[0] + dx, frame_width - scope_size[0] // 2))
    new_center_y = max(scope_size[1] // 2, min(current_center[1] + dy, frame_height - scope_size[1] // 2))

    return (new_center_x, new_center_y)