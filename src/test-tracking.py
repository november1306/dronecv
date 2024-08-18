import os
import time

import cv2
import numpy as np

from src.enum_tracker import TrackerType
from src.tracker_art import ArtTracker
from src.visualization import draw_bbox
from video_processing import VideoProcessor
from video_to_frames import video_to_frames

# Available trackers
AVAILABLE_TRACKERS = ['CSRT', 'KCF', 'MIL', 'ART']

# Constants
# VIDEO_PATH = r"../video/60m_return.avi"
# INITIAL_BBOX = (270, 182, 10, 10)  # 20m_takeoff
VIDEO_PATH = r"../video/20m_takeoff.avi"
INITIAL_BBOX = (441, 333, 15, 15)  # 20m_takeoff
# VIDEO_PATH = r"../video/20m_short.avi"
# INITIAL_BBOX = (315, 225, 15, 15) #20m_short
# VIDEO_PATH = r"../video/60m_away.avi"
# INITIAL_BBOX = (318, 230, 20, 20) #60m short
NUM_FRAMES = 200
START_FRAME = 0
FPS = None  # Set to None to process all frames
OUTPUT_DIR = os.path.join("output", "tracker_test_" + time.strftime("%Y%m%d_%H%M%S"))
PAUSE_BETWEEN_FRAMES = 10  # In milliseconds


def create_tracker(tracker_type: TrackerType):
    if tracker_type == TrackerType.MIL:
        return cv2.TrackerMIL_create()
    elif tracker_type == TrackerType.KCF:
        return cv2.TrackerKCF_create()
    elif tracker_type == TrackerType.CSRT:
        return cv2.TrackerCSRT_create()
    elif tracker_type == TrackerType.ART:
        return ArtTracker()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")


def log_initialization(tracker_type, frame, bbox, output_dir):
    draw_bbox(frame, bbox, (255, 0, 0))
    initial_frame_path = os.path.join(output_dir, f'initial_frame_{tracker_type.name}.jpg')
    cv2.imwrite(initial_frame_path, frame)
    print(f"Saved initial frame for {tracker_type.name} to: {initial_frame_path}")


def log_frame(tracker_type, frame, bbox, status, frame_time, output_dir, frame_number):
    if status == "Tracked":
        draw_bbox(frame, bbox, (0, 255, 0))
    output_path = os.path.join(output_dir, f'result_frame_{tracker_type.name}_{frame_number:04d}.jpg')
    cv2.imwrite(output_path, frame)
    return frame_time, status


# This method has been moved to the logger module

def test_trackers(trackers, tracker_types):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving output images to: {OUTPUT_DIR}")

    video_path = os.path.normpath(VIDEO_PATH)
    frame_count, frames_folder = video_to_frames(video_path, FPS)
    print(f"Total frames extracted: {frame_count}")

    processor = VideoProcessor(frames_folder, START_FRAME)
    print(f"Video dimensions: {processor.frame_width}x{processor.frame_height}")

    frame_times = {tracker_type: [] for tracker_type in tracker_types}
    tracking_status = {tracker_type: [] for tracker_type in tracker_types}

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 1280, 720)

    for frame_number, frame in processor.process_frames(NUM_FRAMES):
        print(f"\nProcessing frame {frame_number}")

        if frame is None:
            print(f"Error: Frame {frame_number} is None")
            continue

        tracker_frames = {tracker_type: frame.copy() for tracker_type in tracker_types}

        if frame_number == START_FRAME:
            for tracker, tracker_type in zip(trackers, tracker_types):
                tracker_ok = tracker.init(frame, INITIAL_BBOX)
                tracking_status[tracker_type].append("Initialized" if tracker_ok else "Failed")
                log_initialization(tracker_type, tracker_frames[tracker_type], INITIAL_BBOX, OUTPUT_DIR)
        else:
            for tracker, tracker_type in zip(trackers, tracker_types):
                start_time = time.time()
                ok, bbox = tracker.update(frame)
                frame_time = time.time() - start_time

                status = "Tracked" if ok else "Lost"
                frame_time, status = log_frame(tracker_type, tracker_frames[tracker_type], bbox, status, frame_time,
                                               OUTPUT_DIR, frame_number)

                frame_times[tracker_type].append(frame_time)
                tracking_status[tracker_type].append(status)

                if not ok:
                    print(f"{tracker_type.name} tracking failure detected")

        combined_frame = np.hstack(list(tracker_frames.values()))

        for i, tracker_type in enumerate(tracker_types):
            cv2.putText(combined_frame, tracker_type.name, (10 + i * frame.shape[1], 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

        cv2.imshow("Tracking", combined_frame)

        if cv2.waitKey(PAUSE_BETWEEN_FRAMES) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    for tracker_type in tracker_types:
        log_performance_metrics(tracker_type, frame_times[tracker_type], tracking_status[tracker_type])

    print(f"\nProcessing complete. {NUM_FRAMES} frames processed.")
    print(f"Output images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    tracker_types = [TrackerType.CSRT, TrackerType.ART]
    trackers = [create_tracker(tracker_type) for tracker_type in tracker_types]
    test_trackers(trackers, tracker_types)
