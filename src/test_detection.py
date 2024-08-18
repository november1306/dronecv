import os
import time
import cv2
from detector import create_detector
from detector_enum import DetectorType
from scope import Scope
from video_to_frames import video_to_frames
from video_processing import VideoProcessor
from visualization import visualize_results, draw_mog2_mask
from utils import log_debug_info, save_debug_image

VIDEO_PATH = r"../video/20m_takeoff.avi"
SCOPE_CENTER = (450, 340) #20m takeoff
# VIDEO_PATH = r"../video/20m_short.avi"
# SCOPE_CENTER = (318, 230) #20m short
# VIDEO_PATH = r"../video/60m_away.avi"
# SCOPE_CENTER = (318, 230)  # 60m_away
# VIDEO_PATH = r"../video/60m_return.avi"
# SCOPE_CENTER = (290, 200)
SCOPE_SIZE = (100, 100)
NUM_FRAMES = 100
START_FRAME = 0
FPS = None  # Set to None to process all frames
OUTPUT_DIR = os.path.join("output", "detection_test_" + time.strftime("%Y%m%d_%H%M%S"))


def test_detection(detector_type: DetectorType):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving output images to: {OUTPUT_DIR}")

    video_path = os.path.normpath(VIDEO_PATH)

    # Extract frames from the video
    frame_count, frames_folder = video_to_frames(video_path, FPS)
    print(f"Total frames extracted: {frame_count}")

    # Initialize VideoProcessor with the extracted frames
    processor = VideoProcessor(frames_folder, START_FRAME)
    print(f"Video dimensions: {processor.frame_width}x{processor.frame_height}")

    # Calculate top-left corner from center and size
    scope_top_left = (
        SCOPE_CENTER[0] - SCOPE_SIZE[0] // 2,
        SCOPE_CENTER[1] - SCOPE_SIZE[1] // 2
    )

    # Initialize Scope
    scope = Scope(scope_top_left, SCOPE_SIZE)

    # Create detector
    detector = create_detector(detector_type)

    print(f"Processing up to {NUM_FRAMES} frames")
    print(f"Video dimensions: {processor.frame_width}x{processor.frame_height}")
    print(f"Scope center: {SCOPE_CENTER}")
    print(f"Scope top-left: {scope.top_left}")
    print(f"Scope size: {scope.size}")
    print(f"Detector type: {detector_type.name}")

    for frame_number, frame in processor.process_frames(NUM_FRAMES):
        print(f"\nProcessing frame {frame_number}")

        prev_frame = processor.get_frame(frame_number - 1)
        next_frame = processor.get_frame(frame_number + 1)

        if prev_frame is not None and next_frame is not None:
            detection, debug_info = detector.detect(prev_frame, frame, next_frame, scope)

            # Log debug info
            log_debug_info(frame_number, detector_type.name, detection, debug_info, f"{OUTPUT_DIR}/debug_log.json")

            # Save debug images
            for key in ['mask', 'diff_image', 'threshold_image', 'flow_image']:
                if key in debug_info:
                    suffix = '_' + key.split('_')[0] if '_' in key else ''
                    save_debug_image(OUTPUT_DIR, frame_number, detector_type.name, debug_info[key], suffix)

            result_frame = visualize_results(frame, detection, debug_info, detector_type.name)
            scope.draw(result_frame)

            if 'mask' in debug_info:
                draw_mog2_mask(result_frame, debug_info['mask'], scope)
        else:
            result_frame = frame
            scope.draw(result_frame)

        output_path = os.path.join(OUTPUT_DIR, f'result_frame_{frame_number:04d}.jpg')
        cv2.imwrite(output_path, result_frame)

    print(f"Processing complete. {NUM_FRAMES} frames processed.")
    print(f"Output images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    test_detection(DetectorType.FRAME_DIFF_ART)

    # for detector_type in AVAILABLE_DETECTORS:
    #     test_detection(detector_type)
