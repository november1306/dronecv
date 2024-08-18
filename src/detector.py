from src import save_debug_image, log_debug_info
from src.detector_frame_diff import FrameDiffDetector
from src.detector_mog2 import MOG2Detector
from src.detector_optical_flow import OpticalFlowDetector
from src.detector_enum import DetectorType

def create_detector(detector_type: DetectorType):
    if detector_type.name == 'MOG2':
        return MOG2Detector()
    elif detector_type.name == 'FRAME_DIFF':
        return FrameDiffDetector()
    elif detector_type.name == 'FRAME_DIFF_ART':
        return FrameDiffDetector()
    elif detector_type.name == 'OPT_FLOW':
        return OpticalFlowDetector()
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")



class ObjectDetector:
    def __init__(self, detector_type):
        self.detector = create_detector(detector_type)
        self.detector_type = detector_type

    def detect_objects(self, prev_frame, current_frame, next_frame, scope, frame_num, output_dir):
        # Perform detection
        detection, debug_info = self.detector.detect(prev_frame, current_frame, next_frame, scope)

        # Log debug info
        log_debug_info(frame_num, self.detector_type, detection, debug_info, f"{output_dir}/debug_log.json")

        # Save debug images
        self._save_debug_images(output_dir, frame_num, debug_info)

        return detection, debug_info

    def _save_debug_images(self, output_dir, frame_num, debug_info):
        for key in ['mask', 'diff_image', 'threshold_image', 'flow_image']:
            if key in debug_info:
                suffix = '_' + key.split('_')[0] if '_' in key else ''
                save_debug_image(output_dir, frame_num, self.detector_type, debug_info[key], suffix)
