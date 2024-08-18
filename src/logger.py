import time

import numpy as np

from src.utils import log_debug_info, save_debug_image, calculate_performance_metrics
from src.visualization import draw_detection, create_debug_image


class Logger:
    def __init__(self, output_dir, component_type, component_name):
        self.output_dir = output_dir
        self.component_type = component_type  # 'tracker' or 'detector'
        self.component_name = component_name
        self.frame_times = []
        self.status_log = []
        self.log_file = f"{output_dir}/{component_type}_{component_name}_log.json"

    def log_frame(self, frame_num, frame, result, debug_info=None):
        start_time = time.time()

        if self.component_type == 'tracker':
            ok, bbox = result
            status = "Tracked" if ok else "Lost"
        else:  # detector
            bbox = result
            status = "Detected" if bbox is not None else "Not Detected"

        self.status_log.append(status)

        # Log debug info
        log_entry = log_debug_info(frame_num, self.component_name, bbox, debug_info or {}, self.log_file)

        # Save debug images
        if debug_info:
            for key, image in debug_info.items():
                if isinstance(image, np.ndarray):
                    save_debug_image(self.output_dir, frame_num, self.component_name, image, f"_{key}")

        # Create and save visualization
        vis_frame = create_debug_image(frame, bbox, debug_info or {}, self.component_name,
                                       None)  # We don't have scope info here
        save_debug_image(self.output_dir, frame_num, self.component_name, vis_frame, "_visualization")

        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)

        return log_entry

    def print_performance_metrics(self):
        total_time, avg_frame_time, fps = calculate_performance_metrics(self.frame_times)

        print(f"\nPerformance Metrics for {self.component_type.capitalize()} {self.component_name}:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average frame processing time: {avg_frame_time:.4f} seconds")
        print(f"Frames per second (FPS): {fps:.2f}")

        if self.component_type == 'tracker':
            success_rate = self.status_log.count("Tracked") / len(self.status_log) * 100
            print(f"Tracking Success Rate: {success_rate:.2f}%")

        print(f"{self.component_type.capitalize()} Status Summary:")
        for status in set(self.status_log):
            print(f"{status}: {self.status_log.count(status)}")

    def log_initialization(self, frame, bbox):
        draw_detection(frame, bbox, color=(255, 0, 0), label=f"Initial {self.component_type.capitalize()}")
        save_debug_image(self.output_dir, 0, self.component_name, frame, "_initialization")
        print(
            f"Saved initial frame with {self.component_type} to: {self.output_dir}/{self.component_name}_initialization.jpg")


def get_logger(output_dir, component_type, component_name):
    return Logger(output_dir, component_type, component_name)
