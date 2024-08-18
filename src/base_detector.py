from abc import ABC, abstractmethod


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, prev_frame, current_frame, next_frame, scope):
        """
        Detect objects in the given frames within the specified scope.

        :param prev_frame: Previous frame
        :param current_frame: Current frame
        :param next_frame: Next frame
        :param scope: Scope object defining the region of interest
        :return: Tuple of (detection, debug_info)
                 detection: (x, y, w, h) or None if no detection
                 debug_info: Dictionary containing debug information
        """
        pass
