from enum import Enum, auto


class DetectorType(Enum):
    OPT_FLOW = auto()
    FRAME_DIFF = auto()
    FRAME_DIFF_ART = auto()
    MOG2 = auto()

    @classmethod
    def list(cls):
        return list(cls)
