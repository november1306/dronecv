from enum import Enum, auto


class TrackerType(Enum):
    CSRT = auto()
    KCF = auto()
    MIL = auto()
    ART = auto()

    @classmethod
    def list(cls):
        return list(cls)
