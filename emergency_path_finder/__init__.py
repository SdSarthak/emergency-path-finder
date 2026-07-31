"""Emergency Path Finder - offline exit detection and navigation.

Typical use::

    import cv2
    from emergency_path_finder import PathFinder

    finder = PathFinder()
    advice = finder.analyze(cv2.imread("corridor.jpg"))
    print(advice.direction, advice.instruction)

``emergency_path_finder.training`` is intentionally not imported here: it pulls
in the heavy ultralytics/torch stack, which inference does not need.
"""

from .config import DetectorConfig, Settings, get_settings
from .detection import FallbackDetector, enhance_low_light, estimate_light_quality
from .geometry import BoundingBox, Detection, LightSource, iou, non_max_suppression
from .navigation import Direction, NavigationAdvice, NavigationHelper, Urgency
from .pipeline import FrameAnalysis, PathFinder

__version__ = "1.0.0"

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectorConfig",
    "Direction",
    "FallbackDetector",
    "FrameAnalysis",
    "LightSource",
    "NavigationAdvice",
    "NavigationHelper",
    "PathFinder",
    "Settings",
    "Urgency",
    "enhance_low_light",
    "estimate_light_quality",
    "get_settings",
    "iou",
    "non_max_suppression",
    "__version__",
]
