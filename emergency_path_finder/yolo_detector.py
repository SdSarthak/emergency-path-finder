"""Thin wrapper around a trained YOLOv8 detector.

``ultralytics`` and ``torch`` are heavy and are only needed once a model has
actually been trained, so they are imported lazily. Everything else in this
package runs without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .geometry import BoundingBox, Detection

__all__ = [
    "YoloDetector",
    "CLASS_NAMES",
    "normalize_label",
    "model_class_names",
    "ultralytics_available",
]

#: Class order used by the training configs; index must match ``data.yaml``.
CLASS_NAMES: Sequence[str] = ("exit", "stairs", "door")

#: The navigation layer only understands these three labels. Public datasets
#: spell them a dozen ways, and a stairs model trained from Roboflow emits
#: ``escalator`` - mapped here rather than being dropped or, worse, passed
#: through as an unknown label that ``select_target`` silently ignores.
_LABEL_ALIASES: Dict[str, str] = {
    "exit": "exit",
    "exits": "exit",
    "exit-sign": "exit",
    "exit sign": "exit",
    "exitsign": "exit",
    "emergency exit": "exit",
    "emergency-exit": "exit",
    "fire exit": "exit",
    "stair": "stairs",
    "stairs": "stairs",
    "staircase": "stairs",
    "stairway": "stairs",
    "steps": "stairs",
    "escalator": "stairs",
    "escalators": "stairs",
    "door": "door",
    "doors": "door",
    "doorway": "door",
    "gate": "door",
}


def normalize_label(label: str) -> str:
    """Map a dataset class name onto the pipeline vocabulary.

    Unknown labels are returned lower-cased and unchanged; they survive into
    the detection list for debugging but will never be chosen as a navigation
    target.
    """
    cleaned = str(label).strip().lower()
    for key in (cleaned, cleaned.replace("_", " "), cleaned.replace("_", "-")):
        if key in _LABEL_ALIASES:
            return _LABEL_ALIASES[key]
    return cleaned


def model_class_names(model: object, fallback: Sequence[str] = CLASS_NAMES) -> List[str]:
    """Read the class names a trained checkpoint carries with it.

    Ultralytics stores ``names`` as ``{index: name}``. Trusting the hardcoded
    ``CLASS_NAMES`` instead is how a stairs model (classes ``stairs``,
    ``escalator``) ends up reporting an ``exit`` that is not there.
    """
    names = getattr(model, "names", None)
    if isinstance(names, dict) and names:
        try:
            return [str(names[key]) for key in sorted(names, key=int)]
        except (KeyError, TypeError, ValueError):
            return [str(value) for value in names.values()]
    if isinstance(names, (list, tuple)) and names:
        return [str(value) for value in names]
    return list(fallback)


def ultralytics_available() -> bool:
    """True when the optional training/inference stack is installed."""
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return True


class YoloDetector:
    """Runs a trained ``.pt`` (or exported) model over BGR frames."""

    def __init__(
        self,
        weights: Path | str,
        confidence_threshold: float = 0.35,
        input_size: int = 416,
        class_names: Sequence[str] = CLASS_NAMES,
        device: Optional[str] = None,
    ) -> None:
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"model weights not found: {self.weights}. Train one with "
                f"`python training/train_exit_detector.py` or point EPF_MODEL_PATH "
                f"at an existing checkpoint."
            )
        if not ultralytics_available():
            raise ImportError(
                "ultralytics is not installed. Install the training extras: "
                "pip install -r training/requirements.txt"
            )

        from ultralytics import YOLO

        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.device = device
        self._model = YOLO(str(self.weights))
        # The checkpoint's own class list wins over the caller's default: the
        # two only agree when the model happens to have been trained on the
        # exit dataset, and a mismatch silently relabels every detection.
        self.class_names = tuple(model_class_names(self._model, class_names))

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Return detections for one BGR frame."""
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("detect() expects a BGR numpy array")

        results = self._model.predict(
            source=image,
            imgsz=self.input_size,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_index = int(box.cls.item())
                label = (
                    self.class_names[class_index]
                    if 0 <= class_index < len(self.class_names)
                    else f"class_{class_index}"
                )
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        label=normalize_label(label),
                        confidence=float(box.conf.item()),
                        box=BoundingBox.from_xyxy(x1, y1, x2, y2),
                        source="yolo",
                    )
                )
        return detections
