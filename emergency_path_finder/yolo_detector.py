"""Thin wrapper around a trained YOLOv8 detector.

``ultralytics`` and ``torch`` are heavy and are only needed once a model has
actually been trained, so they are imported lazily. Everything else in this
package runs without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .geometry import BoundingBox, Detection

__all__ = ["YoloDetector", "CLASS_NAMES", "ultralytics_available"]

#: Class order used by the training configs; index must match ``data.yaml``.
CLASS_NAMES: Sequence[str] = ("exit", "stairs", "door")


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
        self.class_names = tuple(class_names)
        self.device = device
        self._model = YOLO(str(self.weights))

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
                    if class_index < len(self.class_names)
                    else f"class_{class_index}"
                )
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf.item()),
                        box=BoundingBox.from_xyxy(x1, y1, x2, y2),
                        source="yolo",
                    )
                )
        return detections
