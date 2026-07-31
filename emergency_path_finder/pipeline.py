"""The end-to-end capability: frame in, navigation instruction out.

This is what the README promises and what the mobile app is a front-end for.
It runs the trained model when one is available and always keeps the classical
fallbacks in the loop, because a model trained on well-lit stock photos is
exactly the thing that fails in a real emergency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .config import Settings, get_settings
from .detection import FallbackDetector, estimate_light_quality
from .geometry import Detection, LightSource, non_max_suppression
from .navigation import NavigationAdvice, NavigationHelper
from .yolo_detector import YoloDetector, ultralytics_available

__all__ = ["PathFinder", "FrameAnalysis"]

#: Below this light-quality score the app should turn the torch on.
TORCH_LIGHT_THRESHOLD = 0.25


@dataclass(frozen=True)
class FrameAnalysis:
    """Everything one pass over a frame produced.

    The CLI and the app need the boxes, the light blobs *and* the advice. They
    used to ask for each separately, which ran every detector twice per frame.
    """

    advice: NavigationAdvice
    detections: List[Detection] = field(default_factory=list)
    lights: List[LightSource] = field(default_factory=list)

    @property
    def light_quality(self) -> float:
        return self.advice.light_quality

    @property
    def torch_advised(self) -> bool:
        return self.advice.light_quality < TORCH_LIGHT_THRESHOLD


class PathFinder:
    """Fuses model detections with classical fallbacks into one instruction."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_path: Optional[Path] = None,
        use_model: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.fallback = FallbackDetector(self.settings.detector)
        self.model: Optional[YoloDetector] = None
        self.model_error: Optional[str] = None

        if use_model:
            explicit = model_path or self.settings.model_path
            if explicit is not None and not Path(explicit).exists():
                # Silently falling back here is how a typo in --model or a stale
                # EPF_MODEL_PATH produces classical-CV results that look like
                # model output.
                weights = None
                self.model_error = (
                    f"weights not found: {explicit} - running on classical CV only"
                )
            else:
                weights = explicit or self.settings.resolve_model_path()

            if weights is None:
                self.model_error = self.model_error or (
                    "no trained weights found - running on classical CV only"
                )
            elif not ultralytics_available():
                self.model_error = (
                    "ultralytics not installed - running on classical CV only"
                )
            else:
                try:
                    self.model = YoloDetector(
                        weights,
                        confidence_threshold=self.settings.detector.confidence_threshold,
                        input_size=self.settings.input_size,
                    )
                except Exception as exc:  # pragma: no cover - needs real weights
                    self.model_error = f"failed to load {weights}: {exc}"

    @property
    def uses_model(self) -> bool:
        return self.model is not None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detections from every enabled source, de-duplicated.

        Model detections are ordered first so NMS keeps them over a
        lower-confidence classical guess for the same object.
        """
        detections: List[Detection] = []
        if self.model is not None:
            try:
                detections.extend(self.model.detect(frame))
            except Exception as exc:  # pragma: no cover - needs real weights
                self.model_error = f"inference failed: {exc}"

        detections.extend(self.fallback.detect_all(frame))
        return self._deduplicate(detections)

    @staticmethod
    def _deduplicate(detections: Sequence[Detection]) -> List[Detection]:
        # Bias towards model output when a model and a fallback describe the
        # same box, without letting a weak model prediction beat a strong
        # classical one outright.
        weighted = [
            Detection(
                label=d.label,
                confidence=min(d.confidence * (1.15 if d.source == "yolo" else 1.0), 1.0),
                box=d.box,
                source=d.source,
                direction=d.direction,
            )
            for d in detections
        ]
        return non_max_suppression(weighted, iou_threshold=0.5)

    def analyze_frame(
        self, frame: np.ndarray, device_orientation_deg: float = 0.0
    ) -> FrameAnalysis:
        """One pass over a frame: detections, lights and the advice together.

        Callers that need the boxes as well as the instruction must use this
        rather than calling ``detect()`` and ``analyze()`` in sequence - that
        ran every detector twice on every frame.
        """
        detections = self.detect(frame)
        lights = self.fallback.detect_light_sources(frame)
        light_quality = estimate_light_quality(frame)

        vanishing_point = None
        if not detections:
            # Only worth the Hough pass when there is nothing better to steer by.
            vanishing_point = self.fallback.detect_vanishing_point(frame)

        advice = NavigationHelper.advise(
            detections=detections,
            image_shape=frame.shape,
            light_quality=light_quality,
            lights=lights,
            vanishing_point=vanishing_point,
            device_orientation_deg=device_orientation_deg,
        )
        return FrameAnalysis(advice=advice, detections=detections, lights=lights)

    def analyze(
        self, frame: np.ndarray, device_orientation_deg: float = 0.0
    ) -> NavigationAdvice:
        """Full per-frame analysis: what is there and where to go."""
        return self.analyze_frame(frame, device_orientation_deg).advice

    def should_enable_torch(
        self, frame: np.ndarray, threshold: float = TORCH_LIGHT_THRESHOLD
    ) -> bool:
        """Whether the app should switch the flashlight on for this frame."""
        return estimate_light_quality(frame) < threshold
