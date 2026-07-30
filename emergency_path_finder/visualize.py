"""Debug drawing helpers - used by the CLI, never on the hot path."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import Detection, LightSource
from .navigation import NavigationAdvice

__all__ = ["LABEL_COLORS", "draw_detections", "draw_advice", "render"]

#: BGR colours, matched to the Flutter overlay palette.
LABEL_COLORS = {
    "exit": (0, 255, 0),
    "stairs": (0, 255, 255),
    "door": (255, 128, 0),
}
_DEFAULT_COLOR = (200, 200, 200)


def draw_detections(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Draw labelled boxes onto a copy of ``image``."""
    canvas = image.copy()
    for detection in detections:
        color = LABEL_COLORS.get(detection.label, _DEFAULT_COLOR)
        x1, y1, x2, y2 = detection.box.as_xyxy()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = f"{detection.label} {detection.confidence:.2f}"
        if detection.direction:
            caption += f" ({detection.direction})"
        cv2.putText(
            canvas,
            caption,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def draw_lights(image: np.ndarray, lights: Sequence[LightSource]) -> np.ndarray:
    canvas = image.copy()
    for light in lights:
        cv2.circle(canvas, (light.x, light.y), 10, (255, 255, 0), 2)
    return canvas


def _draw_arrow(canvas: np.ndarray, angle_deg: float) -> None:
    height, width = canvas.shape[:2]
    center = (width // 2, height // 2)
    length = min(width, height) * 0.25
    radians = np.deg2rad(angle_deg)
    tip = (
        int(center[0] + length * np.sin(radians)),
        int(center[1] - length * np.cos(radians)),
    )
    cv2.arrowedLine(canvas, center, tip, (0, 255, 128), 4, tipLength=0.3)


def draw_advice(
    image: np.ndarray,
    advice: NavigationAdvice,
    vanishing_point: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Overlay the arrow, instruction banner and urgency badge."""
    canvas = image.copy()
    _draw_arrow(canvas, advice.arrow_angle_deg)

    point = vanishing_point or advice.vanishing_point
    if point is not None:
        cv2.drawMarker(
            canvas,
            (int(point[0]), int(point[1])),
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )

    banner = f"{advice.direction}  |  {advice.urgency}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 60), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas, advice.instruction, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (200, 220, 255), 1, cv2.LINE_AA,
    )
    return canvas


def render(
    image: np.ndarray,
    detections: Sequence[Detection],
    advice: NavigationAdvice,
    lights: Sequence[LightSource] = (),
) -> np.ndarray:
    """One-call debug view: boxes + lights + navigation overlay."""
    canvas = draw_detections(image, detections)
    if lights:
        canvas = draw_lights(canvas, lights)
    return draw_advice(canvas, advice)
