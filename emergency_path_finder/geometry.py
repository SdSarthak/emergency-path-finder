"""Geometry primitives shared by the detectors and the navigation layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

__all__ = ["BoundingBox", "Detection", "LightSource", "iou", "non_max_suppression"]


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates (top-left origin)."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError(
                f"BoundingBox width/height must be non-negative, got "
                f"({self.width}, {self.height})"
            )

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        left, right = (x1, x2) if x1 <= x2 else (x2, x1)
        top, bottom = (y1, y2) if y1 <= y2 else (y2, y1)
        return cls(left, top, right - left, bottom - top)

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Height / width. Tall boxes (doors) score above 1."""
        return self.height / (self.width + 1e-6)

    def as_xywh(self) -> Tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.width), int(self.height))

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.right), int(self.bottom))

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox.from_xyxy(
            min(self.x, other.x),
            min(self.y, other.y),
            max(self.right, other.right),
            max(self.bottom, other.bottom),
        )


@dataclass(frozen=True)
class Detection:
    """A single detected object.

    ``source`` records which detector produced it so the UI (and the tests) can
    tell a model prediction apart from a classical-CV fallback guess.
    """

    label: str
    confidence: float
    box: BoundingBox
    source: str = "fallback"
    direction: str | None = None

    @property
    def center_x(self) -> float:
        return self.box.center_x

    @property
    def center_y(self) -> float:
        return self.box.center_y


@dataclass(frozen=True)
class LightSource:
    """A bright blob - emergency lighting, a lit exit sign, or a window."""

    x: int
    y: int
    area: float
    brightness: float


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two boxes. Returns 0.0 when disjoint."""
    inter_w = min(a.right, b.right) - max(a.x, b.x)
    inter_h = min(a.bottom, b.bottom) - max(a.y, b.y)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    union = a.area + b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def non_max_suppression(
    detections: Sequence[Detection], iou_threshold: float = 0.45
) -> List[Detection]:
    """Greedy NMS, applied per label.

    Classical-CV detectors emit heavily overlapping candidates; without this the
    navigation layer counts the same door five times and skews the direction
    vote.
    """
    if not detections:
        return []
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")

    kept: List[Detection] = []
    ordered = sorted(detections, key=lambda d: (d.confidence, d.box.area), reverse=True)
    for candidate in ordered:
        if any(
            k.label == candidate.label and iou(k.box, candidate.box) > iou_threshold
            for k in kept
        ):
            continue
        kept.append(candidate)
    return kept
