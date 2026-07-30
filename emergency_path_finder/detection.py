"""Classical computer-vision detectors.

These run with nothing but OpenCV and NumPy. They are the offline safety net:
they work when no model has been trained yet, when the phone is too slow to run
inference on every frame, and - most importantly - in smoke or darkness where a
model trained on clean daylight photos falls apart.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import DetectorConfig
from .geometry import BoundingBox, Detection, LightSource, non_max_suppression

__all__ = ["FallbackDetector", "enhance_low_light", "estimate_light_quality"]

# Standard exit-sign hues in OpenCV HSV (H in 0..179).
_GREEN_RANGE = (np.array([35, 60, 60]), np.array([85, 255, 255]))
_RED_RANGES = (
    (np.array([0, 60, 60]), np.array([10, 255, 255])),
    (np.array([170, 60, 60]), np.array([180, 255, 255])),
)


def _require_bgr(image: np.ndarray) -> np.ndarray:
    """Validate an incoming frame and return it unchanged.

    Callers routinely hand us ``cv2.imread`` output without checking for
    ``None``; failing here with a clear message beats a cryptic OpenCV error
    three functions deeper.
    """
    if image is None:
        raise ValueError("image is None - cv2.imread returns None for missing files")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"expected a numpy array, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR image, got shape {image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image has zero width or height")
    return image


def enhance_low_light(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE contrast boost, used before every edge-based detector."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def estimate_light_quality(image: np.ndarray) -> float:
    """Score usable visibility in ``[0, 1]``.

    Combines mean brightness with contrast: a uniformly grey frame (smoke) is
    bright but useless, so contrast has to carry part of the score.
    """
    _require_bgr(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    brightness = float(gray.mean())
    contrast = float(gray.std()) * 2.0  # std of 0.5 is already very high contrast
    quality = 0.6 * brightness + 0.4 * min(contrast, 1.0)
    return float(np.clip(quality, 0.0, 1.0))


class FallbackDetector:
    """Detects exits, doors, stairs and lights without a neural network."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()

    # ------------------------------------------------------------------
    # Exit signs
    # ------------------------------------------------------------------
    def detect_color_signs(self, image: np.ndarray) -> List[Detection]:
        """Find green/red exit signage by colour.

        Confidence is how completely the coloured pixels fill the candidate box:
        a real sign is a solid slab of colour, whereas a scattered reflection
        fills its bounding box poorly.
        """
        _require_bgr(image)
        frame_area = float(image.shape[0] * image.shape[1])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, *_GREEN_RANGE)
        for low, high in _RED_RANGES:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.config.min_sign_area_ratio * frame_area
        detections: List[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = BoundingBox(x, y, w, h)
            if box.area < min_area:
                continue
            fill = float(cv2.contourArea(contour)) / (box.area + 1e-6)
            size_score = min(box.area / (frame_area * 0.05), 1.0)
            confidence = float(np.clip(0.5 * fill + 0.5 * size_score, 0.0, 1.0))
            if confidence < self.config.confidence_threshold:
                continue
            detections.append(
                Detection(label="exit", confidence=confidence, box=box, source="color")
            )

        return non_max_suppression(detections, self.config.nms_iou_threshold)

    # ------------------------------------------------------------------
    # Doors
    # ------------------------------------------------------------------
    def detect_doors(self, image: np.ndarray) -> List[Detection]:
        """Find door-shaped regions: tall, narrow, roughly rectangular.

        Works in low light where signage is invisible, because it only needs
        edges rather than colour.
        """
        _require_bgr(image)
        frame_area = float(image.shape[0] * image.shape[1])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = enhance_low_light(gray)

        edges = cv2.Canny(enhanced, self.config.canny_low, self.config.canny_high)
        # A vertical kernel bridges the gaps in a door frame's two long edges
        # without merging them into neighbouring wall clutter.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.config.min_door_area_ratio * frame_area
        detections: List[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = BoundingBox(x, y, w, h)
            if box.area < min_area:
                continue
            ratio = box.aspect_ratio
            if not (
                self.config.door_min_aspect_ratio
                <= ratio
                <= self.config.door_max_aspect_ratio
            ):
                continue
            # Peak score at ratio ~2.1, the proportions of a standard doorway.
            shape_score = max(0.0, 1.0 - abs(ratio - 2.1) / 2.1)
            size_score = min(box.area / (frame_area * 0.15), 1.0)
            confidence = float(np.clip(0.6 * shape_score + 0.4 * size_score, 0.0, 1.0))
            if confidence < self.config.confidence_threshold:
                continue
            detections.append(
                Detection(label="door", confidence=confidence, box=box, source="edges")
            )

        return non_max_suppression(detections, self.config.nms_iou_threshold)

    # ------------------------------------------------------------------
    # Stairs
    # ------------------------------------------------------------------
    def detect_stairs_edges(self, image: np.ndarray) -> List[Detection]:
        """Find staircases from their stack of parallel tread edges.

        A single line segment means nothing; a staircase is a *run* of roughly
        parallel, near-horizontal segments at increasing heights. Requiring
        ``stairs_min_treads`` of them is what separates a staircase from a
        skirting board.
        """
        _require_bgr(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = enhance_low_light(gray)
        edges = cv2.Canny(enhanced, 30, 100)

        min_length = max(20, image.shape[1] // 12)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=min_length,
            maxLineGap=10,
        )
        if lines is None:
            return []

        treads: List[Tuple[float, BoundingBox]] = []
        max_angle = self.config.stairs_max_tread_angle_deg
        for line in lines:
            x1, y1, x2, y2 = (float(v) for v in line[0])
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            angle = min(angle, 180.0 - angle)  # fold to [0, 90]
            if angle > max_angle:
                continue  # too steep to be a tread edge
            treads.append(((y1 + y2) / 2.0, BoundingBox.from_xyxy(x1, y1, x2, y2)))

        if len(treads) < self.config.stairs_min_treads:
            return []

        # Group treads into runs separated by no more than a typical riser.
        treads.sort(key=lambda item: item[0])
        max_gap = max(8.0, image.shape[0] * 0.12)
        runs: List[List[Tuple[float, BoundingBox]]] = [[treads[0]]]
        for entry in treads[1:]:
            if entry[0] - runs[-1][-1][0] <= max_gap:
                runs[-1].append(entry)
            else:
                runs.append([entry])

        frame_area = float(image.shape[0] * image.shape[1])
        # Quantise tread heights before counting them: the top and bottom edge of
        # one physical tread are two Hough lines a few pixels apart, and counting
        # both would let a skirting board pass as a staircase.
        row_tolerance = max(6.0, image.shape[0] * 0.02)
        detections: List[Detection] = []
        for run in runs:
            distinct_rows = {round(y / row_tolerance) for y, _ in run}
            if len(distinct_rows) < self.config.stairs_min_treads:
                continue
            region = run[0][1]
            for _, extra in run[1:]:
                region = region.union(extra)
            # Pad thin line-unions into a usable region.
            region = BoundingBox(
                region.x,
                region.y,
                max(region.width, image.shape[1] * 0.05),
                max(region.height, image.shape[0] * 0.05),
            )
            tread_score = min(len(distinct_rows) / 8.0, 1.0)
            size_score = min(region.area / (frame_area * 0.2), 1.0)
            confidence = float(np.clip(0.7 * tread_score + 0.3 * size_score, 0.0, 1.0))
            if confidence < self.config.confidence_threshold:
                continue
            detections.append(
                Detection(
                    label="stairs",
                    confidence=confidence,
                    box=region,
                    source="edges",
                    direction=self.stair_direction(region, image.shape),
                )
            )

        return non_max_suppression(detections, self.config.nms_iou_threshold)

    @staticmethod
    def stair_direction(box: BoundingBox, image_shape: Tuple[int, ...]) -> str:
        """Guess whether a staircase leads up or down from where it sits.

        Stairs going up recede towards the horizon (upper half of the frame);
        stairs going down fall away below the camera.
        """
        height = image_shape[0]
        if box.center_y < height * 0.45:
            return "up"
        if box.center_y > height * 0.65:
            return "down"
        return "ahead"

    # ------------------------------------------------------------------
    # Lights
    # ------------------------------------------------------------------
    def detect_light_sources(self, image: np.ndarray) -> List[LightSource]:
        """Locate bright blobs: emergency lighting, lit signage, daylight."""
        _require_bgr(image)
        frame_area = float(image.shape[0] * image.shape[1])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]

        _, bright = cv2.threshold(
            value, self.config.brightness_threshold, 255, cv2.THRESH_BINARY
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.config.min_light_area_ratio * frame_area
        lights: List[LightSource] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            mask = np.zeros(value.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            brightness = float(cv2.mean(value, mask=mask)[0]) / 255.0
            lights.append(LightSource(x=cx, y=cy, area=area, brightness=brightness))

        lights.sort(key=lambda light: light.area, reverse=True)
        return lights

    # ------------------------------------------------------------------
    # Corridor geometry
    # ------------------------------------------------------------------
    def detect_vanishing_point(
        self, image: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Estimate the corridor vanishing point by voting.

        Averaging every pairwise line intersection (the naive approach) is
        dominated by outliers - two nearly parallel lines meet far outside the
        frame and drag the mean with them. Instead each intersection votes into
        a coarse grid, and the winning cell is refined by averaging only the
        votes inside it. That makes the estimate robust to clutter.
        """
        _require_bgr(image)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = enhance_low_light(gray)
        edges = cv2.Canny(enhanced, self.config.canny_low, self.config.canny_high)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=max(30, width // 8),
            maxLineGap=10,
        )
        if lines is None or len(lines) < 2:
            return None

        # Keep only oblique lines: perspective edges of a corridor are neither
        # horizontal nor vertical in the image plane.
        oblique: List[Tuple[float, float, float, float]] = []
        for line in lines:
            x1, y1, x2, y2 = (float(v) for v in line[0])
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
            if angle < 8.0 or angle > 172.0 or abs(angle - 90.0) < 8.0:
                continue
            oblique.append((x1, y1, x2, y2))

        if len(oblique) < 2:
            return None

        votes: List[Tuple[float, float]] = []
        for i in range(len(oblique)):
            x1, y1, x2, y2 = oblique[i]
            for j in range(i + 1, len(oblique)):
                x3, y3, x4, y4 = oblique[j]
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-6:
                    continue  # parallel
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                # Allow a small margin outside the frame: the vanishing point of
                # a corridor seen off-axis can sit just past the edge.
                if -0.25 * width <= px <= 1.25 * width and -0.25 * height <= py <= 1.25 * height:
                    votes.append((px, py))

        if not votes:
            return None

        bins = 16
        cell_w = (1.5 * width) / bins
        cell_h = (1.5 * height) / bins
        buckets: dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        for px, py in votes:
            key = (int((px + 0.25 * width) // cell_w), int((py + 0.25 * height) // cell_h))
            buckets.setdefault(key, []).append((px, py))

        best = max(buckets.values(), key=len)
        return (
            float(np.mean([p[0] for p in best])),
            float(np.mean([p[1] for p in best])),
        )

    def estimate_relative_depth(self, image: np.ndarray) -> np.ndarray:
        """Coarse monocular "nearness" map, normalised to ``[0, 1]``.

        There is no true depth here - it is two cheap monocular cues combined:
        ground-plane position (lower in the frame is nearer) and local detail
        (near surfaces are sharper and more textured than distant ones). Good
        enough to tell "wall in your face" from "corridor continues", which is
        all the navigation layer asks of it.
        """
        _require_bgr(image)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detail = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        detail = cv2.GaussianBlur(np.abs(detail), (15, 15), 0)
        detail_max = float(detail.max())
        detail = detail / detail_max if detail_max > 1e-6 else np.zeros_like(detail)

        # Ground-plane prior: 0 at the top of the frame, 1 at the bottom.
        rows = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1)
        prior = np.repeat(rows, width, axis=1)

        depth = 0.6 * prior + 0.4 * detail
        return np.clip(depth, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    def detect_all(self, image: np.ndarray) -> List[Detection]:
        """Run every fallback detector and return one merged list."""
        detections = self.detect_color_signs(image)
        detections += self.detect_stairs_edges(image)
        detections += self.detect_doors(image)
        return detections
