"""Turns detections into something a panicking person can act on.

The mobile app mirrors this logic in ``flutter_app/lib/services/navigation_service.dart``;
keeping the reference implementation here means the thresholds can be tuned and
regression-tested on a laptop instead of on a handset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry import BoundingBox, Detection, LightSource

__all__ = [
    "Direction",
    "Urgency",
    "NavigationAdvice",
    "NavigationHelper",
]


class Direction:
    """The instruction set the arrow overlay can render."""

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STRAIGHT = "STRAIGHT"
    UPSTAIRS = "UPSTAIRS"
    DOWNSTAIRS = "DOWNSTAIRS"
    FORWARD = "FORWARD"

    ALL = (LEFT, RIGHT, STRAIGHT, UPSTAIRS, DOWNSTAIRS, FORWARD)


class Urgency:
    """How confident we are that a usable escape route is in view."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    ALL = (CRITICAL, HIGH, MEDIUM, LOW)


@dataclass(frozen=True)
class NavigationAdvice:
    """The complete answer for one frame."""

    direction: str
    instruction: str
    urgency: str
    arrow_angle_deg: float
    confidence: float
    light_quality: float
    target: Optional[Detection] = None
    distance_m: Optional[float] = None
    vanishing_point: Optional[Tuple[float, float]] = None
    counts: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        """Flat, JSON-friendly view - what the CLI prints and the app consumes."""
        return {
            "direction": self.direction,
            "instruction": self.instruction,
            "urgency": self.urgency,
            "arrow_angle_deg": round(self.arrow_angle_deg, 2),
            "confidence": round(self.confidence, 3),
            "light_quality": round(self.light_quality, 3),
            "target": self.target.label if self.target else None,
            "distance_m": round(self.distance_m, 2) if self.distance_m is not None else None,
            "vanishing_point": (
                [round(self.vanishing_point[0], 1), round(self.vanishing_point[1], 1)]
                if self.vanishing_point
                else None
            ),
            "counts": dict(self.counts),
        }


class NavigationHelper:
    """Pure functions over detections - no OpenCV, no I/O, trivially testable."""

    #: Fraction of frame width that a target must be off-centre before we call
    #: a turn. Below this it is "straight ahead" and turning would be noise.
    TURN_DEADZONE = 0.12

    @staticmethod
    def _sort_by_confidence(detections: Sequence[Detection]) -> List[Detection]:
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    @classmethod
    def select_target(cls, detections: Sequence[Detection]) -> Optional[Detection]:
        """Pick the one thing to steer towards.

        Priority is exit signage, then stairs, then doors - a marked exit beats a
        plausible one. Within a class the most confident detection wins.
        """
        for label in ("exit", "stairs", "door"):
            candidates = [d for d in detections if d.label == label]
            if candidates:
                return cls._sort_by_confidence(candidates)[0]
        return None

    @classmethod
    def get_best_direction(
        cls,
        detections: Sequence[Detection],
        image_shape: Tuple[int, ...],
        vanishing_point: Optional[Tuple[float, float]] = None,
    ) -> str:
        """Recommend a direction from the current detections.

        Falls back to the corridor vanishing point when nothing is detected,
        which is the whole point of "pathfinding without signs".
        """
        width = image_shape[1]
        center_x = width / 2.0
        deadzone = width * cls.TURN_DEADZONE

        target = cls.select_target(detections)
        if target is not None:
            if target.label == "stairs":
                direction = target.direction or "ahead"
                if direction == "up":
                    return Direction.UPSTAIRS
                if direction == "down":
                    return Direction.DOWNSTAIRS
            offset = target.center_x - center_x
            if offset < -deadzone:
                return Direction.LEFT
            if offset > deadzone:
                return Direction.RIGHT
            return Direction.STRAIGHT

        if vanishing_point is not None:
            offset = vanishing_point[0] - center_x
            if offset < -deadzone:
                return Direction.LEFT
            if offset > deadzone:
                return Direction.RIGHT
            return Direction.STRAIGHT

        return Direction.FORWARD

    @classmethod
    def arrow_angle_deg(
        cls,
        target: Optional[Detection],
        image_shape: Tuple[int, ...],
        device_orientation_deg: float = 0.0,
        vanishing_point: Optional[Tuple[float, float]] = None,
    ) -> float:
        """Bearing for the on-screen arrow, in degrees clockwise from "up".

        0 means straight ahead, +90 hard right, 270 hard left. Matches
        ``NavigationService.calculateArrowAngle`` in the Flutter app.
        """
        height, width = image_shape[0], image_shape[1]
        if target is not None:
            point = (target.center_x, target.center_y)
        elif vanishing_point is not None:
            point = vanishing_point
        else:
            return 0.0

        delta_x = point[0] - width / 2.0
        delta_y = point[1] - height / 2.0
        angle = math.degrees(math.atan2(delta_x, -delta_y))
        return (angle - device_orientation_deg) % 360.0

    @staticmethod
    def estimate_distance_m(box: BoundingBox, image_shape: Tuple[int, ...]) -> float:
        """Very rough distance from apparent size.

        Assumes an object roughly 2 m tall. Only the ordering is meaningful -
        "5 m" versus "1.5 m" tells the user whether to keep walking, which is
        the decision the number exists to support.
        """
        height = float(image_shape[0])
        if box.height <= 1e-6:
            return 15.0
        # Pinhole: distance is inversely proportional to apparent height.
        # The constant is calibrated so an object filling half the frame reads
        # as ~2 m, which matches a doorway at arm's reach on a phone camera.
        ratio = box.height / height
        distance = 1.0 / max(ratio, 1e-3)
        return float(min(max(distance, 0.5), 15.0))

    @staticmethod
    def calculate_urgency_level(
        has_exit: bool,
        has_stairs: bool,
        has_doors: bool,
        light_quality: float,
    ) -> str:
        """Classify the situation.

        CRITICAL is deliberately the *best* case here: it means a marked exit is
        visible and the user should move now.
        """
        if not 0.0 <= light_quality <= 1.0:
            raise ValueError(f"light_quality must be in [0, 1], got {light_quality}")
        if has_exit and light_quality > 0.4:
            return Urgency.CRITICAL
        if has_exit or has_stairs or has_doors:
            return Urgency.HIGH
        if light_quality > 0.5:
            return Urgency.MEDIUM
        return Urgency.LOW

    @classmethod
    def instruction_for(
        cls,
        target: Optional[Detection],
        direction: str,
        image_shape: Tuple[int, ...],
        light_quality: float,
    ) -> str:
        """Short, imperative text for the status bar."""
        if target is None:
            if direction in (Direction.LEFT, Direction.RIGHT):
                return f"No exit in view - corridor bends {direction.lower()}"
            if light_quality < 0.25:
                return "Too dark to see - move slowly, keep a hand on the wall"
            return "Searching for exits - keep moving forward"

        distance = cls.estimate_distance_m(target.box, image_shape)
        if target.label == "exit":
            if direction == Direction.STRAIGHT:
                return f"EXIT AHEAD - {distance:.0f} m, go straight"
            return f"EXIT FOUND - {distance:.0f} m, go {direction.lower()}"
        if target.label == "stairs":
            where = target.direction or "ahead"
            caution = "clear" if target.confidence > 0.6 else "take care"
            return f"Stairs {where} at {distance:.0f} m - {caution}"
        return f"Door at {distance:.0f} m - go {direction.lower()}"

    @classmethod
    def advise(
        cls,
        detections: Sequence[Detection],
        image_shape: Tuple[int, ...],
        light_quality: float,
        lights: Optional[Sequence[LightSource]] = None,
        vanishing_point: Optional[Tuple[float, float]] = None,
        device_orientation_deg: float = 0.0,
    ) -> NavigationAdvice:
        """Fuse everything known about a frame into one instruction."""
        labels = [d.label for d in detections]
        target = cls.select_target(detections)
        direction = cls.get_best_direction(detections, image_shape, vanishing_point)

        # A bright blob with nothing else detected is still information: in a
        # smoke-filled corridor the emergency lighting marks the route.
        if target is None and lights and direction == Direction.FORWARD:
            brightest = max(lights, key=lambda light: light.area)
            direction = cls.get_best_direction(
                detections, image_shape, (float(brightest.x), float(brightest.y))
            )
            vanishing_point = vanishing_point or (float(brightest.x), float(brightest.y))

        urgency = cls.calculate_urgency_level(
            has_exit="exit" in labels,
            has_stairs="stairs" in labels,
            has_doors="door" in labels,
            light_quality=light_quality,
        )
        angle = cls.arrow_angle_deg(
            target, image_shape, device_orientation_deg, vanishing_point
        )
        counts = {
            label: labels.count(label) for label in ("exit", "stairs", "door")
        }
        return NavigationAdvice(
            direction=direction,
            instruction=cls.instruction_for(target, direction, image_shape, light_quality),
            urgency=urgency,
            arrow_angle_deg=angle,
            confidence=target.confidence if target else 0.0,
            light_quality=light_quality,
            target=target,
            distance_m=(
                cls.estimate_distance_m(target.box, image_shape) if target else None
            ),
            vanishing_point=vanishing_point,
            counts=counts,
        )
