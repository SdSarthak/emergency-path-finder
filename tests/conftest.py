"""Synthetic scene builders.

Every fixture here draws its own frame with NumPy/OpenCV, so the suite is fully
deterministic and needs no dataset download.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FRAME_SIZE: Tuple[int, int] = (480, 640)  # height, width


def blank_frame(value: int = 60) -> np.ndarray:
    """A featureless mid-grey room."""
    return np.full((FRAME_SIZE[0], FRAME_SIZE[1], 3), value, dtype=np.uint8)


def draw_exit_sign(
    frame: np.ndarray, center: Tuple[int, int], size: Tuple[int, int] = (90, 45)
) -> np.ndarray:
    """Paint a saturated green rectangle - the colour signature of exit signage."""
    frame = frame.copy()
    half_w, half_h = size[0] // 2, size[1] // 2
    cv2.rectangle(
        frame,
        (center[0] - half_w, center[1] - half_h),
        (center[0] + half_w, center[1] + half_h),
        (0, 255, 0),
        thickness=cv2.FILLED,
    )
    return frame


def draw_door(
    frame: np.ndarray, center_x: int, width: int = 90, height: int = 200
) -> np.ndarray:
    """Paint a tall, high-contrast rectangle standing on the floor line."""
    frame = frame.copy()
    top = FRAME_SIZE[0] - height - 40
    cv2.rectangle(
        frame,
        (center_x - width // 2, top),
        (center_x + width // 2, top + height),
        (230, 230, 230),
        thickness=cv2.FILLED,
    )
    cv2.rectangle(
        frame,
        (center_x - width // 2, top),
        (center_x + width // 2, top + height),
        (10, 10, 10),
        thickness=3,
    )
    return frame


def draw_stairs(frame: np.ndarray, treads: int = 7) -> np.ndarray:
    """Paint a stack of parallel near-horizontal tread edges."""
    frame = frame.copy()
    top = 220
    spacing = 26
    for index in range(treads):
        y = top + index * spacing
        inset = index * 14
        cv2.line(
            frame,
            (120 + inset, y),
            (FRAME_SIZE[1] - 120 - inset, y),
            (240, 240, 240),
            thickness=4,
        )
    return frame


def draw_corridor(frame: np.ndarray, vanishing: Tuple[int, int]) -> np.ndarray:
    """Paint four perspective lines converging on a known point."""
    frame = frame.copy()
    height, width = FRAME_SIZE
    corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    for corner in corners:
        cv2.line(frame, corner, vanishing, (235, 235, 235), thickness=3)
    return frame


@pytest.fixture
def blank() -> np.ndarray:
    return blank_frame()


@pytest.fixture
def dark() -> np.ndarray:
    return blank_frame(value=6)


@pytest.fixture
def exit_scene() -> np.ndarray:
    """Exit sign clearly to the right of frame centre."""
    return draw_exit_sign(blank_frame(), center=(540, 140))


@pytest.fixture
def door_scene() -> np.ndarray:
    return draw_door(blank_frame(), center_x=200)


@pytest.fixture
def stairs_scene() -> np.ndarray:
    return draw_stairs(blank_frame())


@pytest.fixture
def corridor_scene() -> np.ndarray:
    return draw_corridor(blank_frame(), vanishing=(400, 240))
