import numpy as np
import pytest

from conftest import (
    FRAME_SIZE,
    blank_frame,
    draw_corridor,
    draw_door,
    draw_exit_sign,
    draw_stairs,
)
from emergency_path_finder.config import DetectorConfig
from emergency_path_finder.detection import (
    FallbackDetector,
    enhance_low_light,
    estimate_light_quality,
)


@pytest.fixture
def detector() -> FallbackDetector:
    return FallbackDetector()


# ----------------------------------------------------------------- guards ---
@pytest.mark.parametrize(
    "bad, expected",
    [
        (None, ValueError),
        ("not-an-image", TypeError),
        (np.zeros((10, 10), dtype=np.uint8), ValueError),
        (np.zeros((10, 10, 4), dtype=np.uint8), ValueError),
        (np.zeros((0, 10, 3), dtype=np.uint8), ValueError),
    ],
)
def test_detectors_reject_malformed_input(detector, bad, expected):
    with pytest.raises(expected):
        detector.detect_color_signs(bad)


# ------------------------------------------------------------ exit signs ---
def test_finds_green_exit_sign(detector, exit_scene):
    signs = detector.detect_color_signs(exit_scene)
    assert len(signs) == 1
    sign = signs[0]
    assert sign.label == "exit"
    assert sign.source == "color"
    assert sign.center_x == pytest.approx(540, abs=15)
    assert sign.center_y == pytest.approx(140, abs=15)
    assert 0.0 < sign.confidence <= 1.0


def test_finds_red_exit_sign(detector):
    frame = blank_frame()
    frame[100:150, 300:400] = (0, 0, 255)  # BGR red
    assert len(detector.detect_color_signs(frame)) == 1


def test_no_signs_in_a_featureless_room(detector, blank):
    assert detector.detect_color_signs(blank) == []


def test_tiny_colour_speck_is_not_a_sign(detector):
    frame = blank_frame()
    frame[100:104, 300:304] = (0, 255, 0)
    assert detector.detect_color_signs(frame) == []


def test_overlapping_signs_are_suppressed(detector):
    frame = draw_exit_sign(blank_frame(), center=(320, 200), size=(120, 60))
    assert len(detector.detect_color_signs(frame)) == 1


# ----------------------------------------------------------------- doors ---
def test_finds_a_door_shaped_region(detector, door_scene):
    doors = detector.detect_doors(door_scene)
    assert doors, "expected the tall rectangle to be detected as a door"
    best = max(doors, key=lambda d: d.confidence)
    assert best.label == "door"
    assert best.center_x == pytest.approx(200, abs=40)
    assert best.box.aspect_ratio > 1.4


def test_wide_flat_shape_is_not_a_door(detector):
    frame = blank_frame()
    frame[200:250, 100:500] = (240, 240, 240)  # a wide banner, not a doorway
    assert detector.detect_doors(frame) == []


# ---------------------------------------------------------------- stairs ---
def test_finds_a_run_of_treads(detector, stairs_scene):
    stairs = detector.detect_stairs_edges(stairs_scene)
    assert stairs, "a stack of parallel tread edges should read as stairs"
    assert stairs[0].label == "stairs"
    assert stairs[0].direction in {"up", "down", "ahead"}


def test_two_lines_are_not_a_staircase(detector):
    frame = blank_frame()
    frame[200:204, 100:500] = (240, 240, 240)
    frame[240:244, 100:500] = (240, 240, 240)
    assert detector.detect_stairs_edges(frame) == []


def test_min_tread_count_is_configurable(detector):
    frame = blank_frame()
    for index in range(4):
        frame[200 + index * 30 : 204 + index * 30, 100:500] = (240, 240, 240)

    strict = FallbackDetector(DetectorConfig(stairs_min_treads=50))
    assert strict.detect_stairs_edges(frame) == []
    assert detector.detect_stairs_edges(frame)


def test_stair_direction_follows_vertical_position():
    from emergency_path_finder.geometry import BoundingBox

    shape = (*FRAME_SIZE, 3)
    assert FallbackDetector.stair_direction(BoundingBox(0, 0, 10, 10), shape) == "up"
    assert (
        FallbackDetector.stair_direction(BoundingBox(0, 460, 10, 10), shape) == "down"
    )
    assert (
        FallbackDetector.stair_direction(BoundingBox(0, 235, 10, 10), shape) == "ahead"
    )


# ---------------------------------------------------------------- lights ---
def test_finds_a_bright_blob(detector):
    import cv2

    frame = blank_frame()
    cv2.circle(frame, (320, 100), 30, (255, 255, 255), thickness=cv2.FILLED)
    lights = detector.detect_light_sources(frame)
    assert len(lights) == 1
    assert lights[0].x == pytest.approx(320, abs=5)
    assert lights[0].y == pytest.approx(100, abs=5)
    assert lights[0].brightness == pytest.approx(1.0, abs=0.05)


def test_no_lights_in_the_dark(detector, dark):
    assert detector.detect_light_sources(dark) == []


def test_lights_are_returned_largest_first(detector):
    import cv2

    frame = blank_frame()
    cv2.circle(frame, (100, 100), 10, (255, 255, 255), thickness=cv2.FILLED)
    cv2.circle(frame, (400, 300), 40, (255, 255, 255), thickness=cv2.FILLED)
    lights = detector.detect_light_sources(frame)
    assert len(lights) == 2
    assert lights[0].area > lights[1].area


# ------------------------------------------------------- corridor geometry ---
def test_vanishing_point_recovers_the_convergence(detector):
    frame = draw_corridor(blank_frame(), vanishing=(400, 240))
    point = detector.detect_vanishing_point(frame)
    assert point is not None
    assert point[0] == pytest.approx(400, abs=25)
    assert point[1] == pytest.approx(240, abs=25)


def test_vanishing_point_off_centre(detector):
    frame = draw_corridor(blank_frame(), vanishing=(180, 260))
    point = detector.detect_vanishing_point(frame)
    assert point is not None
    assert point[0] == pytest.approx(180, abs=30)


def test_no_vanishing_point_without_edges(detector, blank):
    assert detector.detect_vanishing_point(blank) is None


def test_relative_depth_is_normalised_and_near_at_the_bottom(detector, blank):
    depth = detector.estimate_relative_depth(blank)
    assert depth.shape == blank.shape[:2]
    assert depth.dtype == np.float32
    assert 0.0 <= float(depth.min()) and float(depth.max()) <= 1.0
    # The ground-plane prior must make the bottom of the frame read as nearer.
    assert float(depth[-10:].mean()) > float(depth[:10].mean())


# ------------------------------------------------------------- exposure ---
def test_light_quality_ranks_scenes_correctly(dark, blank):
    bright = blank_frame(230)
    assert estimate_light_quality(dark) < estimate_light_quality(blank)
    assert estimate_light_quality(blank) < estimate_light_quality(bright)
    for frame in (dark, blank, bright):
        assert 0.0 <= estimate_light_quality(frame) <= 1.0


def test_low_light_enhancement_increases_contrast(dark):
    import cv2

    gray = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)
    gray[100:200, 100:200] = 14  # a faint object in near-darkness
    assert enhance_low_light(gray).std() > gray.std()


# ------------------------------------------------------------- detect_all ---
def test_detect_all_combines_every_detector(detector):
    frame = draw_stairs(draw_door(draw_exit_sign(blank_frame(), (540, 100)), 160))
    labels = {d.label for d in detector.detect_all(frame)}
    assert "exit" in labels
    assert len(labels) >= 2


def test_detect_all_is_deterministic(detector, exit_scene):
    first = detector.detect_all(exit_scene)
    second = detector.detect_all(exit_scene)
    assert first == second
