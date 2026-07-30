import pytest

from emergency_path_finder.geometry import BoundingBox, Detection, LightSource
from emergency_path_finder.navigation import (
    Direction,
    NavigationHelper,
    Urgency,
)

SHAPE = (480, 640, 3)


def detection(label, cx, cy, w=60, h=60, confidence=0.8, direction=None):
    return Detection(
        label=label,
        confidence=confidence,
        box=BoundingBox(cx - w / 2, cy - h / 2, w, h),
        direction=direction,
    )


# ---------------------------------------------------------- target choice ---
def test_exit_outranks_stairs_and_doors():
    detections = [
        detection("door", 100, 240, confidence=0.99),
        detection("stairs", 200, 240, confidence=0.95),
        detection("exit", 320, 240, confidence=0.40),
    ]
    assert NavigationHelper.select_target(detections).label == "exit"


def test_stairs_outrank_doors():
    detections = [detection("door", 100, 240), detection("stairs", 200, 240)]
    assert NavigationHelper.select_target(detections).label == "stairs"


def test_most_confident_wins_within_a_class():
    weak = detection("exit", 100, 240, confidence=0.4)
    strong = detection("exit", 500, 240, confidence=0.9)
    assert NavigationHelper.select_target([weak, strong]) is strong


def test_no_detections_means_no_target():
    assert NavigationHelper.select_target([]) is None


# -------------------------------------------------------------- direction ---
@pytest.mark.parametrize(
    "center_x, expected",
    [
        (40, Direction.LEFT),
        (320, Direction.STRAIGHT),
        (600, Direction.RIGHT),
    ],
)
def test_direction_follows_target_position(center_x, expected):
    detections = [detection("exit", center_x, 240)]
    assert NavigationHelper.get_best_direction(detections, SHAPE) == expected


def test_small_offset_stays_straight():
    # 340 is 20 px right of centre - inside the dead zone, so no turn.
    detections = [detection("exit", 340, 240)]
    assert NavigationHelper.get_best_direction(detections, SHAPE) == Direction.STRAIGHT


@pytest.mark.parametrize(
    "stair_direction, expected",
    [("up", Direction.UPSTAIRS), ("down", Direction.DOWNSTAIRS)],
)
def test_stairs_produce_vertical_directions(stair_direction, expected):
    detections = [detection("stairs", 320, 240, direction=stair_direction)]
    assert NavigationHelper.get_best_direction(detections, SHAPE) == expected


def test_stairs_ahead_fall_back_to_horizontal_reasoning():
    detections = [detection("stairs", 60, 240, direction="ahead")]
    assert NavigationHelper.get_best_direction(detections, SHAPE) == Direction.LEFT


def test_vanishing_point_steers_when_nothing_is_detected():
    assert (
        NavigationHelper.get_best_direction([], SHAPE, vanishing_point=(60.0, 240.0))
        == Direction.LEFT
    )
    assert (
        NavigationHelper.get_best_direction([], SHAPE, vanishing_point=(600.0, 240.0))
        == Direction.RIGHT
    )


def test_empty_frame_means_keep_walking():
    assert NavigationHelper.get_best_direction([], SHAPE) == Direction.FORWARD


# ------------------------------------------------------------ arrow angle ---
def test_arrow_points_up_for_a_target_above_centre():
    target = detection("exit", 320, 100)
    assert NavigationHelper.arrow_angle_deg(target, SHAPE) == pytest.approx(0.0, abs=1e-6)


def test_arrow_points_right_for_a_target_to_the_right():
    target = detection("exit", 620, 240)
    assert NavigationHelper.arrow_angle_deg(target, SHAPE) == pytest.approx(90.0, abs=1e-6)


def test_arrow_points_left_for_a_target_to_the_left():
    target = detection("exit", 20, 240)
    assert NavigationHelper.arrow_angle_deg(target, SHAPE) == pytest.approx(270.0, abs=1e-6)


def test_arrow_angle_is_always_in_range():
    for cx in range(0, 640, 40):
        for cy in range(0, 480, 40):
            angle = NavigationHelper.arrow_angle_deg(detection("exit", cx, cy), SHAPE)
            assert 0.0 <= angle < 360.0


def test_device_orientation_rotates_the_arrow():
    target = detection("exit", 620, 240)
    rotated = NavigationHelper.arrow_angle_deg(target, SHAPE, device_orientation_deg=30.0)
    assert rotated == pytest.approx(60.0, abs=1e-6)


def test_orientation_wraps_around_zero():
    target = detection("exit", 320, 100)
    assert NavigationHelper.arrow_angle_deg(
        target, SHAPE, device_orientation_deg=45.0
    ) == pytest.approx(315.0, abs=1e-6)


def test_arrow_falls_back_to_the_vanishing_point():
    angle = NavigationHelper.arrow_angle_deg(None, SHAPE, vanishing_point=(620.0, 240.0))
    assert angle == pytest.approx(90.0, abs=1e-6)


def test_arrow_is_zero_with_nothing_to_aim_at():
    assert NavigationHelper.arrow_angle_deg(None, SHAPE) == 0.0


# ---------------------------------------------------------------- distance ---
def test_bigger_boxes_read_as_closer():
    near = BoundingBox(0, 0, 200, 300)
    far = BoundingBox(0, 0, 20, 30)
    assert NavigationHelper.estimate_distance_m(
        near, SHAPE
    ) < NavigationHelper.estimate_distance_m(far, SHAPE)


def test_distance_is_clamped_to_a_sane_range():
    huge = BoundingBox(0, 0, 640, 480)
    sliver = BoundingBox(0, 0, 640, 1)
    assert NavigationHelper.estimate_distance_m(huge, SHAPE) >= 0.5
    assert NavigationHelper.estimate_distance_m(sliver, SHAPE) <= 15.0


def test_degenerate_box_does_not_divide_by_zero():
    assert NavigationHelper.estimate_distance_m(BoundingBox(0, 0, 10, 0), SHAPE) == 15.0


def test_half_frame_object_reads_as_about_two_metres():
    half = BoundingBox(0, 0, 100, 240)
    assert NavigationHelper.estimate_distance_m(half, SHAPE) == pytest.approx(2.0, abs=0.1)


# ----------------------------------------------------------------- urgency ---
def test_visible_exit_in_decent_light_is_critical():
    assert (
        NavigationHelper.calculate_urgency_level(True, False, False, 0.8)
        == Urgency.CRITICAL
    )


def test_exit_in_darkness_is_only_high():
    assert (
        NavigationHelper.calculate_urgency_level(True, False, False, 0.1) == Urgency.HIGH
    )


def test_structure_without_an_exit_is_high():
    assert (
        NavigationHelper.calculate_urgency_level(False, True, False, 0.9) == Urgency.HIGH
    )
    assert (
        NavigationHelper.calculate_urgency_level(False, False, True, 0.9) == Urgency.HIGH
    )


def test_nothing_detected_but_good_light_is_medium():
    assert (
        NavigationHelper.calculate_urgency_level(False, False, False, 0.9)
        == Urgency.MEDIUM
    )


def test_nothing_detected_and_dark_is_low():
    assert (
        NavigationHelper.calculate_urgency_level(False, False, False, 0.1) == Urgency.LOW
    )


def test_urgency_rejects_out_of_range_light_quality():
    with pytest.raises(ValueError):
        NavigationHelper.calculate_urgency_level(True, False, False, 1.5)


# ------------------------------------------------------------------ advise ---
def test_advise_reports_an_exit_to_the_right():
    advice = NavigationHelper.advise(
        [detection("exit", 600, 200, confidence=0.9)], SHAPE, light_quality=0.7
    )
    assert advice.direction == Direction.RIGHT
    assert advice.urgency == Urgency.CRITICAL
    assert advice.target.label == "exit"
    assert advice.counts == {"exit": 1, "stairs": 0, "door": 0}
    assert "EXIT" in advice.instruction
    assert advice.distance_m is not None
    # Target is right of centre and slightly above it: forward-right quadrant.
    assert 0.0 < advice.arrow_angle_deg < 90.0


def test_advise_with_an_empty_frame():
    advice = NavigationHelper.advise([], SHAPE, light_quality=0.6)
    assert advice.direction == Direction.FORWARD
    assert advice.urgency == Urgency.MEDIUM
    assert advice.target is None
    assert advice.distance_m is None
    assert advice.arrow_angle_deg == 0.0


def test_advise_warns_when_it_is_too_dark_to_see():
    advice = NavigationHelper.advise([], SHAPE, light_quality=0.05)
    assert advice.urgency == Urgency.LOW
    assert "dark" in advice.instruction.lower()


def test_advise_steers_towards_emergency_lighting_when_blind():
    lights = [LightSource(x=610, y=120, area=900.0, brightness=0.95)]
    advice = NavigationHelper.advise([], SHAPE, light_quality=0.15, lights=lights)
    assert advice.direction == Direction.RIGHT
    assert advice.vanishing_point == (610.0, 120.0)


def test_lights_do_not_override_a_real_detection():
    lights = [LightSource(x=10, y=10, area=9000.0, brightness=1.0)]
    advice = NavigationHelper.advise(
        [detection("exit", 600, 200)], SHAPE, light_quality=0.7, lights=lights
    )
    assert advice.direction == Direction.RIGHT


def test_advice_serialises_to_plain_types():
    advice = NavigationHelper.advise(
        [detection("stairs", 320, 120, direction="up")], SHAPE, light_quality=0.5
    )
    payload = advice.as_dict()
    assert payload["direction"] == Direction.UPSTAIRS
    assert payload["target"] == "stairs"
    assert isinstance(payload["counts"], dict)
    assert set(payload) == {
        "direction",
        "instruction",
        "urgency",
        "arrow_angle_deg",
        "confidence",
        "light_quality",
        "target",
        "distance_m",
        "vanishing_point",
        "counts",
    }
