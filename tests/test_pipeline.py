import numpy as np
import pytest

from conftest import blank_frame, draw_corridor, draw_door, draw_exit_sign
from emergency_path_finder.config import Settings
from emergency_path_finder.geometry import BoundingBox, Detection
from emergency_path_finder.navigation import Direction, Urgency
from emergency_path_finder.pipeline import PathFinder


@pytest.fixture
def finder(tmp_path) -> PathFinder:
    """A pipeline with no model available - the classical-CV configuration."""
    return PathFinder(
        settings=Settings(project_root=tmp_path, models_dir=tmp_path / "ml_models"),
        use_model=False,
    )


def test_pipeline_runs_without_a_trained_model(finder):
    assert finder.uses_model is False


def test_missing_weights_are_reported_not_raised(tmp_path):
    pipeline = PathFinder(
        settings=Settings(project_root=tmp_path, models_dir=tmp_path / "ml_models"),
        use_model=True,
    )
    assert pipeline.uses_model is False
    assert pipeline.model_error


def test_analyze_finds_an_exit_and_points_at_it(finder):
    frame = draw_exit_sign(blank_frame(), center=(560, 140))
    advice = finder.analyze(frame)
    assert advice.counts["exit"] >= 1
    assert advice.direction == Direction.RIGHT
    assert advice.urgency in {Urgency.CRITICAL, Urgency.HIGH}
    assert advice.target is not None and advice.target.label == "exit"


def test_analyze_on_an_empty_room_keeps_the_user_moving(finder, blank):
    advice = finder.analyze(blank)
    assert advice.direction in Direction.ALL
    assert advice.target is None
    assert advice.instruction


def test_analyze_uses_the_corridor_when_nothing_is_detected(finder):
    frame = draw_corridor(blank_frame(), vanishing=(150, 250))
    advice = finder.analyze(frame)
    if not any(advice.counts.values()):
        assert advice.vanishing_point is not None


def test_device_orientation_is_applied(finder):
    frame = draw_exit_sign(blank_frame(), center=(560, 140))
    upright = finder.analyze(frame, device_orientation_deg=0.0)
    tilted = finder.analyze(frame, device_orientation_deg=90.0)
    assert tilted.arrow_angle_deg == pytest.approx(
        (upright.arrow_angle_deg - 90.0) % 360.0, abs=1e-6
    )


def test_torch_is_advised_in_the_dark_and_not_in_daylight(finder, dark):
    assert finder.should_enable_torch(dark) is True
    assert finder.should_enable_torch(blank_frame(230)) is False


def test_detections_are_deduplicated(finder):
    duplicate = Detection("exit", 0.6, BoundingBox(10, 10, 50, 50), source="color")
    model_hit = Detection("exit", 0.6, BoundingBox(12, 12, 50, 50), source="yolo")
    kept = PathFinder._deduplicate([duplicate, model_hit])
    assert len(kept) == 1
    # The model prediction wins the tie thanks to its confidence bonus.
    assert kept[0].source == "yolo"


def test_deduplicate_never_pushes_confidence_above_one(finder):
    detections = [Detection("exit", 0.99, BoundingBox(0, 0, 10, 10), source="yolo")]
    assert PathFinder._deduplicate(detections)[0].confidence <= 1.0


def test_analyze_is_deterministic(finder):
    frame = draw_door(draw_exit_sign(blank_frame(), (560, 140)), center_x=180)
    assert finder.analyze(frame).as_dict() == finder.analyze(frame).as_dict()


def test_analyze_rejects_a_malformed_frame(finder):
    with pytest.raises((ValueError, TypeError)):
        finder.analyze(np.zeros((10, 10), dtype=np.uint8))
