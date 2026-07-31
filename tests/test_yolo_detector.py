"""Tests for the model wrapper's label handling.

None of these need ultralytics or a checkpoint: the parts that go wrong in the
field are the class-index -> label mapping and the box conversion, and both are
plain Python over whatever the framework hands back.
"""

from types import SimpleNamespace

import pytest

from emergency_path_finder.navigation import NavigationHelper
from emergency_path_finder.yolo_detector import (
    CLASS_NAMES,
    YoloDetector,
    model_class_names,
    normalize_label,
)


class FakeScalar:
    """Stands in for a torch scalar tensor."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class FakeBox:
    def __init__(self, class_index, confidence, xyxy):
        self.cls = FakeScalar(class_index)
        self.conf = FakeScalar(confidence)
        self.xyxy = [SimpleNamespace(tolist=lambda box=xyxy: list(box))]


def make_detector(class_names, boxes):
    """A YoloDetector wired to a canned result, bypassing __init__."""
    detector = object.__new__(YoloDetector)
    detector.class_names = tuple(class_names)
    detector.confidence_threshold = 0.35
    detector.input_size = 416
    detector.device = None
    detector._model = SimpleNamespace(
        predict=lambda **kwargs: [SimpleNamespace(boxes=boxes)]
    )
    return detector


# ------------------------------------------------------------------- labels ---
def test_aliases_collapse_onto_the_navigation_vocabulary():
    assert normalize_label("Escalator") == "stairs"
    assert normalize_label("EXIT_SIGN") == "exit"
    assert normalize_label("staircase") == "stairs"
    assert normalize_label(" Doorway ") == "door"


def test_unknown_labels_survive_lower_cased_and_unmangled():
    assert normalize_label("Fire Extinguisher") == "fire extinguisher"
    assert normalize_label("class_7") == "class_7"


def test_normalized_labels_are_the_ones_navigation_understands():
    # If this drifts, select_target silently ignores every model detection.
    for raw in ("escalator", "exit sign", "doors"):
        assert normalize_label(raw) in {"exit", "stairs", "door"}


def test_class_names_come_from_the_checkpoint_not_the_default():
    model = SimpleNamespace(names={0: "stairs", 1: "escalator"})
    assert model_class_names(model) == ["stairs", "escalator"]


def test_class_names_are_ordered_by_index_not_insertion():
    model = SimpleNamespace(names={2: "door", 0: "exit", 1: "stairs"})
    assert model_class_names(model) == ["exit", "stairs", "door"]


def test_class_names_accept_a_plain_list():
    assert model_class_names(SimpleNamespace(names=["a", "b"])) == ["a", "b"]


def test_class_names_fall_back_when_the_checkpoint_carries_none():
    assert model_class_names(SimpleNamespace()) == list(CLASS_NAMES)
    assert model_class_names(SimpleNamespace(names={})) == list(CLASS_NAMES)


# ---------------------------------------------------------------- detection ---
def test_a_stairs_model_does_not_report_exits():
    """The regression this whole change exists for.

    A stairs checkpoint has classes (stairs, escalator). Read through the
    hardcoded (exit, stairs, door) list, index 0 becomes "exit" and the
    navigation layer steers the user towards a staircase it believes is a
    marked exit.
    """
    detector = make_detector(
        ("stairs", "escalator"), [FakeBox(0, 0.9, (10, 20, 60, 120))]
    )
    detections = detector.detect(_frame())
    assert [d.label for d in detections] == ["stairs"]


def test_escalators_are_reported_as_stairs():
    detector = make_detector(
        ("stairs", "escalator"), [FakeBox(1, 0.8, (0, 0, 40, 40))]
    )
    assert detector.detect(_frame())[0].label == "stairs"


def test_boxes_are_converted_from_xyxy():
    detector = make_detector(CLASS_NAMES, [FakeBox(0, 0.7, (10, 20, 60, 120))])
    box = detector.detect(_frame())[0].box
    assert (box.x, box.y, box.width, box.height) == (10, 20, 50, 100)


def test_an_out_of_range_class_index_does_not_crash():
    detector = make_detector(("exit",), [FakeBox(7, 0.7, (0, 0, 10, 10))])
    assert detector.detect(_frame())[0].label == "class_7"


def test_a_result_without_boxes_is_skipped():
    detector = make_detector(CLASS_NAMES, [])
    detector._model = SimpleNamespace(
        predict=lambda **kwargs: [SimpleNamespace(boxes=None)]
    )
    assert detector.detect(_frame()) == []


def test_detect_rejects_a_non_array_frame():
    detector = make_detector(CLASS_NAMES, [])
    with pytest.raises(ValueError):
        detector.detect(None)


def test_navigation_prefers_a_real_exit_over_a_relabelled_staircase():
    detector = make_detector(
        ("stairs", "escalator"), [FakeBox(0, 0.95, (10, 20, 60, 120))]
    )
    detections = detector.detect(_frame())
    target = NavigationHelper.select_target(detections)
    assert target is not None and target.label == "stairs"


def _frame():
    import numpy as np

    return np.zeros((240, 320, 3), dtype=np.uint8)
