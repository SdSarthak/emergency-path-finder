import pytest

from emergency_path_finder.geometry import (
    BoundingBox,
    Detection,
    iou,
    non_max_suppression,
)


def box(x, y, w, h):
    return BoundingBox(x, y, w, h)


def test_bounding_box_derived_properties():
    b = box(10, 20, 40, 60)
    assert b.right == 50
    assert b.bottom == 80
    assert b.center_x == 30
    assert b.center_y == 50
    assert b.area == 2400
    assert b.aspect_ratio == pytest.approx(1.5, rel=1e-4)


def test_from_xyxy_normalises_reversed_corners():
    assert BoundingBox.from_xyxy(50, 80, 10, 20) == box(10, 20, 40, 60)


def test_negative_dimensions_rejected():
    with pytest.raises(ValueError):
        BoundingBox(0, 0, -5, 10)


def test_as_xywh_and_xyxy():
    b = box(1.6, 2.4, 10.5, 20.5)
    assert b.as_xywh() == (1, 2, 10, 20)
    assert b.as_xyxy() == (1, 2, 12, 22)


def test_union_covers_both_boxes():
    merged = box(0, 0, 10, 10).union(box(20, 30, 10, 10))
    assert merged.as_xyxy() == (0, 0, 30, 40)


def test_iou_identical_boxes_is_one():
    assert iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou(box(0, 0, 10, 10), box(100, 100, 10, 10)) == 0.0


def test_iou_touching_edges_is_zero():
    assert iou(box(0, 0, 10, 10), box(10, 0, 10, 10)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes sharing a 5x10 strip: 50 / (100 + 100 - 50).
    assert iou(box(0, 0, 10, 10), box(5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_nms_keeps_highest_confidence_of_a_cluster():
    strong = Detection("exit", 0.9, box(0, 0, 10, 10))
    weak = Detection("exit", 0.4, box(1, 1, 10, 10))
    kept = non_max_suppression([weak, strong], iou_threshold=0.4)
    assert kept == [strong]


def test_nms_does_not_merge_across_labels():
    a = Detection("exit", 0.9, box(0, 0, 10, 10))
    b = Detection("door", 0.8, box(0, 0, 10, 10))
    kept = non_max_suppression([a, b])
    assert {d.label for d in kept} == {"exit", "door"}


def test_nms_keeps_distinct_objects():
    a = Detection("door", 0.9, box(0, 0, 10, 10))
    b = Detection("door", 0.8, box(200, 0, 10, 10))
    assert len(non_max_suppression([a, b])) == 2


def test_nms_on_empty_input():
    assert non_max_suppression([]) == []


def test_nms_rejects_out_of_range_threshold():
    with pytest.raises(ValueError):
        non_max_suppression([Detection("exit", 0.5, box(0, 0, 1, 1))], iou_threshold=1.5)
