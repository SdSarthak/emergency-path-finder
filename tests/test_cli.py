import json

import cv2
import pytest

from conftest import blank_frame, draw_exit_sign
from emergency_path_finder.cli import build_parser, main
from emergency_path_finder.geometry import BoundingBox, Detection
from emergency_path_finder.navigation import NavigationHelper
from emergency_path_finder.visualize import draw_advice, draw_detections, render


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "corridor.png"
    cv2.imwrite(str(path), draw_exit_sign(blank_frame(), center=(560, 140)))
    return path


def test_parser_requires_a_source():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_rejects_two_sources():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--image", "a.jpg", "--video", "b.mp4"])


def test_camera_defaults_to_index_zero():
    assert build_parser().parse_args(["--camera"]).camera == 0
    assert build_parser().parse_args(["--camera", "2"]).camera == 2


def test_image_run_prints_json(sample_image, capsys):
    exit_code = main(
        ["--image", str(sample_image), "--no-model", "--no-display", "--json"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["direction"] in {"LEFT", "RIGHT", "STRAIGHT", "FORWARD"}
    assert payload["counts"]["exit"] >= 1


def test_image_run_prints_a_human_report(sample_image, capsys):
    assert main(["--image", str(sample_image), "--no-model", "--no-display"]) == 0
    out = capsys.readouterr().out
    assert "Direction" in out
    assert "Instruction" in out


def test_image_run_can_save_an_annotated_copy(sample_image, tmp_path):
    output = tmp_path / "out" / "annotated.png"
    main(
        [
            "--image",
            str(sample_image),
            "--no-model",
            "--no-display",
            "--save",
            str(output),
        ]
    )
    assert output.exists()
    assert cv2.imread(str(output)) is not None


def test_missing_image_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit):
        main(["--image", str(tmp_path / "nope.png"), "--no-model", "--no-display"])


def test_benchmark_reports_every_stage(sample_image, capsys):
    assert (
        main(
            [
                "--benchmark",
                str(sample_image),
                "--no-model",
                "--no-display",
                "--iterations",
                "1",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "full pipeline" in out
    assert "fps" in out


# --------------------------------------------------------------- drawing ---
def test_drawing_helpers_do_not_mutate_the_input():
    frame = blank_frame()
    original = frame.copy()
    detections = [Detection("exit", 0.9, BoundingBox(10, 10, 40, 40))]
    advice = NavigationHelper.advise(detections, frame.shape, light_quality=0.6)

    canvas = render(frame, detections, advice)
    assert (frame == original).all()
    assert canvas.shape == frame.shape
    assert not (canvas == frame).all(), "the overlay should visibly change the frame"


def test_draw_detections_handles_an_empty_list():
    frame = blank_frame()
    assert (draw_detections(frame, []) == frame).all()


def test_draw_advice_marks_the_vanishing_point():
    frame = blank_frame()
    advice = NavigationHelper.advise(
        [], frame.shape, light_quality=0.6, vanishing_point=(320.0, 200.0)
    )
    assert draw_advice(frame, advice).shape == frame.shape
