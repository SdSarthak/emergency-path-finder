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


def test_an_undecodable_file_is_distinguished_from_a_missing_one(tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"this is not a png")
    with pytest.raises(SystemExit) as excinfo:
        main(["--image", str(corrupt), "--no-model", "--no-display"])
    assert "decode" in str(excinfo.value)


def test_a_directory_is_not_mistaken_for_an_image(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["--image", str(tmp_path), "--no-model", "--no-display"])
    assert "directory" in str(excinfo.value)


def test_a_missing_model_path_is_fatal_not_silently_ignored(sample_image, tmp_path):
    # Falling back to classical CV here hands the user fallback output that
    # reads exactly like model output.
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--image",
                str(sample_image),
                "--model",
                str(tmp_path / "absent.pt"),
                "--no-display",
            ]
        )
    assert "absent.pt" in str(excinfo.value)


def test_model_and_no_model_together_are_rejected(sample_image, tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not-a-checkpoint")
    with pytest.raises(SystemExit):
        main(
            [
                "--image",
                str(sample_image),
                "--model",
                str(weights),
                "--no-model",
                "--no-display",
            ]
        )


@pytest.mark.parametrize("flag", ["--every", "--iterations"])
@pytest.mark.parametrize("value", ["0", "-1", "half"])
def test_non_positive_counts_are_rejected(flag, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--image", "a.png", flag, value])


def test_json_output_stays_parseable_when_saving(sample_image, tmp_path, capsys):
    # The "Saved:" line used to go to stdout, so `... --json | jq` broke.
    output = tmp_path / "annotated.png"
    main(
        [
            "--image",
            str(sample_image),
            "--no-model",
            "--no-display",
            "--json",
            "--save",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["counts"]["exit"] >= 1
    assert "Saved" in captured.err
    assert output.exists()


def test_an_unwritable_save_target_is_reported(sample_image, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--image",
                str(sample_image),
                "--no-model",
                "--no-display",
                "--save",
                str(tmp_path / "out.unsupported"),
            ]
        )
    assert "could not write" in str(excinfo.value)


# ------------------------------------------------------------------ video ---
@pytest.fixture
def sample_video(tmp_path):
    """A short synthetic clip with an exit sign drifting across the frame."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (640, 480)
    )
    if not writer.isOpened():
        pytest.skip("no MJPG video writer in this OpenCV build")
    try:
        for index in range(6):
            writer.write(draw_exit_sign(blank_frame(), center=(200 + index * 50, 140)))
    finally:
        writer.release()
    return path


def test_video_run_emits_one_json_object_per_analysed_frame(sample_video, capsys):
    assert (
        main(
            [
                "--video",
                str(sample_video),
                "--no-model",
                "--no-display",
                "--json",
                "--every",
                "2",
            ]
        )
        == 0
    )
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines, "the video loop produced no advice at all"
    for line in lines:
        payload = json.loads(line)
        assert payload["direction"] in {"LEFT", "RIGHT", "STRAIGHT", "FORWARD"}
    # 6 frames, analysed on frame 1 then every 2nd frame.
    assert len(lines) == 4


def test_video_run_writes_an_annotated_copy(sample_video, tmp_path):
    output = tmp_path / "out" / "annotated.avi"
    assert (
        main(
            [
                "--video",
                str(sample_video),
                "--no-model",
                "--no-display",
                "--save",
                str(output),
            ]
        )
        == 0
    )
    assert output.exists() and output.stat().st_size > 0
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        ok, frame = capture.read()
        assert ok and frame is not None
    finally:
        capture.release()


def test_an_unopenable_video_source_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["--video", str(tmp_path / "absent.mp4"), "--no-model", "--no-display"])
    assert "cannot open" in str(excinfo.value)


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
