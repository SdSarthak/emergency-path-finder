"""Command line entry point: ``python -m emergency_path_finder``.

Runs the full pipeline over an image, a video file or a live webcam and prints
(or draws) the navigation advice.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import get_settings
from .detection import estimate_light_quality
from .pipeline import PathFinder
from .visualize import render

__all__ = ["main", "build_parser"]


def _positive_int(raw: str) -> int:
    """argparse type that rejects 0 and negatives instead of clamping them."""
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emergency-path-finder",
        description="Detect emergency exits and print navigation guidance.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="path to a still image")
    source.add_argument("--video", help="path to a video file")
    source.add_argument(
        "--camera",
        nargs="?",
        const=0,
        type=int,
        metavar="INDEX",
        help="use a webcam (default index 0)",
    )
    source.add_argument("--benchmark", help="time each detector on an image")

    parser.add_argument("--model", help="path to trained YOLO weights (.pt)")
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="classical CV only, even if weights are available",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="never open a window (required on headless machines)",
    )
    parser.add_argument("--save", help="write the annotated frame/video to this path")
    parser.add_argument(
        "--json", action="store_true", help="print advice as JSON, one object per frame"
    )
    parser.add_argument(
        "--every",
        type=_positive_int,
        default=3,
        metavar="N",
        help="analyse every Nth video frame (default: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=50,
        help="benchmark iterations (default: 50)",
    )
    return parser


def _load_image(path: str) -> np.ndarray:
    source = Path(path)
    if not source.exists():
        raise SystemExit(f"no such file: {path}")
    if source.is_dir():
        raise SystemExit(f"expected an image file, got a directory: {path}")
    # imread also returns None for a file it cannot decode, which is a different
    # problem from the file not being there.
    image = cv2.imread(str(source))
    if image is None:
        raise SystemExit(f"cannot decode image (unsupported or corrupt): {path}")
    return image


def _make_finder(args: argparse.Namespace) -> PathFinder:
    if args.model:
        weights = Path(args.model)
        if not weights.exists():
            # An explicit --model that quietly degrades to classical CV gives
            # the user fallback results they will read as model output.
            raise SystemExit(f"model weights not found: {weights}")
        if args.no_model:
            raise SystemExit("--model and --no-model contradict each other")
    settings = get_settings()
    finder = PathFinder(
        settings=settings,
        model_path=Path(args.model) if args.model else None,
        use_model=not args.no_model,
    )
    if finder.model_error:
        # stderr always, so `--json` stdout stays machine-readable.
        print(f"[model] {finder.model_error}", file=sys.stderr)
    return finder


class _Display:
    """Frame display that gives up permanently the first time it fails.

    OpenCV builds without a GUI backend raise on every ``imshow``; retrying once
    per frame printed one warning per frame and paid the exception cost on the
    whole video.
    """

    def __init__(self, window: str, enabled: bool) -> None:
        self.window = window
        self.enabled = enabled

    def show(self, frame: np.ndarray) -> bool:
        """Draw a frame. Returns False when the user asked to quit."""
        if not self.enabled:
            return True
        try:
            cv2.imshow(self.window, frame)
            return (cv2.waitKey(1) & 0xFF) != ord("q")
        except cv2.error:
            self.enabled = False
            print(
                "[display] no GUI backend available; continuing headless",
                file=sys.stderr,
            )
            return True

    def wait_for_key(self) -> None:
        if not self.enabled:
            return
        try:
            cv2.waitKey(0)
        except cv2.error:
            self.enabled = False

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            self.enabled = False


def _write_image(path: str, canvas: np.ndarray) -> None:
    destination = Path(path)
    if destination.parent != Path(""):
        destination.parent.mkdir(parents=True, exist_ok=True)
    # imwrite returns False (it does not raise) for an unknown extension or an
    # unwritable directory, which used to look like a successful save.
    if not cv2.imwrite(str(destination), canvas):
        raise SystemExit(
            f"could not write {destination} - check the extension is one OpenCV "
            f"supports (.png, .jpg) and that the directory is writable"
        )


def run_image(args: argparse.Namespace) -> int:
    finder = _make_finder(args)
    image = _load_image(args.image)

    analysis = finder.analyze_frame(image)
    advice = analysis.advice

    if args.json:
        print(json.dumps(advice.as_dict(), indent=2))
    else:
        print(f"Image           : {args.image}")
        print(f"Resolution      : {image.shape[1]}x{image.shape[0]}")
        print(f"Light quality   : {advice.light_quality:.2f}")
        print(f"Torch advised   : {analysis.torch_advised}")
        print(f"Detections      : {advice.counts}")
        print(f"Light sources   : {len(analysis.lights)}")
        print(f"Direction       : {advice.direction}")
        print(f"Arrow angle     : {advice.arrow_angle_deg:.1f} deg")
        print(f"Urgency         : {advice.urgency}")
        print(f"Instruction     : {advice.instruction}")

    display = _Display("Emergency Path Finder", enabled=not args.no_display)
    if args.save or display.enabled:
        canvas = render(image, analysis.detections, advice, analysis.lights)
        if args.save:
            _write_image(args.save, canvas)
            # stderr under --json so stdout stays a single parseable document.
            print(
                f"Saved           : {args.save}",
                file=sys.stderr if args.json else sys.stdout,
            )
        if display.enabled:
            display.show(canvas)
            display.wait_for_key()
            display.close()
    return 0


def _open_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    destination = Path(path)
    if destination.parent != Path(""):
        destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        # An unopened writer swallows every frame and leaves a 0-byte file.
        raise SystemExit(
            f"could not open a video writer for {destination} - the mp4v codec "
            f"may be unavailable in this OpenCV build"
        )
    return writer


def run_stream(args: argparse.Namespace, source) -> int:
    finder = _make_finder(args)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video source: {source}")

    writer: Optional[cv2.VideoWriter] = None
    writer_size: Tuple[int, int] = (0, 0)
    every = max(1, args.every)
    frame_index = 0
    detections: List = []
    lights: List = []
    advice = None
    display = _Display("Emergency Path Finder", enabled=not args.no_display)
    draw = args.save is not None or display.enabled

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                break
            frame_index += 1

            if frame_index % every == 0 or advice is None:
                analysis = finder.analyze_frame(frame)
                detections, lights, advice = (
                    analysis.detections,
                    analysis.lights,
                    analysis.advice,
                )
                if args.json:
                    print(json.dumps(advice.as_dict()), flush=True)

            if not draw:
                continue
            canvas = render(frame, detections, advice, lights)

            if args.save:
                if writer is None:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    # A webcam commonly reports 0, and a broken container NaN.
                    if not fps or not np.isfinite(fps) or fps <= 0:
                        fps = 20.0
                    writer_size = (canvas.shape[1], canvas.shape[0])
                    writer = _open_writer(args.save, float(fps), writer_size)
                if (canvas.shape[1], canvas.shape[0]) != writer_size:
                    # VideoWriter silently drops any frame whose size differs
                    # from the one it was opened with, so a mid-stream
                    # resolution change would truncate the output.
                    canvas = cv2.resize(canvas, writer_size)
                writer.write(canvas)

            if not display.show(canvas):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        display.close()
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    finder = _make_finder(args)
    image = _load_image(args.benchmark)
    iterations = args.iterations

    methods = {
        "colour signs": finder.fallback.detect_color_signs,
        "doors": finder.fallback.detect_doors,
        "stairs": finder.fallback.detect_stairs_edges,
        "light sources": finder.fallback.detect_light_sources,
        "vanishing point": finder.fallback.detect_vanishing_point,
        "relative depth": finder.fallback.estimate_relative_depth,
        "light quality": estimate_light_quality,
        "full pipeline": finder.analyze_frame,
    }

    print(f"Benchmark: {args.benchmark} ({image.shape[1]}x{image.shape[0]}), "
          f"{iterations} iterations")
    print("-" * 52)
    for name, func in methods.items():
        start = time.perf_counter()
        for _ in range(iterations):
            func(image)
        elapsed_ms = (time.perf_counter() - start) / iterations * 1000.0
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else float("inf")
        print(f"{name:<18} {elapsed_ms:8.2f} ms {fps:8.1f} fps")
    print("-" * 52)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # `is not None`, not truthiness: `--camera 0` is a valid source and an empty
    # `--image ""` should fail as an unreadable path, not silently print help.
    if args.image is not None:
        return run_image(args)
    if args.video is not None:
        return run_stream(args, args.video)
    if args.camera is not None:
        return run_stream(args, args.camera)
    if args.benchmark is not None:
        return run_benchmark(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
