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
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .config import get_settings
from .detection import estimate_light_quality
from .pipeline import PathFinder
from .visualize import render

__all__ = ["main", "build_parser"]


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
        type=int,
        default=3,
        metavar="N",
        help="analyse every Nth video frame (default: 3)",
    )
    parser.add_argument(
        "--iterations", type=int, default=50, help="benchmark iterations (default: 50)"
    )
    return parser


def _load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"cannot read image: {path}")
    return image


def _make_finder(args: argparse.Namespace) -> PathFinder:
    settings = get_settings()
    finder = PathFinder(
        settings=settings,
        model_path=Path(args.model) if args.model else None,
        use_model=not args.no_model,
    )
    if finder.model_error and not args.json:
        print(f"[model] {finder.model_error}", file=sys.stderr)
    return finder


def _show(window: str, frame: np.ndarray, no_display: bool) -> bool:
    """Display a frame. Returns False when the user asked to quit."""
    if no_display:
        return True
    try:
        cv2.imshow(window, frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")
    except cv2.error:
        # No GUI backend (headless server, WSL without X). Degrade instead of
        # crashing halfway through a run.
        print("[display] no GUI backend available; continuing headless", file=sys.stderr)
        return True


def run_image(args: argparse.Namespace) -> int:
    finder = _make_finder(args)
    image = _load_image(args.image)

    detections = finder.detect(image)
    advice = finder.analyze(image)
    lights = finder.fallback.detect_light_sources(image)

    if args.json:
        print(json.dumps(advice.as_dict(), indent=2))
    else:
        print(f"Image           : {args.image}")
        print(f"Resolution      : {image.shape[1]}x{image.shape[0]}")
        print(f"Light quality   : {advice.light_quality:.2f}")
        print(f"Torch advised   : {finder.should_enable_torch(image)}")
        print(f"Detections      : {advice.counts}")
        print(f"Light sources   : {len(lights)}")
        print(f"Direction       : {advice.direction}")
        print(f"Arrow angle     : {advice.arrow_angle_deg:.1f} deg")
        print(f"Urgency         : {advice.urgency}")
        print(f"Instruction     : {advice.instruction}")

    canvas = render(image, detections, advice, lights)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.save, canvas)
        print(f"Saved           : {args.save}")
    if not args.no_display:
        _show("Emergency Path Finder", canvas, args.no_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_stream(args: argparse.Namespace, source) -> int:
    finder = _make_finder(args)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video source: {source}")

    writer: Optional[cv2.VideoWriter] = None
    every = max(1, args.every)
    frame_index = 0
    detections: List = []
    advice = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1

            if frame_index % every == 0 or advice is None:
                detections = finder.detect(frame)
                advice = finder.analyze(frame)
                if args.json:
                    print(json.dumps(advice.as_dict()), flush=True)

            canvas = render(frame, detections, advice)

            if args.save:
                if writer is None:
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
                    writer = cv2.VideoWriter(
                        args.save,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (canvas.shape[1], canvas.shape[0]),
                    )
                writer.write(canvas)

            if not _show("Emergency Path Finder", canvas, args.no_display):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    finder = _make_finder(args)
    image = _load_image(args.benchmark)
    iterations = max(1, args.iterations)

    methods = {
        "colour signs": finder.fallback.detect_color_signs,
        "doors": finder.fallback.detect_doors,
        "stairs": finder.fallback.detect_stairs_edges,
        "light sources": finder.fallback.detect_light_sources,
        "vanishing point": finder.fallback.detect_vanishing_point,
        "relative depth": finder.fallback.estimate_relative_depth,
        "light quality": estimate_light_quality,
        "full pipeline": finder.analyze,
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
    args = build_parser().parse_args(argv)
    if args.image:
        return run_image(args)
    if args.video:
        return run_stream(args, args.video)
    if args.camera is not None:
        return run_stream(args, args.camera)
    if args.benchmark:
        return run_benchmark(args)
    build_parser().print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
