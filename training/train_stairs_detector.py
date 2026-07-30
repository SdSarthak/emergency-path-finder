#!/usr/bin/env python
"""Train the stairs / escalator detector.

Stairs are the part of an evacuation route that hurts you if you get it wrong,
so they get their own model trained on a dedicated dataset rather than sharing
capacity with exit signage.

    python training/train_stairs_detector.py
    python training/train_stairs_detector.py --dataset escalator_stairs --epochs 80
"""

import argparse
import sys

import _bootstrap  # noqa: F401  (puts the project root on sys.path)

from emergency_path_finder.config import get_settings
from emergency_path_finder.datasets import DATASETS, is_downloaded, manual_instructions
from emergency_path_finder.training import TrainingConfig, resolve_device, train

CLASS_NAMES = ("stairs", "escalator")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        default="stairs_detection",
        choices=["stairs_detection", "escalator_stairs"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8, help="lower this if you hit OOM")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", help="cuda, cpu or a device index (default: auto)")
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--name", default="stairs_detector", help="run name under ml_models/")
    parser.add_argument(
        "--no-export", action="store_true", help="skip the TFLite export step"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()
    spec = DATASETS[args.dataset]
    dataset_dir = spec.target_dir(settings)

    if not is_downloaded(spec, settings):
        print(f"Dataset not found at {dataset_dir}\n", file=sys.stderr)
        print(manual_instructions(spec, dataset_dir), file=sys.stderr)
        print(
            f"\nOr run: python training/download_datasets.py {args.dataset}",
            file=sys.stderr,
        )
        return 1

    device = resolve_device(args.device)
    print("Emergency Path Finder - stairs detector training")
    print(f"  dataset : {dataset_dir}")
    print(f"  device  : {device}")
    print(f"  epochs  : {args.epochs}  imgsz={args.imgsz}  batch={args.batch}")

    try:
        best = train(
            TrainingConfig(
                dataset_dir=dataset_dir,
                run_name=args.name,
                class_names=CLASS_NAMES,
                epochs=args.epochs,
                image_size=args.imgsz,
                batch_size=args.batch,
                patience=args.patience,
                base_model=args.base_model,
                device=device,
                export=not args.no_export,
            ),
            settings=settings,
        )
    except Exception as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nBest weights: {best}")
    print("Point inference at it with EPF_MODEL_PATH or --model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
