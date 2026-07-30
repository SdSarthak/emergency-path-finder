#!/usr/bin/env python
"""Fetch the Roboflow datasets used to train the detectors.

With ``ROBOFLOW_API_KEY`` set (or ``--api-key``) this downloads them. Without a
key it prints the manual steps and exits non-zero, so a scripted setup notices.

    python training/download_datasets.py --list
    python training/download_datasets.py --api-key rf_xxx
    python training/download_datasets.py exit_signs_v2
"""

import argparse
import sys

import _bootstrap  # noqa: F401  (puts the project root on sys.path)

from emergency_path_finder.config import get_settings
from emergency_path_finder.datasets import (
    DATASETS,
    download,
    is_downloaded,
    manual_instructions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        default=[],
        help=f"datasets to download (default: all). One of: {', '.join(sorted(DATASETS))}",
    )
    parser.add_argument("--api-key", help="Roboflow API key (or set ROBOFLOW_API_KEY)")
    parser.add_argument("--list", action="store_true", help="list datasets and exit")
    parser.add_argument(
        "--overwrite", action="store_true", help="re-download even if present"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()
    unknown = [name for name in args.names if name not in DATASETS]
    if unknown:
        print(
            f"unknown dataset(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(DATASETS))}",
            file=sys.stderr,
        )
        return 2
    keys = args.names or sorted(DATASETS)

    if args.list:
        for key in sorted(DATASETS):
            spec = DATASETS[key]
            state = "present" if is_downloaded(spec, settings) else "missing"
            print(f"{key:<20} {state:<8} ~{spec.approx_images:>6} images  {spec.url}")
        return 0

    print(f"Datasets directory: {settings.datasets_dir}")
    failures = 0
    for key in keys:
        spec = DATASETS[key]
        print(f"\n=== {spec.name} ===")
        try:
            path = download(
                key,
                api_key=args.api_key,
                settings=settings,
                overwrite=args.overwrite,
            )
            print(f"ready: {path}")
        except Exception as exc:
            failures += 1
            print(f"could not download automatically: {exc}", file=sys.stderr)
            print(manual_instructions(spec, spec.target_dir(settings)), file=sys.stderr)

    if failures:
        print(
            f"\n{failures} of {len(keys)} dataset(s) still need attention.",
            file=sys.stderr,
        )
        return 1

    print("\nAll datasets ready. Next: python training/train_exit_detector.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
