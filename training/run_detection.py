#!/usr/bin/env python
"""Run detection on an image, video or webcam.

A convenience wrapper - identical to ``python -m emergency_path_finder``.

    python training/run_detection.py --image sample.jpg
    python training/run_detection.py --camera
    python training/run_detection.py --benchmark sample.jpg
"""

import _bootstrap  # noqa: F401  (puts the project root on sys.path)

from emergency_path_finder.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
