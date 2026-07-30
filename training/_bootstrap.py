"""Make the project importable when the scripts are run in place.

``pip install -e .`` is the recommended setup, but these scripts are also run
straight out of a fresh clone, so put the repository root on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
