"""Runtime configuration.

Every path and threshold that used to be hardcoded lives here and can be
overridden with an environment variable (see ``.env.example``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = ["Settings", "DetectorConfig", "PROJECT_ROOT", "get_settings"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        # "nan" parses happily as a float, and a NaN threshold silently disables
        # every comparison that uses it - every detection would pass the filter.
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    return value


def _check_range(name: str, value: float, low: float, high: float) -> None:
    """Reject NaN/inf and out-of-range tunables at construction time."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class DetectorConfig:
    """Tunables for the classical-CV fallback detectors.

    Defaults were chosen against 416x416 frames; sizes are expressed as a
    fraction of the frame area so they hold at any resolution.
    """

    min_sign_area_ratio: float = 0.0015
    min_door_area_ratio: float = 0.02
    door_min_aspect_ratio: float = 1.4
    door_max_aspect_ratio: float = 5.0
    min_light_area_ratio: float = 0.0005
    brightness_threshold: int = 200
    canny_low: int = 50
    canny_high: int = 150
    stairs_min_treads: int = 3
    stairs_max_tread_angle_deg: float = 25.0
    nms_iou_threshold: float = 0.45
    confidence_threshold: float = 0.35

    def __post_init__(self) -> None:
        """Fail loudly on a nonsensical override.

        Without this an ``EPF_CONFIDENCE_THRESHOLD=-1`` keeps every blob in the
        frame and ``canny_low > canny_high`` inverts hysteresis - both produce
        plausible-looking but wrong output rather than an error.
        """
        for name in (
            "min_sign_area_ratio",
            "min_door_area_ratio",
            "min_light_area_ratio",
        ):
            _check_range(name, getattr(self, name), 0.0, 1.0)
        _check_range("confidence_threshold", self.confidence_threshold, 0.0, 1.0)
        _check_range("nms_iou_threshold", self.nms_iou_threshold, 0.0, 1.0)
        _check_range("brightness_threshold", self.brightness_threshold, 0, 255)
        _check_range("canny_low", self.canny_low, 0, 255)
        _check_range("canny_high", self.canny_high, 0, 255)
        _check_range("door_min_aspect_ratio", self.door_min_aspect_ratio, 0.0, 100.0)
        _check_range("door_max_aspect_ratio", self.door_max_aspect_ratio, 0.0, 100.0)
        _check_range(
            "stairs_max_tread_angle_deg", self.stairs_max_tread_angle_deg, 0.0, 90.0
        )
        if self.canny_low >= self.canny_high:
            raise ValueError(
                f"canny_low must be below canny_high, got "
                f"{self.canny_low} >= {self.canny_high}"
            )
        if self.door_min_aspect_ratio >= self.door_max_aspect_ratio:
            raise ValueError(
                f"door_min_aspect_ratio must be below door_max_aspect_ratio, got "
                f"{self.door_min_aspect_ratio} >= {self.door_max_aspect_ratio}"
            )
        if self.stairs_min_treads < 2:
            raise ValueError(
                f"stairs_min_treads must be at least 2 - a single line is not a "
                f"staircase, got {self.stairs_min_treads}"
            )

    @classmethod
    def from_env(cls) -> "DetectorConfig":
        return cls(
            min_sign_area_ratio=_env_float("EPF_MIN_SIGN_AREA_RATIO", cls.min_sign_area_ratio),
            min_door_area_ratio=_env_float("EPF_MIN_DOOR_AREA_RATIO", cls.min_door_area_ratio),
            brightness_threshold=_env_int("EPF_BRIGHTNESS_THRESHOLD", cls.brightness_threshold),
            confidence_threshold=_env_float(
                "EPF_CONFIDENCE_THRESHOLD", cls.confidence_threshold
            ),
            nms_iou_threshold=_env_float("EPF_NMS_IOU_THRESHOLD", cls.nms_iou_threshold),
        )


@dataclass(frozen=True)
class Settings:
    """Project-wide paths and credentials."""

    project_root: Path = PROJECT_ROOT
    datasets_dir: Path = PROJECT_ROOT / "datasets"
    models_dir: Path = PROJECT_ROOT / "ml_models"
    flutter_assets_dir: Path = PROJECT_ROOT / "flutter_app" / "assets" / "models"
    model_path: Optional[Path] = None
    roboflow_api_key: Optional[str] = None
    input_size: int = 416
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def __post_init__(self) -> None:
        # YOLO downsamples by 32; anything smaller (or negative, from a typo'd
        # EPF_INPUT_SIZE) is silently rounded up by ultralytics or crashes the
        # export, so reject it here where the message can name the variable.
        if self.input_size < 32 or self.input_size % 32 != 0:
            raise ValueError(
                f"input_size must be a positive multiple of 32 (EPF_INPUT_SIZE), "
                f"got {self.input_size}"
            )

    @classmethod
    def from_env(cls) -> "Settings":
        root = _env_path("EPF_PROJECT_ROOT", PROJECT_ROOT)
        model_path = os.environ.get("EPF_MODEL_PATH")
        return cls(
            project_root=root,
            datasets_dir=_env_path("EPF_DATASETS_DIR", root / "datasets"),
            models_dir=_env_path("EPF_MODELS_DIR", root / "ml_models"),
            flutter_assets_dir=_env_path(
                "EPF_FLUTTER_ASSETS_DIR", root / "flutter_app" / "assets" / "models"
            ),
            model_path=Path(model_path).expanduser().resolve() if model_path else None,
            roboflow_api_key=os.environ.get("ROBOFLOW_API_KEY") or None,
            input_size=_env_int("EPF_INPUT_SIZE", cls.input_size),
            detector=DetectorConfig.from_env(),
        )

    def resolve_model_path(self) -> Optional[Path]:
        """Return the weights to run with, or ``None`` if nothing is trained yet.

        Order: explicit ``EPF_MODEL_PATH`` first, then the newest ``best.pt``
        under ``models_dir``.
        """
        if self.model_path is not None:
            return self.model_path if self.model_path.exists() else None
        if not self.models_dir.exists():
            return None

        def mtime(path: Path) -> float:
            # A training run writing into models_dir can remove a checkpoint
            # between the glob and the stat; treat that as "oldest" rather than
            # letting an OSError escape from what is only a lookup.
            try:
                return path.stat().st_mtime
            except OSError:
                return float("-inf")

        candidates = sorted(
            self.models_dir.glob("**/weights/best.pt"), key=mtime, reverse=True
        )
        return candidates[0] if candidates else None


def get_settings() -> Settings:
    """Build ``Settings`` from the current environment.

    Not cached on purpose - tests and notebooks flip env vars between calls.
    """
    return Settings.from_env()
