"""Runtime configuration.

Every path and threshold that used to be hardcoded lives here and can be
overridden with an environment variable (see ``.env.example``).
"""

from __future__ import annotations

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
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


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
        candidates = sorted(
            self.models_dir.glob("**/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None


def get_settings() -> Settings:
    """Build ``Settings`` from the current environment.

    Not cached on purpose - tests and notebooks flip env vars between calls.
    """
    return Settings.from_env()
