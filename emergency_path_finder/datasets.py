"""Dataset registry and download helpers.

The datasets are hosted on Roboflow Universe and are free, but they are far too
large to live in this repository. With ``ROBOFLOW_API_KEY`` set this module
fetches them; without it, it prints the manual steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import Settings, get_settings

__all__ = ["DatasetSpec", "DATASETS", "manual_instructions", "download", "download_all"]


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to fetch one Roboflow dataset."""

    key: str
    name: str
    workspace: str
    project: str
    version: int
    approx_images: int

    @property
    def url(self) -> str:
        return f"https://universe.roboflow.com/{self.workspace}/{self.project}"

    def target_dir(self, settings: Optional[Settings] = None) -> Path:
        settings = settings or get_settings()
        return settings.datasets_dir / self.key


DATASETS: Dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in (
        DatasetSpec(
            key="exit_signs_v2",
            name="Emergency Exit Signs v2",
            workspace="emergency-exit-signs",
            project="emergency-exit-signs-v2",
            version=1,
            approx_images=1070,
        ),
        DatasetSpec(
            key="stairs_detection",
            name="Stairs Detection",
            workspace="stairs-detection",
            project="stairs-fo4v5",
            version=1,
            approx_images=7890,
        ),
        DatasetSpec(
            key="escalator_stairs",
            name="Escalator-Stairs",
            workspace="escalatorstairsdetection",
            project="escalator-stairs",
            version=1,
            approx_images=8690,
        ),
        DatasetSpec(
            key="exit_detection",
            name="Exit-Detection (doors, obstacles, exits)",
            workspace="project1exits",
            project="exit-detection-w00yi",
            version=1,
            approx_images=36,
        ),
    )
}


def manual_instructions(spec: DatasetSpec, target: Path) -> str:
    """Copy-pasteable steps for downloading a dataset by hand."""
    return "\n".join(
        [
            f"{spec.name}  (~{spec.approx_images} images)",
            f"  1. Open {spec.url}",
            "  2. Click 'Download this Dataset' and choose the 'YOLOv8' format",
            "  3. Download the zip",
            f"  4. Extract it so that {target / 'data.yaml'} exists",
        ]
    )


def is_downloaded(spec: DatasetSpec, settings: Optional[Settings] = None) -> bool:
    """A dataset counts as present once its YOLO ``data.yaml`` is in place."""
    return (spec.target_dir(settings) / "data.yaml").exists()


def download(
    key: str,
    api_key: Optional[str] = None,
    settings: Optional[Settings] = None,
    overwrite: bool = False,
) -> Path:
    """Download one dataset in YOLOv8 layout and return its directory.

    Raises ``RuntimeError`` with the manual steps when no API key is available
    or the ``roboflow`` package is missing, so an unattended run fails loudly
    instead of training on an empty folder.
    """
    settings = settings or get_settings()
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(DATASETS)}")

    spec = DATASETS[key]
    target = spec.target_dir(settings)

    if is_downloaded(spec, settings) and not overwrite:
        return target

    api_key = api_key or settings.roboflow_api_key
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Get a free key at "
            "https://app.roboflow.com/settings/api, or download manually:\n"
            + manual_instructions(spec, target)
        )

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError(
            "the 'roboflow' package is not installed "
            "(pip install -r training/requirements.txt), or download manually:\n"
            + manual_instructions(spec, target)
        ) from exc

    target.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(spec.workspace).project(spec.project)
    project.version(spec.version).download("yolov8", location=str(target), overwrite=overwrite)
    return target


def download_all(
    api_key: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, str]:
    """Attempt every dataset; report per-dataset outcome instead of aborting."""
    results: Dict[str, str] = {}
    for key in DATASETS:
        try:
            path = download(key, api_key=api_key, settings=settings)
            results[key] = f"ok: {path}"
        except Exception as exc:
            results[key] = f"skipped: {exc}"
    return results


def missing_datasets(settings: Optional[Settings] = None) -> List[DatasetSpec]:
    """Datasets that are registered but not present on disk."""
    return [spec for spec in DATASETS.values() if not is_downloaded(spec, settings)]
