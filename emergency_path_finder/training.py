"""Shared YOLOv8 training plumbing.

``training/train_exit_detector.py`` and ``training/train_stairs_detector.py``
are thin CLIs over this module - previously the exit trainer carried all the
logic and the stairs trainer simply did not exist.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import yaml

from .config import Settings, get_settings

__all__ = ["TrainingConfig", "resolve_device", "write_data_yaml", "train", "export_tflite"]


@dataclass
class TrainingConfig:
    """Hyper-parameters for one training run."""

    dataset_dir: Path
    run_name: str
    class_names: Sequence[str]
    epochs: int = 50
    image_size: int = 416
    batch_size: int = 8
    patience: int = 20
    base_model: str = "yolov8n.pt"
    device: Optional[str] = None
    export: bool = True


def resolve_device(requested: Optional[str] = None) -> str:
    """Pick a training device, preferring CUDA when torch reports one."""
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_data_yaml(
    dataset_dir: Path, class_names: Sequence[str], force: bool = False
) -> Path:
    """Ensure a YOLO ``data.yaml`` exists for ``dataset_dir``.

    Roboflow exports ship one already; this only fills in for hand-assembled
    datasets. It is written with ``yaml.safe_dump`` rather than string
    formatting so Windows paths do not break the file.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"dataset directory not found: {dataset_dir}. "
            f"Run `python training/download_datasets.py` first."
        )

    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.exists() and not force:
        return yaml_path

    splits = {}
    for split, candidates in (
        ("train", ("train/images", "images/train")),
        ("val", ("valid/images", "val/images", "images/val")),
        ("test", ("test/images", "images/test")),
    ):
        for candidate in candidates:
            if (dataset_dir / candidate).exists():
                splits[split] = candidate
                break
    if "train" not in splits:
        raise FileNotFoundError(
            f"no training images under {dataset_dir}. Expected one of "
            f"'train/images' or 'images/train'."
        )
    splits.setdefault("val", splits["train"])

    payload = {
        "path": str(dataset_dir),
        **splits,
        "nc": len(class_names),
        "names": list(class_names),
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def export_tflite(
    weights: Path, image_size: int, settings: Optional[Settings] = None
) -> Optional[Path]:
    """Export to TFLite and copy it into the Flutter assets folder.

    Returns ``None`` when the export toolchain is unavailable - the PyTorch
    weights are still perfectly usable from Python, so this must not be fatal.
    """
    settings = settings or get_settings()
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    try:
        model = YOLO(str(weights))
        exported = Path(model.export(format="tflite", imgsz=image_size))
    except Exception:
        return None

    candidates: List[Path] = (
        [exported] if exported.suffix == ".tflite" else sorted(exported.glob("*.tflite"))
    )
    if not candidates:
        parent = exported if exported.is_dir() else exported.parent
        candidates = sorted(parent.glob("**/*.tflite"))
    if not candidates:
        return None

    settings.flutter_assets_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.flutter_assets_dir / "exit_detector.tflite"
    shutil.copy(candidates[0], destination)
    return destination


def train(config: TrainingConfig, settings: Optional[Settings] = None) -> Path:
    """Run training and return the path to the best checkpoint."""
    settings = settings or get_settings()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for training: "
            "pip install -r training/requirements.txt"
        ) from exc

    data_yaml = write_data_yaml(config.dataset_dir, config.class_names)
    device = resolve_device(config.device)

    model = YOLO(config.base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        patience=config.patience,
        device=device,
        project=str(settings.models_dir),
        name=config.run_name,
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        # Mixed precision only helps on CUDA and is unstable on CPU builds.
        amp=device != "cpu",
        hsv_h=0.015,
        hsv_s=0.7,
        # Exit signs must be findable in near-darkness, so train across a wide
        # brightness range.
        hsv_v=0.6,
        degrees=10,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        # Never flip vertically: an arrow on an exit sign pointing left is not
        # the same sign upside down.
        flipud=0.0,
        fliplr=0.5,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if config.export and best.exists():
        export_tflite(best, config.image_size, settings)
    return best
