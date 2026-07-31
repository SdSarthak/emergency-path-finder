"""Shared YOLOv8 training plumbing.

``training/train_exit_detector.py`` and ``training/train_stairs_detector.py``
are thin CLIs over this module - previously the exit trainer carried all the
logic and the stairs trainer simply did not exist.
"""

from __future__ import annotations

import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml

from .config import Settings, get_settings

__all__ = [
    "TrainingConfig",
    "resolve_device",
    "write_data_yaml",
    "make_val_split",
    "check_class_names",
    "train",
    "export_tflite",
]

#: Extensions ultralytics will read as training images.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


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
    #: Passed to ultralytics so a rerun reproduces the same weights.
    seed: int = 0
    #: Fraction of the training images carved off when the dataset ships no
    #: validation split of its own.
    val_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.image_size < 32 or self.image_size % 32 != 0:
            raise ValueError(
                f"image_size must be a positive multiple of 32, got {self.image_size}"
            )
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(
                f"val_fraction must be in (0, 1), got {self.val_fraction}"
            )
        if not self.class_names:
            raise ValueError("class_names must not be empty")


def resolve_device(requested: Optional[str] = None) -> str:
    """Pick a training device, preferring CUDA when torch reports one."""
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _list_images(directory: Path) -> List[Path]:
    """Image files directly under ``directory``, in a stable order."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _labels_dir_for(images_dir: Path) -> Path:
    """The YOLO label folder that pairs with an images folder.

    ``a/train/images`` -> ``a/train/labels`` and ``a/images/train`` ->
    ``a/labels/train``, which are the two layouts ultralytics accepts.
    """
    if images_dir.name == "images":
        return images_dir.with_name("labels")
    return images_dir.parent.parent / "labels" / images_dir.name


def make_val_split(
    dataset_dir: Path,
    train_rel: str,
    fraction: float = 0.2,
    seed: int = 0,
    min_images: int = 5,
) -> Optional[str]:
    """Carve a held-out validation split out of the training images.

    Returns the new split's path relative to ``dataset_dir``, or ``None`` when
    there are too few images to split meaningfully.

    Validating on the training images - which is what this function exists to
    avoid - does not merely inflate mAP: ultralytics selects ``best.pt`` and
    triggers early stopping on the validation score, so the leak changes which
    weights are shipped, not just the number printed next to them. The
    selection is seeded so a rerun reproduces the same holdout.
    """
    dataset_dir = Path(dataset_dir)
    train_images = dataset_dir / train_rel
    images = _list_images(train_images)
    if len(images) < min_images:
        return None

    holdout = max(1, int(round(len(images) * fraction)))
    if len(images) - holdout < 1:
        return None

    val_rel = "images/val" if train_images.name == "train" else "valid/images"
    val_images = dataset_dir / val_rel
    if _list_images(val_images):
        return val_rel

    train_labels = _labels_dir_for(train_images)
    val_labels = _labels_dir_for(val_images)
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    # Sample from a sorted list with an explicit Random instance: neither the
    # filesystem order nor the global random seed can change the result.
    chosen = random.Random(seed).sample(images, holdout)
    for image in sorted(chosen):
        shutil.move(str(image), str(val_images / image.name))
        label = train_labels / f"{image.stem}.txt"
        if label.exists():
            shutil.move(str(label), str(val_labels / label.name))
    return val_rel


def check_class_names(yaml_path: Path, class_names: Sequence[str]) -> Optional[str]:
    """Compare a dataset's own class list against the one we train with.

    Returns a description of the mismatch, or ``None`` when they agree. A
    mismatch is how a stairs dataset labelled ``(stairs, escalator)`` gets
    trained as ``(exit, stairs, door)`` and then reports exits that are
    staircases.
    """
    try:
        payload = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return f"could not read {yaml_path}: {exc}"
    if not isinstance(payload, dict):
        return f"{yaml_path} does not contain a YAML mapping"

    names = payload.get("names")
    if isinstance(names, dict):
        try:
            names = [names[key] for key in sorted(names, key=int)]
        except (KeyError, TypeError, ValueError):
            names = list(names.values())
    if not isinstance(names, (list, tuple)):
        return f"{yaml_path} has no usable 'names' list"

    actual = [str(name) for name in names]
    expected = [str(name) for name in class_names]
    if actual == expected:
        return None
    return (
        f"class names in {yaml_path} are {actual}, but training was asked for "
        f"{expected}. The trained model will emit {actual}; pass matching "
        f"--class-names or edit data.yaml."
    )


def _discover_splits(dataset_dir: Path) -> dict:
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
    return splits


def write_data_yaml(
    dataset_dir: Path,
    class_names: Sequence[str],
    force: bool = False,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> Path:
    """Ensure a YOLO ``data.yaml`` exists for ``dataset_dir``.

    Roboflow exports ship one already; this only fills in for hand-assembled
    datasets. It is written with ``yaml.safe_dump`` rather than string
    formatting so Windows paths do not break the file.

    When the dataset has no validation split, a deterministic holdout is carved
    out of the training images rather than pointing ``val`` at ``train`` - see
    ``make_val_split``.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"dataset directory not found: {dataset_dir}. "
            f"Run `python training/download_datasets.py` first."
        )

    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.exists() and not force:
        mismatch = check_class_names(yaml_path, class_names)
        if mismatch:
            warnings.warn(mismatch, RuntimeWarning, stacklevel=2)
        return yaml_path

    splits = _discover_splits(dataset_dir)
    if "train" not in splits:
        raise FileNotFoundError(
            f"no training images under {dataset_dir}. Expected one of "
            f"'train/images' or 'images/train'."
        )

    if "val" not in splits:
        val_rel = make_val_split(
            dataset_dir, splits["train"], fraction=val_fraction, seed=seed
        )
        if val_rel is not None:
            splits["val"] = val_rel
        else:
            warnings.warn(
                f"{dataset_dir} has no validation split and too few training "
                f"images to carve one out; validating on the training images. "
                f"Every reported metric is optimistic and best.pt is selected "
                f"on data the model has seen.",
                RuntimeWarning,
                stacklevel=2,
            )
            splits["val"] = splits["train"]

    # Keep the canonical train/val/test key order regardless of discovery order.
    splits = {key: splits[key] for key in ("train", "val", "test") if key in splits}

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

    data_yaml = write_data_yaml(
        config.dataset_dir,
        config.class_names,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
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
        # Without an explicit seed the augmentation stream and the weight init
        # differ between runs, so two runs of the same command are not
        # comparable and a regression cannot be reproduced.
        seed=config.seed,
        deterministic=True,
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
