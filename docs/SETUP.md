# Setup Guide

## Prerequisites

| | |
|---|---|
| Python | 3.9 or newer |
| Flutter | 3.10+ (only for the mobile app) |
| Android Studio / Xcode | only for building the app |
| GPU | optional - training works on CPU, just slower |

---

## 1. Python side (5 minutes)

```bash
git clone https://github.com/SdSarthak/emergency-path-finder.git
cd emergency-path-finder

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

That is all that is needed for detection and navigation - three packages, no
torch, no TensorFlow. Verify it:

```bash
pip install -r requirements-dev.txt
pytest
```

The suite draws its own test scenes, so it passes on a fresh clone with no
dataset and no trained model.

Then point it at a real photo:

```bash
python -m emergency_path_finder --image path/to/corridor.jpg
```

You will see `[model] no trained weights found - running on classical CV only`
on stderr. That is expected until you train something.

---

## 2. Configuration

```bash
cp .env.example .env
```

Only `ROBOFLOW_API_KEY` is needed, and only for dataset downloads. Everything
else in the file is an optional override with a sensible default. The variables
are read from the process environment - if you want `.env` loaded automatically,
`pip install python-dotenv` and call `load_dotenv()` in your own entry point.

`.env` is gitignored. Do not commit it.

---

## 3. Datasets

All four are free on Roboflow Universe. Sign up at <https://roboflow.com> and
copy your key from <https://app.roboflow.com/settings/api>.

```bash
python training/download_datasets.py --list      # what is registered, what is present
python training/download_datasets.py             # download everything
python training/download_datasets.py exit_signs_v2 --overwrite
```

**Manual route.** If you would rather not use an API key, the script prints
per-dataset instructions. For each one: open the Universe page, export in
**YOLOv8** format, download the zip, and extract it so that
`datasets/<key>/data.yaml` exists.

| Key | Images | Disk |
|---|---|---|
| `exit_signs_v2` | ~1,070 | ~120 MB |
| `stairs_detection` | ~7,890 | ~900 MB |
| `escalator_stairs` | ~8,690 | ~1 GB |
| `exit_detection` | ~36 | ~5 MB |

Budget about 2 GB. Nothing under `datasets/` is committed.

To keep the data off your repo drive:

```bash
export EPF_DATASETS_DIR=/mnt/big-disk/epf-datasets
```

---

## 4. Training

```bash
pip install -r training/requirements.txt
```

This pulls torch, torchvision and ultralytics - a few GB. Then:

```bash
python training/train_exit_detector.py       # classes: exit, stairs, door
python training/train_stairs_detector.py     # classes: stairs, escalator
```

Useful flags: `--epochs`, `--batch` (lower it on OOM), `--imgsz`, `--device`,
`--name`, `--no-export`.

| Hardware | 50 epochs on `exit_signs_v2` |
|---|---|
| NVIDIA GPU | 20-30 min |
| CPU | 2-4 hours |

Results land in `ml_models/<name>/`:

```
ml_models/exit_detector/
├── weights/
│   ├── best.pt          <- picked up automatically by PathFinder
│   └── last.pt
├── results.png
└── confusion_matrix.png
```

### TFLite export

If TensorFlow is installed the trainer also writes
`flutter_app/assets/models/exit_detector.tflite`. It is not in
`training/requirements.txt` because it is a large download that does not have
wheels for every Python version:

```bash
pip install "tensorflow>=2.13,<2.17"
```

Export failure is never fatal - the `.pt` weights still work from Python. Pass
`--no-export` to skip the attempt entirely.

---

## 5. Mobile app

This repository versions the parts of the Flutter project worth reviewing -
`lib/`, `test/`, `pubspec.yaml`, the Android manifest and the iOS plist - and
not the generated Gradle/Xcode scaffolding. Generate that on first checkout:

```bash
cd flutter_app
flutter create --platforms=android,ios --project-name emergency_path_finder .
flutter pub get
```

`flutter create` leaves existing files alone, so your `lib/`, manifest and plist
survive.

```bash
flutter test        # unit tests, no device needed
flutter devices
flutter run
flutter build apk --release
flutter build ios --release
```

Copy a trained model in before building:

```bash
cp ml_models/exit_detector/weights/best.tflite \
   flutter_app/assets/models/exit_detector.tflite
```

Without it the app still launches, shows the camera and torch, and reports that
the model is missing.

---

## Project structure

```
emergency-path-finder/
├── emergency_path_finder/     # the reference implementation
│   ├── config.py              # env-driven paths and thresholds
│   ├── geometry.py            # BoundingBox, Detection, IoU, NMS
│   ├── detection.py           # classical CV detectors
│   ├── yolo_detector.py       # optional YOLOv8 wrapper
│   ├── navigation.py          # direction, urgency, arrow angle
│   ├── pipeline.py            # PathFinder - frame in, advice out
│   ├── datasets.py            # Roboflow registry and downloads
│   ├── training.py            # shared YOLOv8 training plumbing
│   ├── visualize.py           # debug drawing
│   └── cli.py                 # python -m emergency_path_finder
├── training/                  # thin CLIs over the package
│   ├── download_datasets.py
│   ├── train_exit_detector.py
│   ├── train_stairs_detector.py
│   └── run_detection.py
├── tests/                     # pytest, synthetic scenes only
├── flutter_app/
│   ├── lib/                   # app source
│   └── test/                  # dart unit tests
├── datasets/                  # gitignored
├── ml_models/                 # gitignored
└── docs/
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'emergency_path_finder'`**
Run from the repository root, or `pip install -e .`. The scripts in `training/`
add the root to `sys.path` themselves.

**`cannot open video source: 0`**
No webcam attached, or another application is holding it. Try `--camera 1`.

**`no GUI backend available`**
OpenCV cannot open a window (headless server, WSL without an X server). Add
`--no-display`; combine with `--save out.png` to inspect the result.

**Training killed / CUDA out of memory**
Lower `--batch` (8 → 4 → 2), then `--imgsz 320`.

**`ROBOFLOW_API_KEY is not set`**
Expected without a key. Follow the printed manual download steps.

**Torch does not turn on**
The device has no flash, or another app holds the camera. Everything else keeps
working; the button just stays grey.

**Detections are noisy**
Raise `EPF_CONFIDENCE_THRESHOLD` (default 0.35). For doors specifically, raise
`EPF_MIN_DOOR_AREA_RATIO`.

**Slow inference on the phone**
Lower the export size (`--imgsz 320`) and retrain, or raise
`MLDetector.confidenceThreshold` so fewer boxes reach NMS.
