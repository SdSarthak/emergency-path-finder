# Emergency Path Finder

Find the way out of an unfamiliar building when the lights are gone.

The camera looks for emergency exit signage, doorways and staircases; the app
turns whatever it can see into a single arrow and a single sentence. Everything
runs on the handset, offline, on hardware that was cheap five years ago.

Two halves live in this repository:

| | |
|---|---|
| `emergency_path_finder/` | The reference implementation in Python. Detection, navigation logic, training and a CLI you can point at a photo, a video or a webcam. |
| `flutter_app/` | The Android/iOS app: camera stream, TFLite inference, torch control and the arrow overlay. |

The navigation rules are implemented once in Python, tested there against
synthetic scenes, and mirrored in Dart - so the thresholds that decide "turn
left" can be tuned on a laptop instead of on a phone in a stairwell.

---

## Why classical CV and not just a model

A YOLO model trained on well-lit stock photographs of exit signs is very good at
finding well-lit exit signs. That is not the situation the app exists for.

So the pipeline always runs both:

- **Model detections** (YOLOv8-nano, when you have trained one) for signage,
  doors and stairs.
- **Classical fallbacks** that need no weights at all:
  - green/red HSV segmentation for illuminated signage,
  - CLAHE-boosted edge detection for door-shaped openings,
  - a run of parallel tread edges for staircases,
  - a voting-based corridor vanishing point for when nothing is detected,
  - bright-blob detection so emergency lighting can be steered towards.

If no model has been trained, the whole thing still works - with lower
precision, and it says so.

---

## Quick start

```bash
git clone https://github.com/SdSarthak/emergency-path-finder.git
cd emergency-path-finder

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Analyse a photo of a corridor - no model or dataset required.
python -m emergency_path_finder --image path/to/corridor.jpg

# Live from a webcam.
python -m emergency_path_finder --camera
```

Typical output:

```
Image           : corridor.jpg
Resolution      : 1280x720
Light quality   : 0.31
Torch advised   : False
Detections      : {'exit': 1, 'stairs': 0, 'door': 2}
Light sources   : 3
Direction       : RIGHT
Arrow angle     : 74.2 deg
Urgency         : CRITICAL
Instruction     : EXIT FOUND - 6 m, go right
```

Add `--json` for machine-readable output, `--save out.png` for an annotated
copy, `--benchmark photo.jpg` to time each stage.

---

## Using it as a library

```python
import cv2
from emergency_path_finder import PathFinder

finder = PathFinder()                       # picks up trained weights if present
advice = finder.analyze(cv2.imread("corridor.jpg"))

print(advice.direction)        # 'RIGHT'
print(advice.instruction)      # 'EXIT FOUND - 6 m, go right'
print(advice.urgency)          # 'CRITICAL'
print(advice.arrow_angle_deg)  # 74.2  (0 = ahead, 90 = hard right)
print(advice.as_dict())        # JSON-friendly view

if finder.should_enable_torch(frame):
    ...                                     # the scene is too dark to read
```

Individual detectors are available on their own:

```python
from emergency_path_finder import FallbackDetector

detector = FallbackDetector()
signs = detector.detect_color_signs(frame)
stairs = detector.detect_stairs_edges(frame)
point = detector.detect_vanishing_point(frame)   # (x, y) or None
```

---

## Getting the data

The datasets are free on Roboflow Universe but far too large to commit here.

```bash
cp .env.example .env          # then put your key in ROBOFLOW_API_KEY
python training/download_datasets.py --list
python training/download_datasets.py           # all of them
python training/download_datasets.py exit_signs_v2
```

Without an API key the script prints the manual download steps for each dataset
and exits non-zero. Extract each one so that
`datasets/<name>/data.yaml` exists.

| Key | Dataset | Images |
|---|---|---|
| `exit_signs_v2` | Emergency Exit Signs v2 | ~1,070 |
| `stairs_detection` | Stairs Detection | ~7,890 |
| `escalator_stairs` | Escalator-Stairs | ~8,690 |
| `exit_detection` | Exit-Detection (doors, obstacles) | ~36 |

---

## Training

```bash
pip install -r requirements.txt -r training/requirements.txt

python training/train_exit_detector.py                 # exit / stairs / door
python training/train_stairs_detector.py               # stairs / escalator
python training/train_exit_detector.py --epochs 100 --batch 16 --device cuda
```

Checkpoints land in `ml_models/<run-name>/weights/best.pt` and are picked up
automatically on the next inference run. Roughly 20-30 minutes on a modern GPU,
2-4 hours on CPU.

If TensorFlow is installed, the trainer also exports
`flutter_app/assets/models/exit_detector.tflite` for the mobile app. That step
is optional - pass `--no-export` to skip it; the PyTorch weights work fine from
Python either way.

Nothing under `ml_models/`, `datasets/` or `flutter_app/assets/models/` is
committed. Weights and datasets are reproduced with the commands above.

---

## The mobile app

```bash
cd flutter_app

# This repository carries lib/, test/, the manifest and the plist. Generate the
# rest of the platform scaffolding on first checkout:
flutter create --platforms=android,ios --project-name emergency_path_finder .

flutter pub get
flutter test
flutter run
```

`flutter create` will not overwrite the checked-in `lib/`,
`android/app/src/main/AndroidManifest.xml` or `ios/Runner/Info.plist`; it only
fills in the Gradle, Xcode and resource files that are not worth versioning.

Drop a trained `exit_detector.tflite` into `flutter_app/assets/models/` before
building. Without it the app still launches: the camera, the torch and the
manual guidance work, and the status bar says the model is missing.

**Requirements:** Android 8+ or iOS 12+, a rear camera. A torch is used when
present.

---

## Configuration

Everything is environment-driven; see `.env.example` for the full list.

| Variable | Purpose |
|---|---|
| `ROBOFLOW_API_KEY` | Dataset downloads |
| `EPF_MODEL_PATH` | Explicit weights to run with |
| `EPF_DATASETS_DIR` / `EPF_MODELS_DIR` | Move data off the repo drive |
| `EPF_CONFIDENCE_THRESHOLD` | Detection cut-off (default 0.35) |
| `EPF_BRIGHTNESS_THRESHOLD` | What counts as a light source (default 200) |

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest                      # Python: detection, navigation, pipeline, CLI
cd flutter_app && flutter test    # Dart: navigation service, NMS, geometry
```

Every Python test draws its own synthetic scene with NumPy and OpenCV, so the
suite is deterministic and needs no dataset download and no trained model.

---

## Documentation

- [`docs/SETUP.md`](docs/SETUP.md) - detailed setup, troubleshooting
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - system design and data flow
- [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) - command cheat sheet

---

## Limitations

Worth being explicit about, given the subject matter:

- Distances are inferred from apparent size against an assumed 2 m object.
  Treat them as ordering, not measurement.
- The vanishing point assumes a straight corridor with visible perspective
  lines. It returns `None` rather than guessing when it cannot find one.
- `estimate_relative_depth` is two monocular cues combined, not real depth.
- **This is a navigation aid, not a life-safety system.** It is not certified,
  it has not been validated against building codes, and it should never replace
  marked evacuation routes or a fire warden's instructions.

## License

MIT
