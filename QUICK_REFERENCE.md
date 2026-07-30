# Quick Reference

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # detection + navigation
pip install -r requirements-dev.txt  # + pytest
pip install -r training/requirements.txt   # + ultralytics/torch, for training
```

## Run detection

```bash
python -m emergency_path_finder --image corridor.jpg
python -m emergency_path_finder --video walkthrough.mp4 --every 5
python -m emergency_path_finder --camera
python -m emergency_path_finder --camera 1              # second webcam
python -m emergency_path_finder --benchmark corridor.jpg
```

Same thing via the wrapper script: `python training/run_detection.py --image ...`

| Flag | Effect |
|---|---|
| `--json` | One JSON object per analysed frame |
| `--save PATH` | Write the annotated image/video |
| `--no-display` | Never open a window (headless/CI) |
| `--no-model` | Classical CV only, ignore trained weights |
| `--model PATH` | Use specific weights |
| `--every N` | Analyse every Nth video frame (default 3) |

## Datasets

```bash
python training/download_datasets.py --list
python training/download_datasets.py
python training/download_datasets.py exit_signs_v2 --api-key rf_xxx
```

## Training

```bash
python training/train_exit_detector.py
python training/train_stairs_detector.py --dataset escalator_stairs
python training/train_exit_detector.py --epochs 100 --batch 16 --device cuda
python training/train_exit_detector.py --no-export     # skip TFLite
```

Output: `ml_models/<name>/weights/best.pt`, auto-detected on the next run.

## Tests

```bash
pytest                                  # deterministic, no data required
pytest tests/test_navigation.py -v
cd flutter_app && flutter test
```

## Mobile app

```bash
cd flutter_app
flutter create --platforms=android,ios --project-name emergency_path_finder .
flutter pub get
flutter run
flutter build apk --release
```

| Action | How |
|---|---|
| Toggle torch | Flash button, bottom left |
| Re-read the instruction | Red emergency button, bottom right |
| Follow guidance | Walk along the green arrow |
| Detection counts | Panel, bottom right |

### What the app does on launch

1. Asks for camera permission (motion is optional, location is never requested).
2. Opens the rear camera at medium resolution.
3. Loads `assets/models/exit_detector.tflite` if present, and says so if not.
4. Turns the torch on.
5. Streams frames through the detector and draws the arrow and instruction.
6. Releases the camera and torch when backgrounded.

## Python API

```python
from emergency_path_finder import PathFinder, FallbackDetector

finder = PathFinder()
advice = finder.analyze(frame, device_orientation_deg=0.0)
advice.direction        # LEFT | RIGHT | STRAIGHT | UPSTAIRS | DOWNSTAIRS | FORWARD
advice.urgency          # CRITICAL | HIGH | MEDIUM | LOW
advice.arrow_angle_deg  # 0 = ahead, 90 = hard right, 270 = hard left
advice.as_dict()

finder.should_enable_torch(frame)      # bool

detector = FallbackDetector()
detector.detect_color_signs(frame)     # -> List[Detection]
detector.detect_doors(frame)
detector.detect_stairs_edges(frame)
detector.detect_light_sources(frame)   # -> List[LightSource]
detector.detect_vanishing_point(frame) # -> (x, y) | None
detector.estimate_relative_depth(frame)# -> float32 map in [0, 1]
```

## Tuning

Set these in `.env` or the environment - no code edits needed.

| Variable | Default | Effect |
|---|---|---|
| `ROBOFLOW_API_KEY` | unset | Dataset downloads |
| `EPF_MODEL_PATH` | newest `ml_models/**/weights/best.pt` | Which weights to run |
| `EPF_DATASETS_DIR` | `datasets/` | Where datasets live |
| `EPF_MODELS_DIR` | `ml_models/` | Where checkpoints live |
| `EPF_INPUT_SIZE` | `416` | Lower (320) is faster, less accurate |
| `EPF_CONFIDENCE_THRESHOLD` | `0.35` | Raise to cut false positives |
| `EPF_NMS_IOU_THRESHOLD` | `0.45` | Raise to keep more overlapping boxes |
| `EPF_BRIGHTNESS_THRESHOLD` | `200` | What counts as a light source |
| `EPF_MIN_DOOR_AREA_RATIO` | `0.02` | Smallest door, as a fraction of frame |

On the app side, `MLDetector.confidenceThreshold` and
`NavigationService.turnDeadzone` are the two knobs worth touching.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no trained weights found` | Expected without training - classical CV takes over. Train, or set `EPF_MODEL_PATH`. |
| `ultralytics not installed` | `pip install -r training/requirements.txt` |
| `ROBOFLOW_API_KEY is not set` | Put a key in `.env`, or follow the printed manual steps |
| `cannot open video source: 0` | No webcam, or another app is holding it |
| `no GUI backend available` | Headless machine - add `--no-display` |
| Everything detected as a door | Raise `EPF_CONFIDENCE_THRESHOLD` or `EPF_MIN_DOOR_AREA_RATIO` |
| Slow on video | Raise `--every`, or lower `EPF_INPUT_SIZE` |
| App shows "model not loaded" | Copy a trained `.tflite` to `flutter_app/assets/models/` |
