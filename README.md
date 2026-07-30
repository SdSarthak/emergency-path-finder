# Emergency Path Finder

A real-time mobile application that helps users find emergency exits using computer vision and ML, with arrow-based navigation guidance. Works offline on normal smartphones.

## Project Structure

```
├── flutter_app/          # Mobile app (iOS/Android)
├── ml_models/            # Trained ML models (TFLite format)
├── datasets/             # Raw and processed datasets
├── training/             # ML training scripts and notebooks
└── docs/                 # Documentation
```

## Features

- ✅ Real-time exit sign detection
- ✅ Stairs and door detection
- ✅ Automatic flashlight activation
- ✅ Arrow-based navigation overlay
- ✅ Offline operation (no internet needed)
- ✅ Works on normal phones (low-end compatible)
- ✅ Fallback pathfinding without signs

## Quick Start

### 1. Setup Training Environment
```bash
cd training
pip install -r requirements.txt
python download_datasets.py
```

### 2. Train Models
```bash
python train_exit_detector.py
python train_stairs_detector.py
```

### 3. Build Mobile App
```bash
cd flutter_app
flutter pub get
flutter run
```

## Hardware Requirements

- **Phone**: Any Android 8+ or iOS 11+ device
- **Laptop**: 8GB RAM, any GPU (CPU works but slower)
- **Storage**: 2GB for datasets, 100MB for app

## ML Models Used

- YOLOv8-nano (exit signs, doors, stairs)
- TensorFlow Lite (on-device inference)
- Classical CV (vanishing point, depth estimation)

## Dataset Sources

- Emergency Exit Signs v2 (1,070 images)
- Stairs Detection (7,890 images)
- Escalator-Stairs (8,690 images)
- Exit-Detection (with doors/obstacles)

All datasets are free and publicly available on Roboflow.
