# Emergency Path Finder - Quick Reference

## 🚀 Installation & Running

### Windows/Mac/Linux
```bash
# 1. Install dependencies
cd training
pip install -r requirements.txt

# 2. Download datasets (free, from Roboflow)
python download_datasets.py

# 3. Train model (optional - takes hours)
python train_exit_detector.py

# 4. Test detection
python test_detection.py --camera

# 5. Build mobile app
cd ../flutter_app
flutter pub get
flutter run
```

## 📱 Mobile App Controls

| Action | How |
|--------|-----|
| **Start Detection** | App launches automatically |
| **Toggle Flash** | Tap flash icon (bottom left) |
| **Emergency Signal** | Tap red emergency button (bottom right) |
| **View Detections** | Info displayed in top-left corner |
| **Follow Arrow** | Walk in direction of green arrow |
| **Exit App** | Android: Back button / iOS: Swipe right |

## 🎯 How It Works

### When You Open the App:
1. ✅ Camera turns on automatically
2. ✅ Flash activates automatically
3. ✅ ML model starts detecting objects
4. ✅ Green arrow appears pointing to exit/door/stairs
5. ✅ Status text shows: "EXIT FOUND" or "STAIRS DETECTED" etc
6. ✅ Follow arrow to reach safety

### If Exit Sign Visible:
- Arrow points directly to it
- Shows distance in meters
- Green border = high confidence

### If No Exit Sign:
- App detects stairs or doors
- Shows arrow to them
- Or uses vanishing point (corridor perspective)
- Fallback: widest opening

## 🔧 Customization

### Change Detection Confidence Threshold
File: `flutter_app/lib/services/ml_detector.dart`
```dart
static const double CONFIDENCE_THRESHOLD = 0.5;  // Change this (0.0-1.0)
```

### Adjust Arrow Size/Color
File: `flutter_app/lib/widgets/arrow_overlay.dart`
```dart
paint..color = Colors.greenAccent  // Change color
arrowLength = 150.0                // Change arrow size
```

### Increase FPS (mobile performance)
File: `flutter_app/lib/services/ml_detector.py`
```python
INPUT_SIZE = 320  # Reduce from 416 for faster inference
```

## 📊 Performance Optimization

### On Slow Phones:
```
1. Use YOLOv8-nano model (6 MB) - already configured
2. Reduce resolution: 320×320 instead of 416×416
3. Process every 2nd frame (5 FPS instead of 10)
4. Disable GPU inference if available
```

### Battery Saver Mode:
```
1. Reduce brightness
2. Process every 3rd frame (3-5 FPS)
3. Use edge detection instead of ML (fallback mode)
4. Disable GPS if enabled
```

## 🐛 Troubleshooting

### App Crashes on Start
```bash
# Check logcat
flutter logs

# Ensure permissions granted
# Try on physical device instead of emulator
# Update Flutter: flutter upgrade
```

### Model Not Loading
```bash
# Check file exists
flutter app/assets/models/exit_detector.tflite

# Verify in pubspec.yaml
assets:
  - assets/models/

# Clean build
flutter clean
flutter pub get
flutter run
```

### Low FPS
```
- Reduce input size (320×320)
- Use TFLite instead of PyTorch
- Close other apps
- Check CPU usage: flutter analyze
```

### Camera Permission Denied
```
Android: Settings > Apps > Permissions > Camera
iOS: Settings > [App Name] > Camera
```

## 📈 Dataset Sources

All **FREE** public datasets:

| Dataset | Images | Use |
|---------|--------|-----|
| [Emergency Exit Signs v2](https://universe.roboflow.com/emergency-exit-signs/emergency-exit-signs-v2) | 1,070 | Exit detection |
| [Stairs Detection](https://universe.roboflow.com/stairs-detection/stairs-fo4v5) | 7,890 | Stair detection |
| [Escalator-Stairs](https://universe.roboflow.com/escalatorstairsdetection/escalator-stairs) | 8,690 | Elevator/escalator |
| [Exit-Detection](https://universe.roboflow.com/project1exits/exit-detection-w00yi) | 36 | Doors + obstacles |

## 🔍 Testing Commands

```bash
# Test on image
python test_detection.py --image path/to/image.jpg

# Test on video
python test_detection.py --video path/to/video.mp4

# Test on webcam (real-time)
python test_detection.py --camera

# Benchmark speed
python test_detection.py --benchmark path/to/image.jpg
```

## 📦 Project Files Overview

```
flutter_app/lib/
├── main.dart                      # App entry point
├── screens/
│   └── emergency_detector_screen.dart  # Main camera screen
├── services/
│   ├── ml_detector.dart           # ML inference
│   └── navigation_service.dart    # Arrow calculation
├── models/
│   └── detection_result.dart      # Data classes
└── widgets/
    └── arrow_overlay.dart         # Arrow UI

training/
├── download_datasets.py           # Get data from Roboflow
├── train_exit_detector.py         # Train YOLOv8-nano
├── fallback_detection.py          # Edge detection methods
├── test_detection.py              # Test on images/video
└── requirements.txt               # Dependencies
```

## 🎓 Learning Resources

### Computer Vision
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [TFLite Guide](https://www.tensorflow.org/lite/guide)

### Flutter
- [Flutter Camera Plugin](https://pub.dev/packages/camera)
- [TFLite Flutter](https://pub.dev/packages/tflite_flutter)
- [Flutter Sensors](https://pub.dev/packages/sensors_plus)

### Emergency Detection
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Roboflow Datasets](https://roboflow.com/)

## 📝 File Size Reference

```
Model Sizes:
- YOLOv8-nano PyTorch: 6.2 MB
- YOLOv8-nano TFLite: 3.2 MB (quantized)
- YOLOv8-small: 22 MB
- YOLOv8-medium: 49 MB

App Size:
- Flutter base: 15-20 MB
- With TFLite model: 18-25 MB (APK)
- Release optimized: 12-18 MB

Dataset Sizes (raw):
- Exit Signs v2: 150 MB
- Stairs Detection: 800 MB
- Escalator-Stairs: 900 MB
```

## ⚡ Performance Targets

```
Target FPS: 10 fps (mobile CPU)
Inference Time: 100-150 ms
Memory: < 300 MB
Battery: ~20% per hour continuous use
Model Size: < 20 MB
App Size: < 30 MB (final release)
```

## 🔒 Privacy & Security

✅ **Privacy:**
- All processing is on-device
- No data sent to servers
- No internet required
- No location tracking (optional GPS)

✅ **Security:**
- App doesn't store detection logs
- No personal data collection
- Model cannot identify people (only detects objects)
- Open-source for audit

## 📞 Support

**If app crashes:**
1. Check `/flutter_app/` for `main.dart` entry point
2. Run `flutter doctor` to check setup
3. Check device logs: `flutter logs`

**If detection doesn't work:**
1. Check camera is enabled (torch should light up)
2. Test with `python test_detection.py --camera`
3. Verify model file exists: `assets/models/exit_detector.tflite`

**For questions:**
- Read SETUP.md
- Check ARCHITECTURE.md for system design
- Review training/fallback_detection.py for alternative methods

---

**Project Status:** ✅ Complete & Ready for Use  
**Last Updated:** January 2026  
**Version:** 1.0.0
