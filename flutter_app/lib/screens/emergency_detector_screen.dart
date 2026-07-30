import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:torch_light/torch_light.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/ml_detector.dart';
import '../services/navigation_service.dart';
import '../widgets/arrow_overlay.dart';
import '../models/detection_result.dart';

class EmergencyDetectorScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const EmergencyDetectorScreen({required this.cameras, Key? key})
      : super(key: key);

  @override
  State<EmergencyDetectorScreen> createState() =>
      _EmergencyDetectorScreenState();
}

class _EmergencyDetectorScreenState extends State<EmergencyDetectorScreen> {
  late CameraController _cameraController;
  late MLDetector _mlDetector;
  late NavigationService _navigationService;
  
  DetectionResult? _detectionResult;
  bool _torchOn = false;
  bool _processing = false;
  double _deviceOrientation = 0.0;
  String _statusText = 'Initializing...';
  bool _canNavigate = false;

  @override
  void initState() {
    super.initState();
    _initializeApp();
  }

  Future<void> _initializeApp() async {
    try {
      // Initialize camera
      _cameraController = CameraController(
        widget.cameras[0],
        ResolutionPreset.high,
        enableAudio: false,
      );

      await _cameraController.initialize();

      // Initialize ML detector
      _mlDetector = MLDetector();
      await _mlDetector.initialize();

      // Initialize navigation service
      _navigationService = NavigationService();

      // Listen to device orientation
      userAccelerometerEvents.listen((UserAccelerometerEvent event) {
        final x = event.x;
        final y = event.y;
        _deviceOrientation = math.atan2(y, x) * 180 / math.pi;
      });

      // Start camera processing
      _startCameraStream();

      // Auto enable torch
      _enableTorch();

      setState(() => _statusText = 'Ready');
      _canNavigate = true;
    } catch (e) {
      setState(() => _statusText = 'Error: $e');
    }
  }

  void _startCameraStream() {
    _cameraController.startImageStream((image) async {
      if (_processing) return;
      _processing = true;

      try {
        final result = await _mlDetector.detectFrame(image);
        
        setState(() {
          _detectionResult = result;
          _updateStatusText(result);
        });

        // Calculate arrow direction
        if (_canNavigate && result.hasDetections) {
          final arrowAngle =
              _navigationService.calculateArrowAngle(result, _deviceOrientation);
          setState(() => _detectionResult?.arrowAngle = arrowAngle);
        }
      } catch (e) {
        debugPrint('Detection error: $e');
      } finally {
        _processing = false;
      }
    });
  }

  void _updateStatusText(DetectionResult result) {
    if (result.exits.isNotEmpty) {
      _statusText = 'Exit found! Distance: ${result.exits.first.confidence}m';
    } else if (result.stairs.isNotEmpty) {
      _statusText = 'Stairs detected (${result.stairs.first.direction})';
    } else if (result.doors.isNotEmpty) {
      _statusText = 'Door detected ahead';
    } else {
      _statusText = 'Searching for exits...';
    }
  }

  Future<void> _enableTorch() async {
    try {
      await TorchLight.enableTorch();
      setState(() => _torchOn = true);
    } catch (e) {
      debugPrint('Torch error: $e');
    }
  }

  Future<void> _disableTorch() async {
    try {
      await TorchLight.disableTorch();
      setState(() => _torchOn = false);
    } catch (e) {
      debugPrint('Torch error: $e');
    }
  }

  @override
  void dispose() {
    _cameraController.dispose();
    _mlDetector.dispose();
    if (_torchOn) _disableTorch();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 20),
              Text(_statusText),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera feed
          CameraPreview(_cameraController),

          // Arrow overlay
          if (_detectionResult != null && _detectionResult!.hasDetections)
            ArrowOverlay(
              angle: _detectionResult!.arrowAngle ?? 0.0,
              detectionResult: _detectionResult!,
            ),

          // Status overlay
          Positioned(
            top: 20,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _statusText,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),

          // Control buttons
          Positioned(
            bottom: 30,
            left: 20,
            right: 20,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                FloatingActionButton(
                  heroTag: 'torch',
                  backgroundColor: _torchOn ? Colors.orange : Colors.grey,
                  onPressed: _torchOn ? _disableTorch : _enableTorch,
                  child: Icon(_torchOn ? Icons.flash_on : Icons.flash_off),
                ),
                FloatingActionButton(
                  heroTag: 'emergency',
                  backgroundColor: Colors.red,
                  child: const Icon(Icons.emergency, size: 32),
                  onPressed: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Emergency activated!')),
                    );
                  },
                ),
              ],
            ),
          ),

          // Detection info
          if (_detectionResult != null)
            Positioned(
              bottom: 150,
              right: 20,
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.black87,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (_detectionResult!.exits.isNotEmpty)
                      Text(
                        'Exits: ${_detectionResult!.exits.length}',
                        style: const TextStyle(color: Colors.greenAccent),
                      ),
                    if (_detectionResult!.stairs.isNotEmpty)
                      Text(
                        'Stairs: ${_detectionResult!.stairs.length}',
                        style: const TextStyle(color: Colors.yellowAccent),
                      ),
                    if (_detectionResult!.doors.isNotEmpty)
                      Text(
                        'Doors: ${_detectionResult!.doors.length}',
                        style: const TextStyle(color: Colors.blueAccent),
                      ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
