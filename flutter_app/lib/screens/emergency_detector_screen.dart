import 'dart:async';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:torch_light/torch_light.dart';

import '../models/detection_result.dart';
import '../services/ml_detector.dart';
import '../services/navigation_service.dart';
import '../widgets/arrow_overlay.dart';

class EmergencyDetectorScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const EmergencyDetectorScreen({required this.cameras, super.key});

  @override
  State<EmergencyDetectorScreen> createState() =>
      _EmergencyDetectorScreenState();
}

class _EmergencyDetectorScreenState extends State<EmergencyDetectorScreen>
    with WidgetsBindingObserver {
  CameraController? _cameraController;
  final MLDetector _mlDetector = MLDetector();
  final NavigationService _navigationService = NavigationService();

  StreamSubscription<AccelerometerEvent>? _accelerometerSubscription;

  DetectionResult? _detectionResult;
  DetectionResult? _previousResult;
  bool _torchOn = false;
  bool _processing = false;
  bool _ready = false;
  double _deviceOrientation = 0.0;
  String _statusText = 'Initializing...';
  String _urgency = Urgency.low;
  String? _modelWarning;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  Future<void> _initialize() async {
    if (widget.cameras.isEmpty) {
      setState(() => _statusText = 'No camera available on this device');
      return;
    }

    try {
      // Medium resolution keeps the detection loop usable on low-end phones,
      // which is the whole point of the project.
      final controller = CameraController(
        widget.cameras.first,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      _cameraController = controller;

      // A missing model is not fatal: the preview, torch and manual guidance
      // still work, so report it instead of bailing out.
      try {
        await _mlDetector.initialize();
      } catch (error) {
        _modelWarning = 'Detection model not loaded - see README to train it';
        debugPrint('Model load failed: $error');
      }

      _accelerometerSubscription = accelerometerEvents.listen((event) {
        // Roll of the handset around the viewing axis, used to keep the arrow
        // pointing at the world rather than at the screen.
        _deviceOrientation = math.atan2(event.x, event.y) * 180 / math.pi;
      });

      await _startStream();
      await _enableTorch();

      if (!mounted) return;
      setState(() {
        _ready = true;
        _statusText = _modelWarning ?? 'Ready - point the camera down a corridor';
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _statusText = 'Initialization failed: $error');
    }
  }

  Future<void> _startStream() async {
    final controller = _cameraController;
    if (controller == null || controller.value.isStreamingImages) return;

    await controller.startImageStream((image) async {
      if (_processing || !mounted) return;
      _processing = true;
      try {
        final result = await _mlDetector.detectFrame(image);
        result.arrowAngle =
            _navigationService.calculateArrowAngle(result, _deviceOrientation);
        if (!mounted) return;
        setState(() {
          _previousResult = _detectionResult;
          _detectionResult = result;
          _statusText = _navigationService.getNavigationInstruction(result);
          // Torch state is a decent proxy for how dark the scene is.
          _urgency =
              _navigationService.calculateUrgency(result, _torchOn ? 0.3 : 0.7);
        });
      } catch (error) {
        debugPrint('Detection error: $error');
      } finally {
        _processing = false;
      }
    });
  }

  Future<void> _enableTorch() async {
    try {
      await TorchLight.enableTorch();
      if (!mounted) return;
      setState(() => _torchOn = true);
    } catch (error) {
      debugPrint('Torch unavailable: $error');
    }
  }

  Future<void> _disableTorch() async {
    try {
      await TorchLight.disableTorch();
    } catch (error) {
      debugPrint('Torch unavailable: $error');
    }
    if (!mounted) return;
    setState(() => _torchOn = false);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Release the camera and the torch when backgrounded; leaving the torch on
    // would flatten a battery the user may badly need.
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _releaseHardware();
    } else if (state == AppLifecycleState.resumed && _cameraController == null) {
      _initialize();
    }
  }

  void _releaseHardware() {
    if (_torchOn) {
      TorchLight.disableTorch().catchError((_) {});
      _torchOn = false;
    }
    final controller = _cameraController;
    _cameraController = null;
    controller?.dispose();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _accelerometerSubscription?.cancel();
    _releaseHardware();
    _mlDetector.dispose();
    super.dispose();
  }

  Color _urgencyColor() {
    switch (_urgency) {
      case Urgency.critical:
        return Colors.green.shade700;
      case Urgency.high:
        return Colors.orange.shade800;
      case Urgency.medium:
        return Colors.blueGrey.shade700;
      default:
        return Colors.black87;
    }
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 20),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Text(_statusText, textAlign: TextAlign.center),
              ),
            ],
          ),
        ),
      );
    }

    final result = _detectionResult;

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(controller),

          if (result != null && result.hasDetections)
            ArrowOverlay(
              angle: result.arrowAngle ?? 0.0,
              detectionResult: result,
            ),

          Positioned(
            top: 20,
            left: 16,
            right: 16,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: _urgencyColor(),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    _statusText,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  if (result != null && !result.hasDetections)
                    Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Text(
                        _navigationService.analyzeCorridorLayout(result),
                        style: const TextStyle(
                          color: Colors.white70,
                          fontSize: 13,
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),

          if (result != null && result.hasDetections)
            Positioned(
              bottom: 150,
              right: 16,
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
                    if (result.exits.isNotEmpty)
                      Text(
                        'Exits: ${result.exits.length}',
                        style: const TextStyle(color: Colors.greenAccent),
                      ),
                    if (result.stairs.isNotEmpty)
                      Text(
                        'Stairs: ${result.stairs.length}',
                        style: const TextStyle(color: Colors.yellowAccent),
                      ),
                    if (result.doors.isNotEmpty)
                      Text(
                        'Doors: ${result.doors.length}',
                        style: const TextStyle(color: Colors.blueAccent),
                      ),
                    if (_previousResult != null)
                      Text(
                        _navigationService.isMovingCorrectly(
                          result,
                          _previousResult!,
                        )
                            ? 'Getting closer'
                            : 'Target shrinking - turn back?',
                        style: const TextStyle(color: Colors.white70),
                      ),
                  ],
                ),
              ),
            ),

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
                  backgroundColor: _ready ? Colors.red : Colors.grey,
                  onPressed: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text(
                          _detectionResult?.hasDetections ?? false
                              ? _navigationService
                                  .getNavigationInstruction(_detectionResult!)
                              : 'No exit in view - sweep the camera slowly',
                        ),
                      ),
                    );
                  },
                  child: const Icon(Icons.emergency, size: 32),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
