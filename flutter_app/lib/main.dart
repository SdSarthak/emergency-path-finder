import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import 'screens/emergency_detector_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // availableCameras() throws on emulators and devices without a camera; the
  // app should still start and say so rather than crash on launch.
  List<CameraDescription> cameras = const [];
  try {
    cameras = await availableCameras();
  } catch (error) {
    debugPrint('Could not enumerate cameras: $error');
  }

  runApp(EmergencyPathFinderApp(cameras: cameras));
}

class EmergencyPathFinderApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const EmergencyPathFinderApp({required this.cameras, super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Emergency Path Finder',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: Colors.red,
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      home: PermissionGate(cameras: cameras),
    );
  }
}

/// Asks for the permissions the app needs before handing over to the detector.
///
/// Only the camera is mandatory. Motion and location are nice-to-have, and
/// blocking the whole app on them - as this screen used to - left users stuck
/// on a spinner on any device that declines them.
class PermissionGate extends StatefulWidget {
  final List<CameraDescription> cameras;

  const PermissionGate({required this.cameras, super.key});

  @override
  State<PermissionGate> createState() => _PermissionGateState();
}

class _PermissionGateState extends State<PermissionGate> {
  bool _requesting = true;
  String _message = 'Requesting camera permission...';

  @override
  void initState() {
    super.initState();
    _requestPermissions();
  }

  Future<void> _requestPermissions() async {
    setState(() {
      _requesting = true;
      _message = 'Requesting camera permission...';
    });

    final camera = await Permission.camera.request();

    // Motion is a nice-to-have: it keeps the arrow stable as the handset turns,
    // but the app is usable without it. Location is never requested - nothing
    // here needs it, and the app is fully offline.
    await Permission.sensors.request();

    if (!mounted) return;

    if (camera.isGranted) {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => EmergencyDetectorScreen(cameras: widget.cameras),
        ),
      );
      return;
    }

    setState(() {
      _requesting = false;
      _message = camera.isPermanentlyDenied
          ? 'Camera access is blocked. Enable it in system settings to detect exits.'
          : 'Camera access is required to detect emergency exits.';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Emergency Path Finder')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_requesting) ...[
                const CircularProgressIndicator(),
                const SizedBox(height: 24),
              ],
              Text(_message, textAlign: TextAlign.center),
              const SizedBox(height: 32),
              if (!_requesting) ...[
                ElevatedButton(
                  onPressed: _requestPermissions,
                  child: const Text('Try again'),
                ),
                const SizedBox(height: 12),
                TextButton(
                  onPressed: openAppSettings,
                  child: const Text('Open settings'),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
