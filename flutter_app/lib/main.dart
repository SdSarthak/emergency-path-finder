import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:torch_light/torch_light.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:permission_handler/permission_handler.dart';
import 'screens/emergency_detector_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(EmergencyPathFinderApp(cameras: cameras));
}

class EmergencyPathFinderApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const EmergencyPathFinderApp({required this.cameras, Key? key})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Emergency Path Finder',
      theme: ThemeData(
        primarySwatch: Colors.red,
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      home: PermissionHandler(cameras: cameras),
    );
  }
}

class PermissionHandler extends StatefulWidget {
  final List<CameraDescription> cameras;

  const PermissionHandler({required this.cameras, Key? key}) : super(key: key);

  @override
  State<PermissionHandler> createState() => _PermissionHandlerState();
}

class _PermissionHandlerState extends State<PermissionHandler> {
  @override
  void initState() {
    super.initState();
    _requestPermissions();
  }

  Future<void> _requestPermissions() async {
    final status = await Future.wait([
      Permission.camera.request(),
      Permission.sensors.request(),
      Permission.location.request(),
    ]);

    if (mounted && status.every((s) => s.isGranted)) {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (context) =>
              EmergencyDetectorScreen(cameras: widget.cameras),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Emergency Path Finder')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 20),
            const Text('Requesting permissions...'),
            const SizedBox(height: 40),
            ElevatedButton(
              onPressed: _requestPermissions,
              child: const Text('Grant Permissions'),
            ),
          ],
        ),
      ),
    );
  }
}
