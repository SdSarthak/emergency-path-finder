import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import '../models/detection_result.dart';

class MLDetector {
  late Interpreter _interpreter;
  static const String _modelPath = 'assets/models/exit_detector.tflite';
  static const int INPUT_SIZE = 416;
  static const double CONFIDENCE_THRESHOLD = 0.5;

  Future<void> initialize() async {
    try {
      final buffer = await rootBundle.load(_modelPath);
      _interpreter = Interpreter.fromBuffer(buffer);
      print('Model loaded successfully');
    } catch (e) {
      print('Error loading model: $e');
      rethrow;
    }
  }

  Future<DetectionResult> detectFrame(CameraImage image) async {
    try {
      // Convert camera image to processable format
      final inputImage = _preprocessImage(image);

      // Run inference
      final output = List.filled(
        1 * 13 * 13 * 25,
        0.0,
        growable: false,
      ).reshape([1, 13, 13, 25]);

      _interpreter.run(inputImage, output);

      // Parse results
      final result = _parseOutput(output, image.width, image.height);
      return result;
    } catch (e) {
      print('Detection error: $e');
      return DetectionResult(
        exits: [],
        stairs: [],
        doors: [],
        frameWidth: image.width,
        frameHeight: image.height,
      );
    }
  }

  List<List<List<List<double>>>> _preprocessImage(CameraImage image) {
    // Convert NV21 to RGB
    final rgbImage = _convertNV21ToRGB(image);

    // Resize to 416x416
    final resized = img.copyResize(rgbImage,
        width: INPUT_SIZE, height: INPUT_SIZE);

    // Normalize and prepare input
    final input = List.filled(1 * INPUT_SIZE * INPUT_SIZE * 3, 0.0);
    for (int i = 0; i < resized.length; i++) {
      final pixel = resized[i];
      final r = img.getRed(pixel) / 255.0;
      final g = img.getGreen(pixel) / 255.0;
      final b = img.getBlue(pixel) / 255.0;

      input[i * 3] = r;
      input[i * 3 + 1] = g;
      input[i * 3 + 2] = b;
    }

    return [
      [
        [input]
      ]
    ];
  }

  img.Image _convertNV21ToRGB(CameraImage image) {
    final rgbImage = img.Image(width: image.width, height: image.height);
    final uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

    for (int x = 0; x < image.width; x++) {
      for (int y = 0; y < image.height; y++) {
        final uvIndex =
            uvPixelStride * (x / 2).floor() + (y / 2).floor() * image.planes[1].bytesPerRow;
        final index = y * image.width + x;

        final yp = image.planes[0].bytes[index];
        final u = image.planes[1].bytes[uvIndex];
        final v = image.planes[2].bytes[uvIndex];

        rgbImage[index] = img.getColor(
          _yuv2r(yp, u, v),
          _yuv2g(yp, u, v),
          _yuv2b(yp, u, v),
        );
      }
    }
    return rgbImage;
  }

  int _yuv2r(int y, int u, int v) {
    final r = y + v * 1436 / 1024 - 179;
    return _clip(r.toInt());
  }

  int _yuv2g(int y, int u, int v) {
    final g = y - u * 46549 / 131072 + 44 - v * 93604 / 131072 + 91;
    return _clip(g.toInt());
  }

  int _yuv2b(int y, int u, int v) {
    final b = y + u * 1814 / 1024 - 227;
    return _clip(b.toInt());
  }

  int _clip(int value) {
    return value.clamp(0, 255);
  }

  DetectionResult _parseOutput(
    List<List<List<List<double>>>> output,
    int imageWidth,
    int imageHeight,
  ) {
    final exits = <Detection>[];
    final stairs = <Detection>[];
    final doors = <Detection>[];

    // Parse YOLO output (13x13 grid with 25 values per cell)
    // Format: [bx, by, bw, bh, objectness, class0, class1, ...]

    for (int y = 0; y < 13; y++) {
      for (int x = 0; x < 13; x++) {
        final predictions = output[0][y][x];

        final bx = predictions[0];
        final by = predictions[1];
        final bw = predictions[2];
        final bh = predictions[3];
        final objectness = predictions[4];

        if (objectness < CONFIDENCE_THRESHOLD) continue;

        // Parse class probabilities
        final exitProb = predictions[5];
        final stairProb = predictions[6];
        final doorProb = predictions[7];

        final cellWidth = imageWidth / 13.0;
        final cellHeight = imageHeight / 13.0;

        final centerX = (x + bx) * cellWidth;
        final centerY = (y + by) * cellHeight;
        final width = bw * imageWidth;
        final height = bh * imageHeight;

        final box = Rect(
          x: centerX - width / 2,
          y: centerY - height / 2,
          width: width,
          height: height,
        );

        if (exitProb > CONFIDENCE_THRESHOLD) {
          exits.add(Detection(
            label: 'exit',
            confidence: exitProb,
            box: box,
          ));
        }

        if (stairProb > CONFIDENCE_THRESHOLD) {
          stairs.add(Detection(
            label: 'stairs',
            confidence: stairProb,
            box: box,
            direction: _detectStairDirection(box, imageWidth),
          ));
        }

        if (doorProb > CONFIDENCE_THRESHOLD) {
          doors.add(Detection(
            label: 'door',
            confidence: doorProb,
            box: box,
          ));
        }
      }
    }

    return DetectionResult(
      exits: exits,
      stairs: stairs,
      doors: doors,
      frameWidth: imageWidth,
      frameHeight: imageHeight,
    );
  }

  String _detectStairDirection(Rect box, int imageWidth) {
    final centerX = box.centerX;
    final midPoint = imageWidth / 2;

    if (box.centerY < imageWidth * 0.3) {
      return 'up';
    } else if (box.centerY > imageWidth * 0.7) {
      return 'down';
    } else if (centerX < midPoint - imageWidth * 0.2) {
      return 'left';
    } else if (centerX > midPoint + imageWidth * 0.2) {
      return 'right';
    }
    return 'ahead';
  }

  void dispose() {
    _interpreter.close();
  }
}
