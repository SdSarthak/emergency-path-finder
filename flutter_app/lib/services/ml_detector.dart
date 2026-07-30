import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/detection_result.dart';

/// Runs the exported YOLOv8 model on camera frames.
///
/// The interpreter is queried for its own tensor shapes rather than assuming a
/// fixed grid: the export size is a training-time choice, and hardcoding it is
/// how this silently produced garbage boxes before.
class MLDetector {
  static const String modelAsset = 'assets/models/exit_detector.tflite';
  static const List<String> classNames = ['exit', 'stairs', 'door'];
  static const double confidenceThreshold = 0.4;
  static const double nmsIouThreshold = 0.45;

  Interpreter? _interpreter;

  int _inputWidth = 416;
  int _inputHeight = 416;

  /// Number of predictions per frame (YOLOv8 anchor-free grid total).
  int _numPredictions = 0;

  /// Values per prediction: 4 box coordinates plus one score per class.
  int _numAttributes = 0;

  /// True when the output is laid out as [1, attributes, predictions] rather
  /// than [1, predictions, attributes]. Both appear in the wild depending on
  /// the ultralytics version used to export.
  bool _attributesFirst = true;

  bool get isReady => _interpreter != null;
  int get inputWidth => _inputWidth;
  int get inputHeight => _inputHeight;

  Future<void> initialize() async {
    final options = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset(modelAsset, options: options);
    _readTensorShapes();
  }

  void _readTensorShapes() {
    final interpreter = _interpreter!;
    final inputShape = interpreter.getInputTensor(0).shape;
    // Expected NHWC: [1, height, width, 3].
    if (inputShape.length == 4) {
      _inputHeight = inputShape[1];
      _inputWidth = inputShape[2];
    }

    final outputShape = interpreter.getOutputTensor(0).shape;
    if (outputShape.length != 3) {
      throw StateError(
        'Unsupported model output shape $outputShape; expected 3 dimensions.',
      );
    }

    final expectedAttributes = 4 + classNames.length;
    if (outputShape[1] == expectedAttributes) {
      _attributesFirst = true;
      _numAttributes = outputShape[1];
      _numPredictions = outputShape[2];
    } else if (outputShape[2] == expectedAttributes) {
      _attributesFirst = false;
      _numPredictions = outputShape[1];
      _numAttributes = outputShape[2];
    } else {
      // Unknown class count: assume the smaller axis carries the attributes.
      _attributesFirst = outputShape[1] < outputShape[2];
      _numAttributes = _attributesFirst ? outputShape[1] : outputShape[2];
      _numPredictions = _attributesFirst ? outputShape[2] : outputShape[1];
      debugPrint(
        'MLDetector: unexpected output shape $outputShape, '
        'assuming $_numAttributes attributes x $_numPredictions predictions',
      );
    }
  }

  /// Analyses one camera frame. Never throws: a detection failure must not take
  /// the camera preview down in an emergency.
  Future<DetectionResult> detectFrame(CameraImage image) async {
    if (_interpreter == null) {
      return DetectionResult.empty(image.width, image.height);
    }
    try {
      final rgb = convertToImage(image);
      final detections = detectOnImage(rgb, image.width, image.height);
      return DetectionResult.fromDetections(
        detections,
        image.width,
        image.height,
      );
    } catch (error, stack) {
      debugPrint('MLDetector.detectFrame failed: $error\n$stack');
      return DetectionResult.empty(image.width, image.height);
    }
  }

  /// Runs inference on an already decoded image. Split out from [detectFrame]
  /// so it can be exercised without a camera.
  List<Detection> detectOnImage(img.Image frame, int frameWidth, int frameHeight) {
    final interpreter = _interpreter;
    if (interpreter == null) return const [];

    final resized =
        img.copyResize(frame, width: _inputWidth, height: _inputHeight);
    final input = _toInputTensor(resized);

    final output = List.generate(
      1,
      (_) => List.generate(
        _attributesFirst ? _numAttributes : _numPredictions,
        (_) => List<double>.filled(
          _attributesFirst ? _numPredictions : _numAttributes,
          0.0,
        ),
      ),
    );

    interpreter.run(input, output);
    return _parseOutput(output[0], frameWidth, frameHeight);
  }

  List<List<List<List<double>>>> _toInputTensor(img.Image resized) {
    return [
      List.generate(
        _inputHeight,
        (y) => List.generate(_inputWidth, (x) {
          final pixel = resized.getPixel(x, y);
          return [
            pixel.rNormalized.toDouble(),
            pixel.gNormalized.toDouble(),
            pixel.bNormalized.toDouble(),
          ];
        }),
      ),
    ];
  }

  List<Detection> _parseOutput(
    List<List<double>> output,
    int frameWidth,
    int frameHeight,
  ) {
    double attribute(int prediction, int attributeIndex) => _attributesFirst
        ? output[attributeIndex][prediction]
        : output[prediction][attributeIndex];

    // Ultralytics exports boxes either normalised to 0..1 or in input-pixel
    // units, depending on the export flags. Sniff it once from the data.
    var maxCoordinate = 0.0;
    for (var i = 0; i < _numPredictions; i++) {
      final value = attribute(i, 2); // width
      if (value > maxCoordinate) maxCoordinate = value;
    }
    final normalised = maxCoordinate <= 1.5;
    final scaleX = normalised ? frameWidth : frameWidth / _inputWidth;
    final scaleY = normalised ? frameHeight : frameHeight / _inputHeight;

    final detections = <Detection>[];
    for (var i = 0; i < _numPredictions; i++) {
      var bestClass = -1;
      var bestScore = 0.0;
      for (var c = 4; c < _numAttributes; c++) {
        final score = attribute(i, c);
        if (score > bestScore) {
          bestScore = score;
          bestClass = c - 4;
        }
      }
      if (bestClass < 0 || bestScore < confidenceThreshold) continue;

      final centerX = attribute(i, 0) * scaleX;
      final centerY = attribute(i, 1) * scaleY;
      final width = attribute(i, 2) * scaleX;
      final height = attribute(i, 3) * scaleY;

      final box = Rect(
        x: centerX - width / 2,
        y: centerY - height / 2,
        width: width,
        height: height,
      );

      final label =
          bestClass < classNames.length ? classNames[bestClass] : 'object';
      detections.add(
        Detection(
          label: label,
          confidence: bestScore,
          box: box,
          direction: label == 'stairs'
              ? stairDirection(box, frameWidth, frameHeight)
              : null,
        ),
      );
    }

    return nonMaxSuppression(detections, nmsIouThreshold);
  }

  /// Greedy per-label NMS. Without it a single exit sign yields a dozen boxes
  /// and the arrow jitters between them.
  static List<Detection> nonMaxSuppression(
    List<Detection> detections,
    double iouThreshold,
  ) {
    final sorted = [...detections]
      ..sort((a, b) => b.confidence.compareTo(a.confidence));
    final kept = <Detection>[];
    for (final candidate in sorted) {
      final overlaps = kept.any(
        (k) =>
            k.label == candidate.label &&
            k.box.iou(candidate.box) > iouThreshold,
      );
      if (!overlaps) kept.add(candidate);
    }
    return kept;
  }

  /// Stairs high in the frame recede upwards; stairs low in the frame drop away.
  static String stairDirection(Rect box, int frameWidth, int frameHeight) {
    if (box.centerY < frameHeight * 0.45) return 'up';
    if (box.centerY > frameHeight * 0.65) return 'down';
    if (box.centerX < frameWidth * 0.3) return 'left';
    if (box.centerX > frameWidth * 0.7) return 'right';
    return 'ahead';
  }

  /// Converts a camera frame to RGB.
  ///
  /// Android delivers YUV420 planes and iOS delivers BGRA8888, so both have to
  /// be handled; the previous version assumed NV21 and indexed the chroma
  /// planes with the luma stride, which scrambled the colours it then tried to
  /// detect exit signs by.
  static img.Image convertToImage(CameraImage image) {
    switch (image.format.group) {
      case ImageFormatGroup.bgra8888:
        return img.Image.fromBytes(
          width: image.width,
          height: image.height,
          bytes: image.planes[0].bytes.buffer,
          rowStride: image.planes[0].bytesPerRow,
          order: img.ChannelOrder.bgra,
        );
      case ImageFormatGroup.yuv420:
        return _yuv420ToImage(image);
      default:
        throw UnsupportedError(
          'Unsupported camera image format: ${image.format.group}',
        );
    }
  }

  static img.Image _yuv420ToImage(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final output = img.Image(width: width, height: height);

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;

    for (var y = 0; y < height; y++) {
      final yRow = y * yPlane.bytesPerRow;
      final uvRow = (y >> 1) * uvRowStride;
      for (var x = 0; x < width; x++) {
        final uvIndex = uvRow + (x >> 1) * uvPixelStride;
        final yValue = yPlane.bytes[yRow + x];
        final uValue = uPlane.bytes[uvIndex];
        final vValue = vPlane.bytes[uvIndex];
        output.setPixelRgb(
          x,
          y,
          _clamp8(yValue + 1.402 * (vValue - 128)),
          _clamp8(yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128)),
          _clamp8(yValue + 1.772 * (uValue - 128)),
        );
      }
    }
    return output;
  }

  static int _clamp8(double value) =>
      value < 0 ? 0 : (value > 255 ? 255 : value.round());

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
