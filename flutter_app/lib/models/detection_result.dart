/// Data model shared by the detector, the navigation service and the overlay.
///
/// Mirrors `emergency_path_finder/geometry.py` so the Python reference
/// implementation and the app agree on what a detection is.

class Detection {
  final String label;
  final double confidence;
  final Rect box;

  /// 'up', 'down', 'left', 'right' or 'ahead' - only set for stairs.
  final String? direction;

  const Detection({
    required this.label,
    required this.confidence,
    required this.box,
    this.direction,
  });

  double get centerX => box.centerX;
  double get centerY => box.centerY;

  @override
  String toString() =>
      'Detection($label, ${confidence.toStringAsFixed(2)}, $box)';
}

class Rect {
  final double x;
  final double y;
  final double width;
  final double height;

  const Rect({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  factory Rect.fromLTRB(double left, double top, double right, double bottom) {
    final l = left <= right ? left : right;
    final r = left <= right ? right : left;
    final t = top <= bottom ? top : bottom;
    final b = top <= bottom ? bottom : top;
    return Rect(x: l, y: t, width: r - l, height: b - t);
  }

  double get right => x + width;
  double get bottom => y + height;
  double get centerX => x + width / 2;
  double get centerY => y + height / 2;
  double get area => width * height;

  /// Intersection over union - used by the detector's NMS pass.
  double iou(Rect other) {
    final interWidth =
        (right < other.right ? right : other.right) - (x > other.x ? x : other.x);
    final interHeight = (bottom < other.bottom ? bottom : other.bottom) -
        (y > other.y ? y : other.y);
    if (interWidth <= 0 || interHeight <= 0) return 0.0;
    final intersection = interWidth * interHeight;
    final unionArea = area + other.area - intersection;
    if (unionArea <= 0) return 0.0;
    return intersection / unionArea;
  }

  @override
  String toString() => 'Rect(${x.toStringAsFixed(0)}, ${y.toStringAsFixed(0)}, '
      '${width.toStringAsFixed(0)}x${height.toStringAsFixed(0)})';
}

class DetectionResult {
  final List<Detection> exits;
  final List<Detection> stairs;
  final List<Detection> doors;
  final int frameWidth;
  final int frameHeight;

  /// Filled in by [NavigationService] once the frame has been analysed.
  double? arrowAngle;
  double? confidence;

  DetectionResult({
    required this.exits,
    required this.stairs,
    required this.doors,
    required this.frameWidth,
    required this.frameHeight,
  });

  /// An empty result for a frame of the given size.
  factory DetectionResult.empty(int width, int height) => DetectionResult(
        exits: const [],
        stairs: const [],
        doors: const [],
        frameWidth: width,
        frameHeight: height,
      );

  /// Builds a result by bucketing a flat detection list by label.
  factory DetectionResult.fromDetections(
    List<Detection> detections,
    int width,
    int height,
  ) {
    return DetectionResult(
      exits: detections.where((d) => d.label == 'exit').toList(),
      stairs: detections.where((d) => d.label == 'stairs').toList(),
      doors: detections.where((d) => d.label == 'door').toList(),
      frameWidth: width,
      frameHeight: height,
    );
  }

  bool get hasDetections =>
      exits.isNotEmpty || stairs.isNotEmpty || doors.isNotEmpty;

  int get totalDetections => exits.length + stairs.length + doors.length;

  /// Priority: a marked exit beats stairs, which beat a plain door.
  /// Within a class the most confident detection wins.
  Detection? get primaryTarget {
    for (final group in [exits, stairs, doors]) {
      if (group.isEmpty) continue;
      var best = group.first;
      for (final candidate in group) {
        if (candidate.confidence > best.confidence) best = candidate;
      }
      return best;
    }
    return null;
  }
}
