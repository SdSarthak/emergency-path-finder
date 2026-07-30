class Detection {
  final String label;
  final double confidence;
  final Rect box;
  final String? direction; // 'up', 'down', 'left', 'right' for stairs

  Detection({
    required this.label,
    required this.confidence,
    required this.box,
    this.direction,
  });
}

class Rect {
  final double x;
  final double y;
  final double width;
  final double height;

  Rect({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  double get centerX => x + width / 2;
  double get centerY => y + height / 2;
}

class DetectionResult {
  final List<Detection> exits;
  final List<Detection> stairs;
  final List<Detection> doors;
  final int frameWidth;
  final int frameHeight;
  double? arrowAngle;
  double? confidence;

  DetectionResult({
    required this.exits,
    required this.stairs,
    required this.doors,
    required this.frameWidth,
    required this.frameHeight,
  });

  bool get hasDetections =>
      exits.isNotEmpty || stairs.isNotEmpty || doors.isNotEmpty;

  Detection? get primaryTarget {
    // Priority: exits > stairs > doors
    if (exits.isNotEmpty) return exits.first;
    if (stairs.isNotEmpty) return stairs.first;
    if (doors.isNotEmpty) return doors.first;
    return null;
  }
}
