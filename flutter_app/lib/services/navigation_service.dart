import 'dart:math' as math;

import '../models/detection_result.dart';

/// Direction labels the overlay can render.
class NavDirection {
  static const String left = 'LEFT';
  static const String right = 'RIGHT';
  static const String straight = 'STRAIGHT';
  static const String upstairs = 'UPSTAIRS';
  static const String downstairs = 'DOWNSTAIRS';
  static const String forward = 'FORWARD';
}

/// How confident we are that a usable escape route is on screen.
class Urgency {
  static const String critical = 'CRITICAL';
  static const String high = 'HIGH';
  static const String medium = 'MEDIUM';
  static const String low = 'LOW';
}

/// Turns detections into an instruction.
///
/// This mirrors `emergency_path_finder/navigation.py`; the thresholds are
/// regression-tested there against synthetic scenes.
class NavigationService {
  /// Fraction of frame width a target must be off-centre before we call a turn.
  static const double turnDeadzone = 0.12;

  /// Arrow bearing in degrees clockwise from straight ahead (0 = ahead,
  /// 90 = hard right, 270 = hard left).
  double calculateArrowAngle(
    DetectionResult result,
    double deviceOrientation,
  ) {
    final target = result.primaryTarget;
    if (target == null) return 0.0;
    if (result.frameWidth <= 0 || result.frameHeight <= 0) return 0.0;

    final deltaX = target.centerX - result.frameWidth / 2.0;
    final deltaY = target.centerY - result.frameHeight / 2.0;

    final angle = math.atan2(deltaX, -deltaY) * 180 / math.pi;
    // Dart's % always returns a non-negative value for a positive divisor, so
    // this lands in [0, 360) without an extra wrap.
    return (angle - deviceOrientation) % 360;
  }

  /// Which way to move, given what is on screen.
  String getBestDirection(DetectionResult result) {
    final target = result.primaryTarget;
    if (target == null) return NavDirection.forward;

    if (target.label == 'stairs') {
      if (target.direction == 'up') return NavDirection.upstairs;
      if (target.direction == 'down') return NavDirection.downstairs;
    }

    final centerX = result.frameWidth / 2.0;
    final deadzone = result.frameWidth * turnDeadzone;
    final offset = target.centerX - centerX;
    if (offset < -deadzone) return NavDirection.left;
    if (offset > deadzone) return NavDirection.right;
    return NavDirection.straight;
  }

  /// Rough distance from apparent size, assuming a ~2 m tall object.
  /// Only the ordering is meaningful.
  double estimateDistance(Detection detection, int frameHeight) {
    if (frameHeight <= 0 || detection.box.height <= 0) return 15.0;
    final ratio = detection.box.height / frameHeight;
    final distance = 1.0 / math.max(ratio, 0.001);
    return distance.clamp(0.5, 15.0);
  }

  /// Short, imperative text for the status bar.
  String getNavigationInstruction(DetectionResult result) {
    final target = result.primaryTarget;
    if (target == null) return 'Searching for exits - keep moving forward';

    final direction = getBestDirection(result);
    final distance = estimateDistance(target, result.frameHeight);

    switch (target.label) {
      case 'exit':
        return direction == NavDirection.straight
            ? 'EXIT AHEAD - ${distance.toStringAsFixed(0)} m, go straight'
            : 'EXIT FOUND - ${distance.toStringAsFixed(0)} m, '
                'go ${direction.toLowerCase()}';
      case 'stairs':
        final where = target.direction ?? 'ahead';
        final caution = target.confidence > 0.6 ? 'clear' : 'take care';
        return 'Stairs $where at ${distance.toStringAsFixed(0)} m - $caution';
      default:
        return 'Door at ${distance.toStringAsFixed(0)} m - '
            'go ${direction.toLowerCase()}';
    }
  }

  /// Urgency classification. CRITICAL is the *best* case: a marked exit is
  /// visible and the user should move now.
  String calculateUrgency(DetectionResult result, double lightQuality) {
    final hasExit = result.exits.isNotEmpty;
    if (hasExit && lightQuality > 0.4) return Urgency.critical;
    if (result.hasDetections) return Urgency.high;
    if (lightQuality > 0.5) return Urgency.medium;
    return Urgency.low;
  }

  /// Pathfinding without signage: pick the side with more openings.
  String analyzeCorridorLayout(DetectionResult result) {
    final centerX = result.frameWidth / 2.0;
    final deadzone = result.frameWidth * turnDeadzone;
    var left = 0;
    var right = 0;

    for (final opening in [...result.doors, ...result.stairs]) {
      if (opening.centerX < centerX - deadzone) {
        left++;
      } else if (opening.centerX > centerX + deadzone) {
        right++;
      }
    }

    if (left > right) return 'Take LEFT - opening detected';
    if (right > left) return 'Take RIGHT - opening detected';
    return 'Continue forward - no obstacles detected';
  }

  /// Average horizontal position of the structural detections, expressed as an
  /// angle in [-45, 45] degrees. Stands in for a corridor vanishing point when
  /// no signage is visible.
  double? detectVanishingPoint(DetectionResult result) {
    final structural = [...result.doors, ...result.stairs];
    if (structural.isEmpty || result.frameWidth <= 0) return null;

    var sum = 0.0;
    for (final detection in structural) {
      sum += detection.centerX;
    }
    final average = sum / structural.length;
    return (average - result.frameWidth / 2.0) / result.frameWidth * 45.0;
  }

  /// True while the target is growing in frame, i.e. the user is closing on it.
  bool isMovingCorrectly(DetectionResult current, DetectionResult previous) {
    final currentTarget = current.primaryTarget;
    final previousTarget = previous.primaryTarget;
    if (currentTarget == null || previousTarget == null) return true;

    // 5% tolerance absorbs the frame-to-frame jitter of the detector.
    return currentTarget.box.area > previousTarget.box.area * 0.95;
  }
}
