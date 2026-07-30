import 'dart:math' as math;
import '../models/detection_result.dart';

class NavigationService {
  /// Calculate arrow angle based on primary detection target
  double calculateArrowAngle(
    DetectionResult result,
    double deviceOrientation,
  ) {
    final target = result.primaryTarget;
    if (target == null) return 0.0;

    // Calculate angle from center of frame to target
    final frameCenterX = result.frameWidth / 2.0;
    final frameCenterY = result.frameHeight / 2.0;

    final deltaX = target.centerX - frameCenterX;
    final deltaY = target.centerY - frameCenterY;

    // Calculate angle in degrees
    var angle = math.atan2(deltaX, -deltaY) * 180 / math.pi;

    // Adjust for device orientation
    angle = (angle - deviceOrientation + 360) % 360;

    return angle;
  }

  /// Get distance estimate based on bounding box size
  double estimateDistance(Detection detection, int frameWidth) {
    // Rough approximation: larger box = closer object
    final boxArea = detection.box.width * detection.box.height;
    final frameArea = frameWidth * frameWidth;
    final ratio = boxArea / frameArea;

    // Convert to rough distance in meters (assuming ~2m at 10% of frame)
    if (ratio < 0.01) return 10.0;
    if (ratio < 0.05) return 5.0;
    if (ratio < 0.15) return 3.0;
    if (ratio < 0.30) return 1.5;
    return 0.5;
  }

  /// Generate navigation instruction based on detection
  String getNavigationInstruction(DetectionResult result) {
    if (result.exits.isNotEmpty) {
      final exit = result.exits.first;
      final distance = estimateDistance(exit, result.frameWidth);
      return 'EXIT FOUND! ${distance.toStringAsFixed(1)}m away';
    }

    if (result.stairs.isNotEmpty) {
      final stair = result.stairs.first;
      final direction = stair.direction ?? 'ahead';
      return 'Stairs $direction - ${stair.confidence > 0.7 ? "Clear path" : "Be careful"}';
    }

    if (result.doors.isNotEmpty) {
      final distance = estimateDistance(result.doors.first, result.frameWidth);
      return 'Door ahead at ${distance.toStringAsFixed(1)}m';
    }

    return 'Searching for exits... Move forward';
  }

  /// Analyze corridor layout for pathfinding without signs
  String analyzeCorridorLayout(DetectionResult result) {
    final frameCenter = result.frameWidth / 2.0;
    int leftOpenings = 0;
    int rightOpenings = 0;

    // Count doors/openings on left and right
    for (final door in result.doors) {
      if (door.centerX < frameCenter - 100) {
        leftOpenings++;
      } else if (door.centerX > frameCenter + 100) {
        rightOpenings++;
      }
    }

    if (leftOpenings > rightOpenings) {
      return 'Take LEFT - opening detected';
    } else if (rightOpenings > leftOpenings) {
      return 'Take RIGHT - opening detected';
    }

    // If no clear indication, suggest going forward
    return 'Continue forward - no obstacles detected';
  }

  /// Vanishing point detection for corridor navigation
  /// Returns angle to vanishing point (center of corridor)
  double? detectVanishingPoint(DetectionResult result) {
    if (result.doors.isEmpty && result.stairs.isEmpty) {
      return null;
    }

    // Calculate average position of structural elements
    final allDetections = [
      ...result.doors,
      ...result.stairs,
    ];

    if (allDetections.isEmpty) return null;

    final avgX = allDetections.fold<double>(0.0, (sum, d) => sum + d.centerX) /
        allDetections.length;
    final frameCenterX = result.frameWidth / 2.0;

    // Vanishing point angle (positive = right, negative = left)
    return (avgX - frameCenterX) / result.frameWidth * 45.0; // ±45 degrees max
  }

  /// Determine if person is going the right direction
  bool isMovingCorrectly(
    DetectionResult current,
    DetectionResult previous, {
    required double personHeading,
  }) {
    if (!current.hasDetections || !previous.hasDetections) return true;

    // If exit is getting closer (larger in frame), good direction
    if (current.primaryTarget != null && previous.primaryTarget != null) {
      final currentArea =
          current.primaryTarget!.box.width * current.primaryTarget!.box.height;
      final previousArea = previous.primaryTarget!.box.width *
          previous.primaryTarget!.box.height;

      return currentArea > previousArea * 0.95; // Allow small variations
    }

    return true;
  }
}
