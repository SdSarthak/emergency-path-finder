import 'package:flutter_test/flutter_test.dart';

import 'package:emergency_path_finder/models/detection_result.dart';
import 'package:emergency_path_finder/services/navigation_service.dart';

const int frameWidth = 640;
const int frameHeight = 480;

Detection detection(
  String label,
  double centerX,
  double centerY, {
  double width = 60,
  double height = 60,
  double confidence = 0.8,
  String? direction,
}) {
  return Detection(
    label: label,
    confidence: confidence,
    box: Rect(
      x: centerX - width / 2,
      y: centerY - height / 2,
      width: width,
      height: height,
    ),
    direction: direction,
  );
}

DetectionResult resultWith(List<Detection> detections) =>
    DetectionResult.fromDetections(detections, frameWidth, frameHeight);

void main() {
  final service = NavigationService();

  group('Rect', () {
    test('derives its centre and area', () {
      const box = Rect(x: 10, y: 20, width: 40, height: 60);
      expect(box.centerX, 30);
      expect(box.centerY, 50);
      expect(box.area, 2400);
      expect(box.right, 50);
      expect(box.bottom, 80);
    });

    test('fromLTRB normalises reversed corners', () {
      final box = Rect.fromLTRB(50, 80, 10, 20);
      expect(box.x, 10);
      expect(box.y, 20);
      expect(box.width, 40);
      expect(box.height, 60);
    });

    test('iou is 1 for identical boxes and 0 when disjoint', () {
      const a = Rect(x: 0, y: 0, width: 10, height: 10);
      const b = Rect(x: 100, y: 100, width: 10, height: 10);
      expect(a.iou(a), closeTo(1.0, 1e-9));
      expect(a.iou(b), 0.0);
    });

    test('iou of a half overlap', () {
      const a = Rect(x: 0, y: 0, width: 10, height: 10);
      const b = Rect(x: 5, y: 0, width: 10, height: 10);
      expect(a.iou(b), closeTo(50 / 150, 1e-9));
    });
  });

  group('DetectionResult', () {
    test('buckets detections by label', () {
      final result = resultWith([
        detection('exit', 100, 100),
        detection('door', 200, 200),
        detection('door', 300, 200),
      ]);
      expect(result.exits.length, 1);
      expect(result.doors.length, 2);
      expect(result.stairs, isEmpty);
      expect(result.totalDetections, 3);
    });

    test('prefers an exit over stairs and doors', () {
      final result = resultWith([
        detection('door', 100, 100, confidence: 0.99),
        detection('stairs', 200, 100, confidence: 0.95),
        detection('exit', 300, 100, confidence: 0.40),
      ]);
      expect(result.primaryTarget!.label, 'exit');
    });

    test('picks the most confident detection within a class', () {
      final result = resultWith([
        detection('exit', 100, 100, confidence: 0.4),
        detection('exit', 500, 100, confidence: 0.9),
      ]);
      expect(result.primaryTarget!.centerX, 500);
    });

    test('an empty result has no target', () {
      final result = DetectionResult.empty(frameWidth, frameHeight);
      expect(result.hasDetections, isFalse);
      expect(result.primaryTarget, isNull);
    });
  });

  group('arrow angle', () {
    test('points straight up for a target above centre', () {
      final result = resultWith([detection('exit', 320, 100)]);
      expect(service.calculateArrowAngle(result, 0), closeTo(0.0, 1e-9));
    });

    test('points right for a target to the right', () {
      final result = resultWith([detection('exit', 620, 240)]);
      expect(service.calculateArrowAngle(result, 0), closeTo(90.0, 1e-9));
    });

    test('points left for a target to the left', () {
      final result = resultWith([detection('exit', 20, 240)]);
      expect(service.calculateArrowAngle(result, 0), closeTo(270.0, 1e-9));
    });

    test('device orientation rotates the arrow and wraps into [0, 360)', () {
      final result = resultWith([detection('exit', 320, 100)]);
      expect(service.calculateArrowAngle(result, 45), closeTo(315.0, 1e-9));
    });

    test('is zero when there is nothing to aim at', () {
      final result = DetectionResult.empty(frameWidth, frameHeight);
      expect(service.calculateArrowAngle(result, 30), 0.0);
    });
  });

  group('direction', () {
    test('turns towards an off-centre target', () {
      expect(
        service.getBestDirection(resultWith([detection('exit', 40, 240)])),
        NavDirection.left,
      );
      expect(
        service.getBestDirection(resultWith([detection('exit', 600, 240)])),
        NavDirection.right,
      );
    });

    test('stays straight inside the dead zone', () {
      expect(
        service.getBestDirection(resultWith([detection('exit', 340, 240)])),
        NavDirection.straight,
      );
    });

    test('stairs produce vertical directions', () {
      expect(
        service.getBestDirection(
          resultWith([detection('stairs', 320, 100, direction: 'up')]),
        ),
        NavDirection.upstairs,
      );
      expect(
        service.getBestDirection(
          resultWith([detection('stairs', 320, 400, direction: 'down')]),
        ),
        NavDirection.downstairs,
      );
    });

    test('keeps the user moving when nothing is detected', () {
      expect(
        service.getBestDirection(DetectionResult.empty(frameWidth, frameHeight)),
        NavDirection.forward,
      );
    });
  });

  group('distance', () {
    test('bigger boxes read as closer', () {
      final near = detection('exit', 320, 240, width: 200, height: 300);
      final far = detection('exit', 320, 240, width: 20, height: 30);
      expect(
        service.estimateDistance(near, frameHeight),
        lessThan(service.estimateDistance(far, frameHeight)),
      );
    });

    test('is clamped to a sane range', () {
      final huge = detection('exit', 320, 240, width: 640, height: 480);
      final sliver = detection('exit', 320, 240, width: 640, height: 1);
      expect(service.estimateDistance(huge, frameHeight), greaterThanOrEqualTo(0.5));
      expect(service.estimateDistance(sliver, frameHeight), lessThanOrEqualTo(15.0));
    });

    test('a degenerate box does not divide by zero', () {
      final flat = detection('exit', 320, 240, width: 10, height: 0);
      expect(service.estimateDistance(flat, frameHeight), 15.0);
    });
  });

  group('urgency', () {
    test('a visible exit in decent light is critical', () {
      expect(
        service.calculateUrgency(resultWith([detection('exit', 320, 240)]), 0.8),
        Urgency.critical,
      );
    });

    test('an exit in darkness is only high', () {
      expect(
        service.calculateUrgency(resultWith([detection('exit', 320, 240)]), 0.1),
        Urgency.high,
      );
    });

    test('nothing detected degrades with the light', () {
      final empty = DetectionResult.empty(frameWidth, frameHeight);
      expect(service.calculateUrgency(empty, 0.9), Urgency.medium);
      expect(service.calculateUrgency(empty, 0.1), Urgency.low);
    });
  });

  group('instructions', () {
    test('name the exit and the turn', () {
      final text = service.getNavigationInstruction(
        resultWith([detection('exit', 600, 240, height: 120)]),
      );
      expect(text, contains('EXIT'));
      expect(text, contains('right'));
    });

    test('fall back to a search prompt', () {
      expect(
        service.getNavigationInstruction(
          DetectionResult.empty(frameWidth, frameHeight),
        ),
        contains('Searching'),
      );
    });
  });

  group('corridor reasoning', () {
    test('prefers the side with more openings', () {
      final result = resultWith([
        detection('door', 60, 240),
        detection('door', 120, 240),
        detection('door', 600, 240),
      ]);
      expect(service.analyzeCorridorLayout(result), contains('LEFT'));
    });

    test('reports no bias when openings are balanced', () {
      final result = resultWith([
        detection('door', 60, 240),
        detection('door', 600, 240),
      ]);
      expect(service.analyzeCorridorLayout(result), contains('forward'));
    });

    test('vanishing point is null without structural detections', () {
      final result = resultWith([detection('exit', 320, 240)]);
      expect(service.detectVanishingPoint(result), isNull);
    });

    test('vanishing point leans towards the detected structure', () {
      final result = resultWith([detection('door', 640, 240)]);
      expect(service.detectVanishingPoint(result), closeTo(22.5, 1e-9));
    });
  });

  group('progress', () {
    test('a growing target means the user is closing in', () {
      final previous = resultWith([
        detection('exit', 320, 240, width: 50, height: 50),
      ]);
      final current = resultWith([
        detection('exit', 320, 240, width: 100, height: 100),
      ]);
      expect(service.isMovingCorrectly(current, previous), isTrue);
    });

    test('a shrinking target means the user is walking away', () {
      final previous = resultWith([
        detection('exit', 320, 240, width: 100, height: 100),
      ]);
      final current = resultWith([
        detection('exit', 320, 240, width: 40, height: 40),
      ]);
      expect(service.isMovingCorrectly(current, previous), isFalse);
    });

    test('assumes the best when there is nothing to compare', () {
      final empty = DetectionResult.empty(frameWidth, frameHeight);
      expect(service.isMovingCorrectly(empty, empty), isTrue);
    });
  });
}
