import 'package:flutter_test/flutter_test.dart';

import 'package:emergency_path_finder/models/detection_result.dart';
import 'package:emergency_path_finder/services/ml_detector.dart';

Detection detection(String label, Rect box, double confidence) =>
    Detection(label: label, confidence: confidence, box: box);

void main() {
  group('non-max suppression', () {
    test('keeps the most confident box of an overlapping cluster', () {
      final strong =
          detection('exit', const Rect(x: 0, y: 0, width: 10, height: 10), 0.9);
      final weak =
          detection('exit', const Rect(x: 1, y: 1, width: 10, height: 10), 0.4);

      final kept = MLDetector.nonMaxSuppression([weak, strong], 0.4);
      expect(kept, hasLength(1));
      expect(kept.first.confidence, 0.9);
    });

    test('does not merge boxes with different labels', () {
      const box = Rect(x: 0, y: 0, width: 10, height: 10);
      final kept = MLDetector.nonMaxSuppression(
        [detection('exit', box, 0.9), detection('door', box, 0.8)],
        0.45,
      );
      expect(kept, hasLength(2));
    });

    test('keeps distinct objects of the same class', () {
      final kept = MLDetector.nonMaxSuppression(
        [
          detection('door', const Rect(x: 0, y: 0, width: 10, height: 10), 0.9),
          detection(
            'door',
            const Rect(x: 200, y: 0, width: 10, height: 10),
            0.8,
          ),
        ],
        0.45,
      );
      expect(kept, hasLength(2));
    });

    test('handles an empty list', () {
      expect(MLDetector.nonMaxSuppression([], 0.45), isEmpty);
    });
  });

  group('stair direction', () {
    Rect at(double centerX, double centerY) =>
        Rect(x: centerX - 20, y: centerY - 20, width: 40, height: 40);

    test('high in the frame reads as up', () {
      expect(MLDetector.stairDirection(at(320, 100), 640, 480), 'up');
    });

    test('low in the frame reads as down', () {
      expect(MLDetector.stairDirection(at(320, 400), 640, 480), 'down');
    });

    test('mid frame falls back to a horizontal call', () {
      expect(MLDetector.stairDirection(at(100, 260), 640, 480), 'left');
      expect(MLDetector.stairDirection(at(560, 260), 640, 480), 'right');
      expect(MLDetector.stairDirection(at(320, 260), 640, 480), 'ahead');
    });
  });

  test('the detector reports it is not ready before initialization', () {
    expect(MLDetector().isReady, isFalse);
  });

  test('class names line up with the training config', () {
    expect(MLDetector.classNames, ['exit', 'stairs', 'door']);
  });
}
