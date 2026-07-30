import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/detection_result.dart';

/// Draws the navigation arrow, detection boxes and confidence readouts over the
/// camera preview.
class ArrowOverlay extends StatelessWidget {
  /// Bearing in degrees clockwise from straight ahead.
  final double angle;
  final DetectionResult detectionResult;

  const ArrowOverlay({
    required this.angle,
    required this.detectionResult,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Stack(
      fit: StackFit.expand,
      children: [
        CustomPaint(
          painter: _ArrowPainter(
            angle: angle,
            boxes: detectionResult,
          ),
        ),
        Positioned(
          top: 96,
          left: 16,
          child: Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: _confidenceColor(detectionResult),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                if (detectionResult.exits.isNotEmpty)
                  _ConfidenceBar(
                    label: 'EXIT',
                    confidence: detectionResult.exits.first.confidence,
                    color: Colors.greenAccent,
                  ),
                if (detectionResult.stairs.isNotEmpty)
                  _ConfidenceBar(
                    label: 'STAIRS',
                    confidence: detectionResult.stairs.first.confidence,
                    color: Colors.yellowAccent,
                  ),
                if (detectionResult.doors.isNotEmpty)
                  _ConfidenceBar(
                    label: 'DOOR',
                    confidence: detectionResult.doors.first.confidence,
                    color: Colors.blueAccent,
                  ),
              ],
            ),
          ),
        ),
        Positioned(
          bottom: 100,
          left: 0,
          right: 0,
          child: Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(30),
              ),
              child: Text(
                headlineFor(detectionResult),
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
        ),
        _label('^ UP', 24, size.width / 2 - 20),
        _label('v DOWN', size.height - 64, size.width / 2 - 28),
        _label('< LEFT', size.height / 2 - 20, 16),
        _label('RIGHT >', size.height / 2 - 20, size.width - 76),
      ],
    );
  }

  /// The one line the user reads while running.
  static String headlineFor(DetectionResult result) {
    if (result.exits.isNotEmpty) return 'EXIT AHEAD - FOLLOW ARROW';
    if (result.stairs.isNotEmpty) {
      final direction = result.stairs.first.direction ?? 'ahead';
      return 'STAIRS - ${direction.toUpperCase()}';
    }
    if (result.doors.isNotEmpty) return 'DOOR DETECTED';
    return 'SCANNING';
  }

  static Color _confidenceColor(DetectionResult result) {
    final target = result.primaryTarget;
    if (target != null && target.label == 'exit' && target.confidence > 0.8) {
      return Colors.green.withAlpha(200);
    }
    return Colors.orange.withAlpha(200);
  }

  static Widget _label(String text, double top, double left) {
    return Positioned(
      top: top,
      left: left,
      child: Text(
        text,
        style: const TextStyle(
          color: Colors.white70,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
}

/// A label and a bar whose filled fraction is the detection confidence.
class _ConfidenceBar extends StatelessWidget {
  final String label;
  final double confidence;
  final Color color;

  const _ConfidenceBar({
    required this.label,
    required this.confidence,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 56,
            child: Text(
              label,
              style: TextStyle(
                color: color,
                fontWeight: FontWeight.bold,
                fontSize: 14,
              ),
            ),
          ),
          const SizedBox(width: 8),
          // A fixed-width Container passes tight constraints to its child, so
          // the fill has to be an Align/widthFactor rather than a sized box -
          // otherwise every bar renders full regardless of confidence.
          Container(
            width: 48,
            height: 6,
            decoration: BoxDecoration(
              color: Colors.grey.shade700,
              borderRadius: BorderRadius.circular(3),
            ),
            child: Align(
              alignment: Alignment.centerLeft,
              widthFactor: confidence.clamp(0.0, 1.0),
              child: Container(
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(3),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ArrowPainter extends CustomPainter {
  final double angle;
  final DetectionResult boxes;

  const _ArrowPainter({required this.angle, required this.boxes});

  @override
  void paint(Canvas canvas, Size size) {
    _paintDetectionBoxes(canvas, size);
    _paintArrow(canvas, size);
  }

  void _paintDetectionBoxes(Canvas canvas, Size size) {
    if (boxes.frameWidth <= 0 || boxes.frameHeight <= 0) return;
    final scaleX = size.width / boxes.frameWidth;
    final scaleY = size.height / boxes.frameHeight;

    void drawGroup(List<Detection> group, Color color) {
      final paint = Paint()
        ..color = color
        ..strokeWidth = 3
        ..style = PaintingStyle.stroke;
      for (final detection in group) {
        canvas.drawRect(
          Offset(detection.box.x * scaleX, detection.box.y * scaleY) &
              Size(detection.box.width * scaleX, detection.box.height * scaleY),
          paint,
        );
      }
    }

    drawGroup(boxes.exits, Colors.greenAccent);
    drawGroup(boxes.stairs, Colors.yellowAccent);
    drawGroup(boxes.doors, Colors.blueAccent);
  }

  void _paintArrow(Canvas canvas, Size size) {
    final stroke = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;
    final fill = Paint()
      ..color = Colors.greenAccent.withAlpha(100)
      ..style = PaintingStyle.fill;

    final center = Offset(size.width / 2, size.height / 2);
    final arrowLength = math.min(size.width, size.height) * 0.25;
    final headSize = arrowLength * 0.28;

    final radians = angle * math.pi / 180;
    final tip = Offset(
      center.dx + arrowLength * math.sin(radians),
      center.dy - arrowLength * math.cos(radians),
    );

    canvas.drawLine(center, tip, stroke);

    // Arrow head: two points swept back from the tip along the bearing.
    final left = radians + math.pi * 5 / 6;
    final right = radians - math.pi * 5 / 6;
    final path = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo(
        tip.dx + headSize * math.sin(left),
        tip.dy - headSize * math.cos(left),
      )
      ..lineTo(
        tip.dx + headSize * math.sin(right),
        tip.dy - headSize * math.cos(right),
      )
      ..close();

    canvas.drawPath(path, fill);
    canvas.drawPath(path, stroke);

    canvas.drawCircle(center, 15, fill);
    canvas.drawCircle(center, 15, stroke);
  }

  @override
  bool shouldRepaint(_ArrowPainter oldDelegate) =>
      oldDelegate.angle != angle || oldDelegate.boxes != boxes;
}
