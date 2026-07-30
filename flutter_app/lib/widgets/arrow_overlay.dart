import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../models/detection_result.dart';

class ArrowOverlay extends StatelessWidget {
  final double angle;
  final DetectionResult detectionResult;

  const ArrowOverlay({
    required this.angle,
    required this.detectionResult,
    Key? key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    final centerX = size.width / 2;
    final centerY = size.height / 2;

    return Stack(
      fit: StackFit.expand,
      children: [
        // Main arrow pointing to target
        CustomPaint(
          painter: ArrowPainter(
            angle: angle,
            size: size,
          ),
        ),

        // Confidence indicator
        Positioned(
          top: 80,
          left: 20,
          child: Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: _getConfidenceColor(detectionResult),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                if (detectionResult.exits.isNotEmpty)
                  _buildDetectionInfo(
                    'EXIT',
                    detectionResult.exits.first.confidence,
                    Colors.greenAccent,
                  ),
                if (detectionResult.stairs.isNotEmpty)
                  _buildDetectionInfo(
                    'STAIRS',
                    detectionResult.stairs.first.confidence,
                    Colors.yellowAccent,
                  ),
                if (detectionResult.doors.isNotEmpty)
                  _buildDetectionInfo(
                    'DOOR',
                    detectionResult.doors.first.confidence,
                    Colors.blueAccent,
                  ),
              ],
            ),
          ),
        ),

        // Distance indicator
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
                _getDistanceText(detectionResult),
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
        ),

        // Direction labels
        _buildDirectionLabel('↑ UP', 20, centerX - 30, context),
        _buildDirectionLabel('↓ DOWN', size.height - 60, centerX - 30, context),
        _buildDirectionLabel('← LEFT', centerY - 20, 20, context),
        _buildDirectionLabel('RIGHT →', centerY - 20, size.width - 70, context),
      ],
    );
  }

  Widget _buildDetectionInfo(
    String label,
    double confidence,
    Color color,
  ) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label,
          style: TextStyle(
            color: color,
            fontWeight: FontWeight.bold,
            fontSize: 14,
          ),
        ),
        const SizedBox(width: 8),
        Container(
          width: 40,
          height: 6,
          decoration: BoxDecoration(
            color: Colors.grey[700],
            borderRadius: BorderRadius.circular(3),
          ),
          child: Container(
            width: 40 * confidence,
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(3),
            ),
          ),
        ),
      ],
    );
  }

  String _getDistanceText(DetectionResult result) {
    if (result.exits.isNotEmpty) {
      return 'EXIT AHEAD! FOLLOW ARROW';
    } else if (result.stairs.isNotEmpty) {
      return 'STAIRS - ${result.stairs.first.direction?.toUpperCase() ?? "AHEAD"}';
    } else if (result.doors.isNotEmpty) {
      return 'DOOR DETECTED';
    }
    return '';
  }

  Color _getConfidenceColor(DetectionResult result) {
    if (result.exits.isNotEmpty &&
        result.exits.first.confidence > 0.8) {
      return Colors.green.withAlpha(200);
    }
    return Colors.orange.withAlpha(200);
  }

  Widget _buildDirectionLabel(
    String text,
    double top,
    double left,
    BuildContext context,
  ) {
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

class ArrowPainter extends CustomPainter {
  final double angle;
  final Size size;

  ArrowPainter({required this.angle, required this.size});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final fillPaint = Paint()
      ..color = Colors.greenAccent.withAlpha(100)
      ..style = PaintingStyle.fill;

    final centerX = size.width / 2;
    final centerY = size.height / 2;
    final arrowLength = 150.0;
    final arrowHeadSize = 40.0;

    // Calculate arrow direction
    final radians = angle * math.pi / 180;
    final endX = centerX + arrowLength * math.sin(radians);
    final endY = centerY - arrowLength * math.cos(radians);

    // Draw arrow shaft
    canvas.drawLine(
      Offset(centerX, centerY),
      Offset(endX, endY),
      paint,
    );

    // Draw arrow head (triangle)
    final headRadians1 = radians + math.pi / 6;
    final headRadians2 = radians - math.pi / 6;

    final head1X = endX + arrowHeadSize * math.sin(headRadians1);
    final head1Y = endY - arrowHeadSize * math.cos(headRadians1);

    final head2X = endX + arrowHeadSize * math.sin(headRadians2);
    final head2Y = endY - arrowHeadSize * math.cos(headRadians2);

    final path = Path()
      ..moveTo(endX, endY)
      ..lineTo(head1X, head1Y)
      ..lineTo(head2X, head2Y)
      ..close();

    canvas.drawPath(path, fillPaint);
    canvas.drawPath(path, paint);

    // Draw circular center indicator
    canvas.drawCircle(Offset(centerX, centerY), 15, fillPaint);
    canvas.drawCircle(Offset(centerX, centerY), 15, paint);
  }

  @override
  bool shouldRepaint(ArrowPainter oldDelegate) {
    return oldDelegate.angle != angle;
  }
}
