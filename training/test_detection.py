"""
Test detection system on sample images or video
Useful for quick validation before deploying to mobile
"""

import cv2
import numpy as np
from pathlib import Path
from fallback_detection import FallbackDetector, NavigationHelper
import argparse

class DetectionTester:
    def __init__(self):
        self.detector = FallbackDetector()
        self.nav_helper = NavigationHelper()
    
    def test_on_image(self, image_path: str, visualize=True):
        """Test detection on single image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image {image_path}")
            return
        
        print(f"\nTesting on: {image_path}")
        print("="*50)
        
        # Run detections
        signs = self.detector.detect_color_signs(image)
        doors = self.detector.detect_doors(image)
        stairs = self.detector.detect_stairs_edges(image)
        lights = self.detector.detect_light_sources(image)
        vanishing = self.detector.detect_vanishing_point(image)
        
        # Print results
        print(f"Exit signs: {len(signs)}")
        print(f"Doors: {len(doors)}")
        print(f"Stairs: {len(stairs)}")
        print(f"Light sources: {len(lights)}")
        print(f"Vanishing point: {vanishing}")
        
        # Get navigation advice
        direction = self.nav_helper.get_best_direction(
            doors, stairs, signs, image.shape
        )
        print(f"\nRecommended direction: {direction}")
        
        # Visualize if requested
        if visualize:
            self._visualize_detections(image, signs, doors, stairs, lights, vanishing)
    
    def test_on_video(self, video_path: str):
        """Test detection on video stream"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        print(f"\nTesting on video: {video_path}")
        print("="*50)
        print("Press 'q' to quit")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detections every 5 frames (faster)
            if frame_count % 5 == 0:
                signs = self.detector.detect_color_signs(frame)
                doors = self.detector.detect_doors(frame)
                stairs = self.detector.detect_stairs_edges(frame)
                lights = self.detector.detect_light_sources(frame)
                
                # Draw on frame
                for sign in signs:
                    x, y, w, h = sign
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "EXIT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                for door in doors:
                    x, y, w, h = door
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, "DOOR", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                for stair in stairs:
                    x, y, w, h = stair
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, "STAIRS", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show info
                cv2.putText(frame, f"Signs: {len(signs)} Doors: {len(doors)} Stairs: {len(stairs)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display
            cv2.imshow("Detection Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _visualize_detections(self, image, signs, doors, stairs, lights, vanishing):
        """Draw detections on image"""
        display = image.copy()
        
        # Draw exits
        for x, y, w, h in signs:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(display, "EXIT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw doors
        for x, y, w, h in doors:
            cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv2.putText(display, "DOOR", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Draw stairs
        for x, y, w, h in stairs:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 3)
            cv2.putText(display, "STAIRS", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Draw light sources
        for x, y in lights:
            cv2.circle(display, (x, y), 10, (255, 255, 0), 2)
        
        # Draw vanishing point
        if vanishing:
            vx, vy = vanishing
            cv2.circle(display, (int(vx), int(vy)), 8, (255, 0, 255), 3)
        
        # Show
        cv2.imshow("Detections", display)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def benchmark(self, image_path: str, iterations=100):
        """Benchmark detection speed"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image {image_path}")
            return
        
        import time
        
        print(f"\nBenchmarking on: {image_path}")
        print("="*50)
        
        # Benchmark each detection method
        methods = {
            "Color signs": lambda: self.detector.detect_color_signs(image),
            "Doors": lambda: self.detector.detect_doors(image),
            "Stairs": lambda: self.detector.detect_stairs_edges(image),
            "Lights": lambda: self.detector.detect_light_sources(image),
            "Vanishing point": lambda: self.detector.detect_vanishing_point(image),
        }
        
        for method_name, method_func in methods.items():
            start = time.time()
            for _ in range(iterations):
                method_func()
            elapsed = time.time() - start
            
            avg_time = (elapsed / iterations) * 1000  # Convert to ms
            fps = 1000 / avg_time if avg_time > 0 else 0
            
            print(f"{method_name:20} | {avg_time:6.2f}ms | {fps:5.1f} FPS")
        
        print("="*50)
        total_time = sum(self._time_method(method_func, iterations) for method_func in methods.values())
        print(f"Total (all methods):  | {total_time:6.2f}ms | {1000/total_time:5.1f} FPS")
    
    @staticmethod
    def _time_method(method_func, iterations):
        import time
        start = time.time()
        for _ in range(iterations):
            method_func()
        return (time.time() - start) / iterations * 1000

def main():
    parser = argparse.ArgumentParser(description="Test detection system")
    parser.add_argument('--image', help='Test on image')
    parser.add_argument('--video', help='Test on video')
    parser.add_argument('--camera', action='store_true', help='Test on webcam')
    parser.add_argument('--benchmark', help='Benchmark on image')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    tester = DetectionTester()
    
    if args.image:
        tester.test_on_image(args.image, not args.no_visualize)
    elif args.video:
        tester.test_on_video(args.video)
    elif args.camera:
        tester.test_on_video(0)
    elif args.benchmark:
        tester.benchmark(args.benchmark)
    else:
        print("Emergency Path Finder - Detection Tester")
        print("="*50)
        print("\nUsage:")
        print("  python test_detection.py --image <image.jpg>")
        print("  python test_detection.py --video <video.mp4>")
        print("  python test_detection.py --camera")
        print("  python test_detection.py --benchmark <image.jpg>")
        print("="*50)

if __name__ == "__main__":
    main()
