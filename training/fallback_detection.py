"""
Utilities for detection, navigation, and fallback methods
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

class FallbackDetector:
    """Fallback detection methods when ML fails (low light, no signs)"""
    
    @staticmethod
    def detect_doors(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect doors using edge detection and vertical lines
        Works in low light when exit signs not visible
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for low light enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Dilate to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        doors = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Door-like characteristics: tall and narrow
            aspect_ratio = h / (w + 1e-5)
            area = w * h
            
            if aspect_ratio > 1.5 and area > 5000:  # Tall, rectangular
                doors.append((x, y, w, h))
        
        return doors
    
    @staticmethod
    def detect_stairs_edges(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect stairs using edge patterns
        Stairs have distinctive diagonal/parallel line patterns
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection with emphasis on diagonal edges
        edges = cv2.Canny(gray, 30, 100)
        
        # Hough line transform to find stair edges
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        stair_regions = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Diagonal or near-diagonal lines suggest stairs
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if 20 < angle < 70 or 110 < angle < 160:
                    x, y, w, h = min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)
                    stair_regions.append((x, y, w, h))
        
        return stair_regions
    
    @staticmethod
    def detect_color_signs(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect green/red exit signs by color
        Most emergency exit signs are bright green or red
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Green detection (standard exit sign color)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Red detection (some exit signs are red)
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Combine masks
        mask = mask_green | mask_red
        
        # Morphology to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        signs = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > 500:  # Minimum size threshold
                signs.append((x, y, w, h))
        
        return signs
    
    @staticmethod
    def detect_vanishing_point(image: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect vanishing point of corridor/hallway
        In corridors, perspective lines converge to a point
        Use this when no exit signs visible
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        if lines is None or len(lines) < 4:
            return None
        
        # Find intersection points of lines
        points = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]
                
                # Line intersection calculation
                denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                if abs(denom) < 1e-5:
                    continue
                
                t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
                
                px = x1 + t*(x2-x1)
                py = y1 + t*(y2-y1)
                
                # Vanishing point should be within image bounds
                if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                    points.append((px, py))
        
        if not points:
            return None
        
        # Average intersection points to find vanishing point
        avg_x = np.mean([p[0] for p in points])
        avg_y = np.mean([p[1] for p in points])
        
        return (avg_x, avg_y)
    
    @staticmethod
    def estimate_depth_map(image: np.ndarray) -> np.ndarray:
        """
        Estimate depth using simple stereo-like approach
        Closer = larger features, farther = smaller
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use edge density as proxy for depth
        edges = cv2.Canny(gray, 50, 150)
        
        # Gaussian blur to smooth depth
        depth = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
        
        return depth
    
    @staticmethod
    def detect_light_sources(image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect bright areas (emergency lights, exit signs)
        Emergency lights are typically very bright
        """
        # Convert to HSV for brightness
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = hsv[:,:,2].astype(np.float32)
        
        # Threshold for bright areas
        _, bright_mask = cv2.threshold(brightness, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lights = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lights.append((cx, cy))
        
        return lights

class NavigationHelper:
    """Helper functions for navigation decisions"""
    
    @staticmethod
    def get_best_direction(doors: List, stairs: List, signs: List, image_shape: Tuple) -> str:
        """
        Given detected objects, recommend direction
        Priority: signs > stairs > doors > widest opening
        """
        h, w = image_shape[:2]
        center_x = w / 2
        
        # Priority 1: Exit signs point directly
        if signs:
            avg_x = np.mean([s[0] + s[2]/2 for s in signs])
            if avg_x < center_x - 50:
                return "LEFT"
            elif avg_x > center_x + 50:
                return "RIGHT"
            else:
                return "STRAIGHT"
        
        # Priority 2: Stairs (go up or forward)
        if stairs:
            return "UPSTAIRS" if stairs[0][1] < h/2 else "DOWNSTAIRS"
        
        # Priority 3: Widest door opening
        if doors:
            widest = max(doors, key=lambda d: d[2])
            door_x = widest[0] + widest[2]/2
            if door_x < center_x - 50:
                return "LEFT"
            elif door_x > center_x + 50:
                return "RIGHT"
            else:
                return "STRAIGHT"
        
        return "FORWARD - No obstacles"
    
    @staticmethod
    def calculate_urgency_level(
        has_exit: bool,
        has_stairs: bool,
        has_doors: bool,
        light_quality: float
    ) -> str:
        """
        Determine urgency level based on detections
        Returns: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
        """
        if has_exit and light_quality > 0.7:
            return "CRITICAL"  # Clear path to exit
        elif has_stairs or has_doors:
            return "HIGH"
        elif light_quality > 0.5:
            return "MEDIUM"
        else:
            return "LOW"  # Low visibility, move carefully

if __name__ == "__main__":
    print("Fallback detection utilities loaded successfully")
