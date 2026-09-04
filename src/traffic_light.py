import cv2
import numpy as np

class TrafficLightClassifier:
    def __init__(self):
        pass

    def classify_state_synthetic(self, frame):
        """
        Extracts the state of the Eastbound traffic light in the synthetic video
        by sampling the color of the specific light circles.
        Returns: "RED", "YELLOW", "GREEN", or "UNKNOWN"
        """
        # Eastbound light positions (cx = 370)
        # Red center: (370, 400), Yellow center: (370, 410), Green center: (370, 420)
        # We read BGR values. OpenCV uses BGR order.
        red_pixel = frame[400, 370]
        yellow_pixel = frame[410, 370]
        green_pixel = frame[420, 370]
        
        # Check colors (Note: BGR format)
        # Red: R > 200, G < 100, B < 100
        if red_pixel[2] > 200 and red_pixel[1] < 100:
            return "RED"
        # Green: G > 200, R < 100, B < 100
        elif green_pixel[1] > 200 and green_pixel[2] < 100:
            return "GREEN"
        # Yellow: R > 200, G > 200, B < 100
        elif yellow_pixel[2] > 200 and yellow_pixel[1] > 200:
            return "YELLOW"
            
        return "UNKNOWN"

    def classify_state_crop(self, light_crop):
        """
        Classifies a generic cropped traffic light bounding box.
        Splits the crop vertically into three equal regions (Top, Middle, Bottom)
        and evaluates the average brightness/intensity in HSV space to see
        which light is active.
        """
        if light_crop is None or light_crop.size == 0:
            return "UNKNOWN"
            
        h, w = light_crop.shape[:2]
        if h < 6 or w < 2:
            return "UNKNOWN"
            
        # Convert to HSV
        hsv = cv2.cvtColor(light_crop, cv2.COLOR_BGR2HSV)
        
        # Split crop into Top, Middle, and Bottom thirds
        third = h // 3
        top_crop = hsv[0:third, :]
        mid_crop = hsv[third:2*third, :]
        bot_crop = hsv[2*third:h, :]
        
        # Calculate mean saturation and value (brightness) for each section
        # We look for high Value (brightness) or Saturation to see which is lit
        top_val = np.mean(top_crop[:, :, 2])
        mid_val = np.mean(mid_crop[:, :, 2])
        bot_val = np.mean(bot_crop[:, :, 2])
        
        # Additionally, inspect HSV color masks to double-check
        # Red mask
        # Top segment red color check
        # Green segment green color check (bottom)
        # Yellow segment yellow color check (middle)
        
        max_idx = np.argmax([top_val, mid_val, bot_val])
        
        if max_idx == 0:
            return "RED"
        elif max_idx == 1:
            return "YELLOW"
        else:
            return "GREEN"
