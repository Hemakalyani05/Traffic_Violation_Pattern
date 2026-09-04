import cv2
import numpy as np

class VehicleDetector:
    def __init__(self, mode="contour", yolo_model_name="yolov8n.pt"):
        """
        Initializes the Vehicle Detector.
        mode: "contour" for synthetic simulation, "yolo" for real-world video.
        """
        self.mode = mode
        self.yolo_model = None
        self.background_frame = None
        
        if self.mode == "yolo":
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_model_name)
            except ImportError:
                print("Warning: 'ultralytics' not installed. Falling back to contour detection mode.")
                self.mode = "contour"

    def set_background(self, frame):
        """
        Sets the reference background frame for contour detection.
        Useful for synthetic simulation to detect stationary vehicles at red lights.
        """
        self.background_frame = frame.copy()

    def detect(self, frame):
        """
        Detects vehicles in the frame.
        Returns: A list of detections: [(x1, y1, x2, y2, confidence, class_id)]
        class_id: 2 (car), 7 (truck), 3 (motorcycle), 5 (bus) - matching COCO convention.
        """
        if self.mode == "yolo" and self.yolo_model is not None:
            return self._detect_yolo(frame)
        else:
            return self._detect_contour(frame)

    def _detect_yolo(self, frame):
        results = self.yolo_model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            # COCO classes: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
            if cls_id in [2, 3, 5, 7]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                # Filter out low confidence detections
                if conf > 0.35:
                    detections.append((int(x1), int(y1), int(x2), int(y2), conf, cls_id))
        return detections

    def _detect_contour(self, frame):
        """
        Fast contour-based detector for synthetic simulations.
        Uses absolute difference against a background frame to find cars,
        allowing detection of both moving and stationary vehicles.
        """
        if self.background_frame is None:
            # If no background frame is set, create a default road background
            # or use MOG2 background subtractor
            self.background_frame = np.zeros_like(frame)
            self.background_frame[:, :] = (34, 139, 34) # Green fields
            # Road area
            cv2.rectangle(self.background_frame, (0, 220), (frame.shape[1], 380), (50, 50, 50), -1)
            # Dashed yellow lane separator
            for x in range(0, frame.shape[1], 40):
                cv2.line(self.background_frame, (x, 300), (x + 20, 300), (0, 255, 255), 2)
            # Road borders
            cv2.line(self.background_frame, (0, 220), (frame.shape[1], 220), (220, 220, 220), 2)
            cv2.line(self.background_frame, (0, 380), (frame.shape[1], 380), (220, 220, 220), 2)
            # Stop lines
            cv2.line(self.background_frame, (350, 300), (350, 380), (255, 255, 255), 4)
            cv2.line(self.background_frame, (450, 220), (450, 300), (255, 255, 255), 4)
            
        # Compute absolute difference
        diff = cv2.absdiff(frame, self.background_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold the diff image
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
        # Perform morphological operations to clean up small noise (like traffic light change)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter contours by size to match car shapes (width: 30-60, height: 15-35)
            # Ignore traffic lights or lane line differences by restricting y region to the road
            if 300 < area < 4000 and 150 < y + h//2 < 450:
                # Classify based on size
                class_id = 2  # default: car
                if w > 50:
                    class_id = 7  # truck
                
                # Bounding box coordinates, confidence (1.0 for perfect mock), class_id
                detections.append((x, y, x + w, y + h, 1.0, class_id))
                
        return detections
