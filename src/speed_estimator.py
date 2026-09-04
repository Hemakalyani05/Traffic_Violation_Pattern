import cv2
import numpy as np

class SpeedEstimator:
    def __init__(self, fps=30, mode="flat", calibration_points=None, real_world_dims=(10, 40)):
        """
        Initializes the Speed Estimator.
        fps: Video frames per second.
        mode: "flat" for synthetic 2D video, "homography" for perspective calibrated video.
        calibration_points: List of 4 coordinate tuples in pixels [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                            ordered: top-left, top-right, bottom-right, bottom-left.
        real_world_dims: Tuple (width, height) in meters representing the real size of the calibration box.
        """
        self.fps = fps
        self.mode = mode
        self.real_world_dims = real_world_dims
        self.H = None
        self.scale_factor = 12.0  # Synthetic mode scale: pixels/frame * 12.0 = km/h
        
        if self.mode == "homography" and calibration_points is not None:
            self.update_calibration(calibration_points, real_world_dims)

    def update_calibration(self, points, dims):
        """
        Computes the homography projection matrix H mapping pixel space to meter space.
        points: List of 4 points in pixels (src).
        dims: (width, height) in meters (dst).
        """
        self.real_world_dims = dims
        w_m, h_m = dims
        
        # Source points in image pixels
        src = np.array(points, dtype=np.float32)
        
        # Destination points in real-world meters
        # top-left, top-right, bottom-right, bottom-left
        dst = np.array([
            [0, 0],
            [w_m, 0],
            [w_m, h_m],
            [0, h_m]
        ], dtype=np.float32)
        
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.mode = "homography"

    def estimate_speed(self, trajectory, window=10):
        """
        Estimates the speed of a vehicle based on its trajectory history.
        window: Number of frames back to look for displacement.
        Returns: Estimated speed in km/h, or 0.0 if not enough data.
        """
        if len(trajectory) < window + 1:
            return 0.0
            
        p_current = trajectory[-1]
        p_past = trajectory[-window - 1]
        
        if self.mode == "homography" and self.H is not None:
            # Transform pixel coordinates to real-world meter coordinates
            pts = np.array([[p_past], [p_current]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pts, self.H)
            
            p_past_m = transformed[0][0]
            p_current_m = transformed[1][0]
            
            # Compute distance in meters
            dist_meters = np.linalg.norm(p_current_m - p_past_m)
            
            # Speed = (distance / time) * 3.6 (to get km/h)
            # time = window / fps
            time_seconds = window / self.fps
            speed_kmh = (dist_meters / time_seconds) * 3.6
            return float(round(speed_kmh, 1))
            
        else:
            # Flat mode: Linear pixel distance calculation
            dx = p_current[0] - p_past[0]
            dy = p_current[1] - p_past[1]
            dist_pixels = np.sqrt(dx**2 + dy**2)
            
            # Pixels per frame
            px_per_frame = dist_pixels / window
            
            # Estimated speed in km/h
            speed_kmh = px_per_frame * self.scale_factor
            return float(round(speed_kmh, 1))
