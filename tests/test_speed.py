import unittest
import sys
import os

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.speed_estimator import SpeedEstimator

class TestSpeedEstimator(unittest.TestCase):
    def setUp(self):
        # 30 fps, default flat mode
        self.estimator_flat = SpeedEstimator(fps=30, mode="flat")
        
        # Homography mode
        # Calibration points in pixels: 4-point trapezoid representing a 10m x 40m area
        # Top-left, top-right, bottom-right, bottom-left
        points = [(300, 200), (500, 200), (600, 400), (200, 400)]
        self.estimator_homo = SpeedEstimator(fps=30, mode="homography", 
                                             calibration_points=points, 
                                             real_world_dims=(10, 40))

    def test_flat_speed_estimation(self):
        # Build a mock trajectory: moving 4 pixels per frame
        # Total displacement = 40 pixels over 10 frames
        trajectory = [(i * 4, 340) for i in range(11)]
        
        speed = self.estimator_flat.estimate_speed(trajectory, window=10)
        # Expected: 4 pixels/frame * scale_factor (12.0) = 48.0 km/h
        self.assertAlmostEqual(speed, 48.0)

    def test_insufficient_data(self):
        # Not enough points in trajectory (window is 10, trajectory has 5 points)
        trajectory = [(i * 4, 340) for i in range(5)]
        speed = self.estimator_flat.estimate_speed(trajectory, window=10)
        self.assertEqual(speed, 0.0)

    def test_homography_speed_estimation(self):
        # Check that homography matrix is initialized
        self.assertIsNotNone(self.estimator_homo.H)
        
        # Test coordinates mapping.
        # Trajectory in pixel coordinates that translates to real displacement.
        # Let's verify that a valid trajectory returns a positive velocity.
        trajectory = [(350, 200 + i * 15) for i in range(15)]
        speed = self.estimator_homo.estimate_speed(trajectory, window=10)
        self.assertGreater(speed, 0.0)

if __name__ == '__main__':
    unittest.main()
