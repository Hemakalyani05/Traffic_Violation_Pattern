import unittest
import sys
import os
import numpy as np

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.anomaly_detector import TrajectoryAnomalyDetector

class TestTrajectoryAnomalyDetector(unittest.TestCase):
    def setUp(self):
        # contamination level of 0.1
        self.detector = TrajectoryAnomalyDetector(num_points=30, contamination=0.1)

    def test_resampling(self):
        # A simple trajectory with 10 points
        traj = [(i * 20, 300) for i in range(10)]
        features = self.detector.resample_trajectory(traj)
        
        self.assertIsNotNone(features)
        # Should return 2 * num_points = 60 values
        self.assertEqual(len(features), 60)
        # The x coords (first 30 values) should start at 0.0 and end at 1.0
        self.assertAlmostEqual(features[0], 0.0)
        self.assertAlmostEqual(features[29], 1.0)
        # The y coords (last 30 values) should all be 300.0
        for y_val in features[30:]:
            self.assertAlmostEqual(y_val, 300.0)

    def test_fit_and_predict(self):
        # Generate 20 normal straight trajectories
        for _ in range(20):
            # Straight Eastbound paths at y=340 with minor random noise
            traj = []
            start_x = np.random.randint(-20, 20)
            y_base = 340 + np.random.randint(-5, 5)
            for step in range(40):
                x = start_x + step * 15
                y = y_base + int(np.random.normal(0, 1.5))
                traj.append((x, y))
            self.detector.add_to_training(traj)
            
        # Fit the model
        fitted = self.detector.fit()
        self.assertTrue(fitted)
        self.assertTrue(self.detector.is_fitted)
        
        # Test 1: Predict on a new normal trajectory
        normal_traj = [(i * 15, 340) for i in range(40)]
        is_anom, score = self.detector.predict(normal_traj)
        self.assertFalse(is_anom)
        self.assertGreater(score, -0.1) # normal score is close to 0 or positive
        
        # Test 2: Predict on an anomalous swerving trajectory
        # Severe swerving between y=300 and y=380
        anomalous_traj = []
        for i in range(40):
            x = i * 15
            y = 340 + int(25 * np.sin(i / 2.0)) # extreme weaving
            anomalous_traj.append((x, y))
            
        is_anom, score = self.detector.predict(anomalous_traj)
        self.assertTrue(is_anom)
        self.assertLess(score, 0.0) # anomaly score is negative

if __name__ == '__main__':
    unittest.main()
