import unittest
import sys
import os

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tracker import CentroidTracker

class TestCentroidTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = CentroidTracker(max_disappeared=5, max_distance=50)

    def test_registration(self):
        # Frame 1: One vehicle detected
        rects = [(100, 100, 140, 120, 1.0, 2)]  # x1, y1, x2, y2, conf, class_id
        active_bboxes = self.tracker.update(rects)
        
        self.assertEqual(len(active_bboxes), 1)
        self.assertIn(1, active_bboxes)
        self.assertEqual(active_bboxes[1], (100, 100, 140, 120))
        self.assertEqual(self.tracker.class_ids[1], 2)
        self.assertEqual(len(self.tracker.trajectories[1]), 1)

    def test_tracking_association(self):
        # Frame 1: Spawn vehicle
        rects = [(100, 100, 140, 120, 1.0, 2)]
        self.tracker.update(rects)
        
        # Frame 2: Vehicle moved slightly to the right
        rects = [(105, 100, 145, 120, 1.0, 2)]
        active_bboxes = self.tracker.update(rects)
        
        # Should keep ID 1
        self.assertEqual(len(active_bboxes), 1)
        self.assertIn(1, active_bboxes)
        self.assertEqual(active_bboxes[1], (105, 100, 145, 120))
        self.assertEqual(len(self.tracker.trajectories[1]), 2)

    def test_direction_determination(self):
        # Frame 1 to 5: Vehicle moves right
        for i in range(5):
            rects = [(100 + i * 5, 100, 140 + i * 5, 120, 1.0, 2)]
            self.tracker.update(rects)
            
        # Direction should be 1 (Left to Right / Eastbound)
        self.assertEqual(self.tracker.directions[1], 1)

    def test_deregistration(self):
        # Frame 1: Spawn vehicle
        rects = [(100, 100, 140, 120, 1.0, 2)]
        self.tracker.update(rects)
        
        # Frame 2 to 7: No detections (exceeds max_disappeared=5)
        for _ in range(6):
            self.tracker.update([])
            
        # Should be deregistered
        self.assertEqual(len(self.tracker.objects), 0)

if __name__ == '__main__':
    unittest.main()
