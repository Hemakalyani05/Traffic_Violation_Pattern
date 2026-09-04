import numpy as np
from sklearn.ensemble import IsolationForest

class TrajectoryAnomalyDetector:
    def __init__(self, num_points=30, contamination=0.15):
        """
        Initializes the Trajectory Anomaly Detector.
        num_points: The fixed number of points to resample trajectories to.
        contamination: The proportion of outliers (anomalies) expected in the training data.
        """
        self.num_points = num_points
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination, random_state=42)
        self.is_fitted = False
        self.training_data = []

    def resample_trajectory(self, trajectory):
        """
        Resamples a trajectory of variable length to a fixed length using linear interpolation.
        trajectory: List of (x, y) coordinate tuples.
        Returns: Flat numpy array of shape (2 * num_points,)
        """
        if len(trajectory) < 2:
            return None
            
        traj_arr = np.array(trajectory)
        x_coords = traj_arr[:, 0]
        y_coords = traj_arr[:, 1]
        
        # Original steps
        old_indices = np.linspace(0, 1, len(trajectory))
        # Target steps
        new_indices = np.linspace(0, 1, self.num_points)
        
        # Interpolate
        x_resampled = np.interp(new_indices, old_indices, x_coords)
        y_resampled = np.interp(new_indices, old_indices, y_coords)
        
        # Normalize x to start at 0 (translation invariance along traffic direction)
        # to ensure we look at shape/direction rather than exact starting coordinate
        x_min = x_resampled.min()
        x_max = x_resampled.max()
        if x_max > x_min:
            # Maintain the direction (left-to-right vs right-to-left)
            # Normalizing to [0, 1] for left-to-right makes it 0 -> 1.
            # For right-to-left, if we normalize standard:
            # (x - x_min)/(x_max - x_min) will make it 1 -> 0 because x starts at x_max and ends at x_min.
            # This is perfect! The model learns that x starts at 1 and ends at 0 for one lane,
            # and starts at 0 and ends at 1 for another lane.
            x_norm = (x_resampled - x_min) / (x_max - x_min)
            # If the car was moving right-to-left, the normalized sequence is reversed
            if x_resampled[0] > x_resampled[-1]:
                x_norm = 1.0 - x_norm
        else:
            x_norm = np.zeros(self.num_points)
            
        # We keep y absolute to represent the lane position,
        # but smooth it to remove noise.
        # Concatenate normalized x and absolute y coordinates
        features = np.concatenate([x_norm, y_resampled])
        return features

    def add_to_training(self, trajectory):
        """Adds a trajectory to the training buffer."""
        features = self.resample_trajectory(trajectory)
        if features is not None:
            self.training_data.append(features)

    def fit(self):
        """Fits the Isolation Forest on the accumulated training trajectories."""
        if len(self.training_data) < 10:
            print(f"Warning: Not enough trajectories to fit. Got {len(self.training_data)}, need at least 10.")
            return False
            
        X = np.array(self.training_data)
        self.model.fit(X)
        self.is_fitted = True
        print(f"Anomaly detector successfully trained on {len(self.training_data)} trajectories.")
        return True

    def predict(self, trajectory):
        """
        Predicts if a trajectory is anomalous.
        Returns:
            is_anomaly (bool): True if anomalous, False if normal.
            score (float): Anomaly score (lower means more anomalous, usually < 0.0 for anomalies).
        """
        if not self.is_fitted:
            # If not fitted, everything is classified as normal (False) with 0.0 score
            return False, 0.0
            
        features = self.resample_trajectory(trajectory)
        if features is None:
            return False, 0.0
            
        features = features.reshape(1, -1)
        
        # decision_function returns scores. Lower means more anomalous.
        # Isolation Forest score: negative values are anomalies, positive are normal.
        score = self.model.decision_function(features)[0]
        prediction = self.model.predict(features)[0]
        
        is_anomaly = (prediction == -1)
        
        return is_anomaly, float(score)
