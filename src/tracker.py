import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=60):
        """
        Initializes the centroid tracker.
        max_disappeared: Number of frames a vehicle can go undetected before being deleted.
        max_distance: Maximum distance between centroids to associate them.
        """
        self.next_object_id = 1
        self.objects = {}       # id -> centroid (cx, cy)
        self.bboxes = {}        # id -> bbox (x1, y1, x2, y2)
        self.class_ids = {}     # id -> class_id
        self.trajectories = {}  # id -> list of (cx, cy)
        self.disappeared = {}   # id -> count of frames disappeared
        
        # Track statistics & violations
        self.speeds = {}        # id -> list of estimated speeds
        self.violations = {}    # id -> set of violations
        self.directions = {}    # id -> direction (1 for Left->Right, -1 for Right->Left)
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, class_id):
        """Registers a new vehicle."""
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.class_ids[obj_id] = class_id
        self.trajectories[obj_id] = [centroid]
        self.disappeared[obj_id] = 0
        self.speeds[obj_id] = []
        self.violations[obj_id] = set()
        self.directions[obj_id] = None  # to be determined as it moves
        
        self.next_object_id += 1
        return obj_id

    def deregister(self, obj_id):
        """Deregisters a vehicle and cleans up."""
        del self.objects[obj_id]
        del self.bboxes[obj_id]
        del self.class_ids[obj_id]
        # Keep trajectories, speeds, violations, directions for historical analytics in the dashboard
        # but stop tracking them actively
        del self.disappeared[obj_id]

    def update(self, rects):
        """
        Updates tracking with newly detected bounding boxes.
        rects: List of bounding boxes [(x1, y1, x2, y2, confidence, class_id)]
        Returns: Dict of active objects id -> bbox
        """
        # If no rects are detected, mark all existing objects as disappeared
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.bboxes

        # Compute centroids and parse class IDs for new detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_classes = []
        for i, (x1, y1, x2, y2, conf, class_id) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
            input_classes.append(class_id)

        # If we are currently not tracking any objects, register each input centroid
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], rects[i][:4], input_classes[i])
        else:
            # Grab the set of object IDs and their corresponding centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute the distance between each pair of object centroids and input centroids
            # Row index = existing object, Column index = new detection
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            # To perform the matching:
            # 1. Find the smallest value in each row
            # 2. Sort the row indexes based on their minimum values
            rows = D.min(axis=1).argsort()

            # 3. Find the smallest value in each column and sort
            cols = D.argmin(axis=1)[rows]

            # Keep track of which rows (existing) and columns (new) have been matched
            used_rows = set()
            used_cols = set()

            # Loop over the combination of the (row, column) index tuples
            for row, col in zip(rows, cols):
                # If we have already examined either the row or column, ignore it
                if row in used_rows or col in used_cols:
                    continue

                # If the distance is too large, do not associate them
                if D[row, col] > self.max_distance:
                    continue

                # Otherwise, update the existing object
                obj_id = object_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.bboxes[obj_id] = rects[col][:4]
                self.class_ids[obj_id] = input_classes[col]
                self.trajectories[obj_id].append(tuple(input_centroids[col]))
                self.disappeared[obj_id] = 0

                # Determine direction of movement (based on x displacement)
                traj = self.trajectories[obj_id]
                if len(traj) >= 5 and self.directions[obj_id] is None:
                    dx = traj[-1][0] - traj[0][0]
                    self.directions[obj_id] = 1 if dx > 0 else -1

                used_rows.add(row)
                used_cols.add(col)

            # Compute both the row and column indexes we have NOT yet examined
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # If the number of object centroids is greater than or equal to input centroids,
            # some existing objects have disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    obj_id = object_ids[row]
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)
            # Otherwise, register the new detections
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], rects[col][:4], input_classes[col])

        return self.bboxes
