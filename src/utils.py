import cv2
import numpy as np

def draw_overlay(frame, tracker, current_light_state, speed_limit=50.0, show_trajectories=True):
    """
    Draws all computer vision overlays onto the video frame:
    - Stop lines (red when light is RED, green when light is GREEN).
    - Vehicle bounding boxes, IDs, and estimated speeds.
    - Historical trajectory tails.
    - Colored badges/text for active violations.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Configuration matches simulation
    road_y = 300
    lane_width = 80
    stop_line_east = 350
    stop_line_west = 450
    
    # 1. Draw stop lines with colors corresponding to traffic light states
    # Eastbound stop line (Left-to-Right)
    east_line_color = (0, 0, 255) if current_light_state == "RED" else ((0, 255, 255) if current_light_state == "YELLOW" else (0, 255, 0))
    cv2.line(frame, (stop_line_east, road_y), (stop_line_east, road_y + lane_width), east_line_color, 4)
    cv2.putText(frame, "STOP LINE", (stop_line_east - 80, road_y + lane_width - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, east_line_color, 1)
    
    # Westbound stop line (Right-to-Left)
    # Westbound light is opposite to Eastbound
    west_light_state = "GREEN" if current_light_state == "RED" else ("RED" if current_light_state == "GREEN" else "YELLOW")
    west_line_color = (0, 0, 255) if west_light_state == "RED" else ((0, 255, 255) if west_light_state == "YELLOW" else (0, 255, 0))
    cv2.line(frame, (stop_line_west, road_y - lane_width), (stop_line_west, road_y), west_line_color, 4)
    cv2.putText(frame, "STOP LINE", (stop_line_west + 10, road_y - lane_width + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, west_line_color, 1)

    # 2. Draw vehicles, trajectories, and violations
    for obj_id in tracker.objects.keys():
        if obj_id not in tracker.bboxes:
            continue
            
        bbox = tracker.bboxes[obj_id]
        class_id = tracker.class_ids[obj_id]
        trajectory = tracker.trajectories[obj_id]
        violations = tracker.violations.get(obj_id, set())
        
        # Determine vehicle label
        class_name = "Car" if class_id == 2 else ("Truck" if class_id == 7 else "Vehicle")
        
        # Get latest estimated speed
        speed = 0.0
        if obj_id in tracker.speeds and len(tracker.speeds[obj_id]) > 0:
            speed = tracker.speeds[obj_id][-1]
            
        # Select bounding box color
        # Red if there are any violations, yellow if warning (like speeding slightly), else light blue/cyan
        if len(violations) > 0:
            box_color = (0, 0, 255)  # Bright Red
        elif speed > speed_limit * 0.9:
            box_color = (0, 255, 255)  # Yellow warning
        else:
            box_color = (255, 255, 0)  # Cyan/Light Blue
            
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Draw trajectory tail
        if show_trajectories and len(trajectory) > 1:
            # Draw lines between recent points
            pts = np.array(trajectory[-25:], dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 0), 1)
            # Draw small circle on current centroid
            cv2.circle(frame, trajectory[-1], 3, box_color, -1)

        # Label above bounding box
        speed_label = f"{speed} km/h" if speed > 0 else "Estimating..."
        label = f"ID {obj_id}: {class_name} | {speed_label}"
        
        # Label background
        (w_label, h_label), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1 - h_label - 8), (x1 + w_label + 6, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw violation labels below bounding box
        if len(violations) > 0:
            y_offset = y2 + 15
            for v in violations:
                cv2.putText(frame, f"*{v}", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                y_offset += 13

    # Add transparency effect for the road overlays
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    return frame
