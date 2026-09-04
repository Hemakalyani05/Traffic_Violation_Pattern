import os
import cv2
import numpy as np
import random

def generate_traffic_video(output_path="data/traffic_demo.mp4", num_frames=1000):
    """
    Generates a synthetic 2D traffic video (800x600) with a horizontal road,
    traffic lights, and cars exhibiting normal and anomalous behaviors.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Video properties
    width, height = 800, 600
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Traffic light cycle properties
    # 0: Red, 1: Yellow, 2: Green
    light_cycle = [
        {"state": "RED", "duration": 180, "color": (0, 0, 255)},
        {"state": "GREEN", "duration": 180, "color": (0, 255, 0)},
        {"state": "YELLOW", "duration": 60, "color": (0, 255, 255)}
    ]
    
    # Active simulation vehicles
    vehicles = []
    # Unique ID tracker for generator
    next_vehicle_id = 1
    
    # Simulation configuration
    road_y = 300
    lane_width = 80
    east_lane_y = road_y + lane_width // 2  # Moving Left -> Right (y = 340)
    west_lane_y = road_y - lane_width // 2  # Moving Right -> Left (y = 260)
    
    stop_line_east = 350
    stop_line_west = 450
    
    # Speed limit in pixels/frame (5 px/frame = ~50 km/h)
    speed_limit = 5.0
    
    for frame_idx in range(num_frames):
        # Determine traffic light state
        cycle_time = frame_idx % (180 + 180 + 60)
        if cycle_time < 180:
            light_state = "RED"
            light_color = (0, 0, 255)
        elif cycle_time < 360:
            light_state = "GREEN"
            light_color = (0, 255, 0)
        else:
            light_state = "YELLOW"
            light_color = (0, 255, 255)
            
        # Draw background (Dark slate-gray road, green fields)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (34, 139, 34)  # Forest Green background
        
        # Draw road
        cv2.rectangle(frame, (0, road_y - lane_width), (width, road_y + lane_width), (50, 50, 50), -1)
        # Draw lane separator (dashed yellow line)
        for x in range(0, width, 40):
            cv2.line(frame, (x, road_y), (x + 20, road_y), (0, 255, 255), 2)
        # Draw road borders
        cv2.line(frame, (0, road_y - lane_width), (width, road_y - lane_width), (220, 220, 220), 2)
        cv2.line(frame, (0, road_y + lane_width), (width, road_y + lane_width), (220, 220, 220), 2)
        
        # Draw stop lines
        # Eastbound stop line (for cars moving left->right)
        cv2.line(frame, (stop_line_east, road_y), (stop_line_east, road_y + lane_width), (255, 255, 255), 4)
        # Westbound stop line (for cars moving right->left)
        cv2.line(frame, (stop_line_west, road_y - lane_width), (stop_line_west, road_y), (255, 255, 255), 4)
        
        # Draw Traffic Lights
        # Eastbound light (placed near stop_line_east)
        cv2.rectangle(frame, (stop_line_east + 10, road_y + lane_width + 10), (stop_line_east + 30, road_y + lane_width + 50), (20, 20, 20), -1)
        cv2.circle(frame, (stop_line_east + 20, road_y + lane_width + 20 if light_state == "RED" else (road_y + lane_width + 30 if light_state == "YELLOW" else road_y + lane_width + 40)), 5, light_color, -1)
        
        # Westbound light (placed near stop_line_west)
        cv2.rectangle(frame, (stop_line_west - 30, road_y - lane_width - 50), (stop_line_west - 10, road_y - lane_width - 10), (20, 20, 20), -1)
        # For simplicity, Westbound light runs green/yellow/red opposite to Eastbound light
        west_light_state = "GREEN" if light_state == "RED" else ("RED" if light_state == "GREEN" else "YELLOW")
        west_light_color = (0, 255, 0) if west_light_state == "GREEN" else ((0, 0, 255) if west_light_state == "RED" else (0, 255, 255))
        cv2.circle(frame, (stop_line_west - 20, road_y - lane_width - 40 if west_light_state == "RED" else (road_y - lane_width - 30 if west_light_state == "YELLOW" else road_y - lane_width - 20)), 5, west_light_color, -1)
        
        # Spawning new vehicles
        # Spawn Eastbound vehicles
        if frame_idx % 120 == 0:
            behavior = random.choice(["NORMAL", "NORMAL", "SPEEDING", "RED_LIGHT_RUNNER", "SWERVING", "TAILGATING"])
            # Ensure tailgater doesn't spawn in an empty lane
            if behavior == "TAILGATING" and not any(v["direction"] == 1 and v["x"] > 100 for v in vehicles):
                behavior = "NORMAL"
                
            y_pos = east_lane_y
            x_pos = -50
            speed = 4.2 + random.random() * 0.8
            color = (random.randint(100, 255), random.randint(50, 150), random.randint(50, 150)) # Distinct colors
            
            if behavior == "SPEEDING":
                speed = 7.5 + random.random() * 1.5
                color = (50, 50, 255) # Reddish-pink for speeder
            elif behavior == "RED_LIGHT_RUNNER":
                speed = 4.5
                color = (255, 100, 50)
            elif behavior == "SWERVING":
                color = (50, 255, 50) # Light Green
            elif behavior == "TAILGATING":
                color = (255, 255, 50) # Yellowish
                # Spawn close to the previous eastbound vehicle
                east_vehicles = [v for v in vehicles if v["direction"] == 1]
                if east_vehicles:
                    last_v = min(east_vehicles, key=lambda v: v["x"])
                    x_pos = last_v["x"] - 60  # close distance
                    speed = last_v["speed"]
            
            vehicles.append({
                "id": next_vehicle_id,
                "x": x_pos,
                "y": y_pos,
                "orig_y": y_pos,
                "speed": speed,
                "behavior": behavior,
                "color": color,
                "direction": 1,  # 1: Left->Right, -1: Right->Left
                "width": 40,
                "height": 20,
                "swerving_phase": random.random() * 10
            })
            next_vehicle_id += 1
            
        # Spawn Westbound vehicles
        if frame_idx % 150 == 45:
            behavior = random.choice(["NORMAL", "NORMAL", "WRONG_WAY"])
            y_pos = west_lane_y
            x_pos = width + 50
            speed = 4.0 + random.random() * 1.0
            color = (random.randint(100, 255), random.randint(150, 250), random.randint(50, 150))
            
            if behavior == "WRONG_WAY":
                # Spawns at x = -50 (wrong side) but moves right-to-left? Or spawns left and moves right in west lane.
                # Wrong way: moves Left->Right in the Westbound lane (against traffic).
                x_pos = -50
                direction = 1
                color = (0, 0, 255) # Red for danger
                speed = 4.0
            else:
                direction = -1
                
            vehicles.append({
                "id": next_vehicle_id,
                "x": x_pos,
                "y": y_pos,
                "orig_y": y_pos,
                "speed": speed,
                "behavior": behavior,
                "color": color,
                "direction": direction,
                "width": 40,
                "height": 20,
                "swerving_phase": random.random() * 10
            })
            next_vehicle_id += 1

        # Update and draw vehicles
        remaining_vehicles = []
        for v in vehicles:
            v_id = v["id"]
            x = v["x"]
            y = v["y"]
            speed = v["speed"]
            behavior = v["behavior"]
            direction = v["direction"]
            
            # Movement logic & Traffic light adherence
            # Stop if light is RED and vehicle is approaching its stop line
            should_stop = False
            
            if direction == 1 and y > road_y:  # Eastbound
                # Check traffic light at stop_line_east (350)
                if light_state == "RED" and x < stop_line_east and (x + v["width"]//2) >= stop_line_east - 40:
                    if behavior != "RED_LIGHT_RUNNER" and behavior != "SPEEDING":
                        should_stop = True
            elif direction == -1 and y < road_y:  # Westbound
                # Check traffic light at stop_line_west (450)
                if west_light_state == "RED" and x > stop_line_west and (x - v["width"]//2) <= stop_line_west + 40:
                    should_stop = True
            
            # Tailgating logic: adapt speed if too close to car in front
            if behavior == "TAILGATING" and not should_stop:
                front_cars = [oc for oc in vehicles if oc["direction"] == direction and oc["x"] > x and oc["id"] != v_id]
                if front_cars:
                    closest_front = min(front_cars, key=lambda oc: oc["x"] - x)
                    distance = closest_front["x"] - x - v["width"]
                    if distance < 30:
                        # Too close! Brake to match speed, but maintain tailgating behavior
                        speed = closest_front["speed"]
                    else:
                        # Try to catch up
                        speed = v["speed"] * 1.1
            
            if should_stop:
                # Decelerate smoothly
                dx = 0
            else:
                dx = speed * direction
            
            # Apply movement
            v["x"] += dx
            
            # Lateral movement for swerving behavior (sine wave oscillation)
            if behavior == "SWERVING":
                v["swerving_phase"] += 0.2
                v["y"] = v["orig_y"] + int(18 * np.sin(v["swerving_phase"]))
            
            # Draw vehicle body (colored rectangle)
            # Center coordinates
            cx, cy = int(v["x"]), int(v["y"])
            hw, hh = v["width"] // 2, v["height"] // 2
            
            # Check if vehicle is off screen
            if (direction == 1 and cx - hw > width + 100) or (direction == -1 and cx + hw < -100):
                continue
                
            # Draw wheels
            cv2.rectangle(frame, (cx - hw + 5, cy - hh - 3), (cx - hw + 12, cy - hh), (0, 0, 0), -1)
            cv2.rectangle(frame, (cx + hw - 12, cy - hh - 3), (cx + hw - 5, cy - hh), (0, 0, 0), -1)
            cv2.rectangle(frame, (cx - hw + 5, cy + hh), (cx - hw + 12, cy + hh + 3), (0, 0, 0), -1)
            cv2.rectangle(frame, (cx + hw - 12, cy + hh), (cx + hw - 5, cy + hh + 3), (0, 0, 0), -1)
            
            # Draw car body
            cv2.rectangle(frame, (cx - hw, cy - hh), (cx + hw, cy + hh), v["color"], -1)
            # Draw roof / cabin (black rectangle)
            cv2.rectangle(frame, (cx - hw + 10, cy - hh + 3), (cx + hw - 10, cy + hh - 3), (20, 20, 20), -1)
            # Draw windshield (light blue)
            windshield_color = (230, 230, 250)
            if direction == 1:
                cv2.rectangle(frame, (cx + hw - 12, cy - hh + 4), (cx + hw - 9, cy + hh - 4), windshield_color, -1)
            else:
                cv2.rectangle(frame, (cx - hw + 9, cy - hh + 4), (cx - hw + 12, cy + hh - 4), windshield_color, -1)
                
            # Draw headlights (yellow circles)
            if direction == 1:
                cv2.circle(frame, (cx + hw, cy - hh + 4), 2, (0, 255, 255), -1)
                cv2.circle(frame, (cx + hw, cy + hh - 4), 2, (0, 255, 255), -1)
            else:
                cv2.circle(frame, (cx - hw, cy - hh + 4), 2, (0, 255, 255), -1)
                cv2.circle(frame, (cx - hw, cy + hh - 4), 2, (0, 255, 255), -1)
            
            # Draw text showing ID and behavior for debugging / demo visualizer
            cv2.putText(frame, f"ID:{v_id}", (cx - hw, cy - hh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            remaining_vehicles.append(v)
            
        vehicles = remaining_vehicles
        
        # Write frame
        out.write(frame)
        
    out.release()
    print(f"Generated synthetic traffic video at: {output_path}")

if __name__ == "__main__":
    generate_traffic_video()
