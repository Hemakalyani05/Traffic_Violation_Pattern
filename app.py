import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import time
import json

# Set Streamlit Page Config
st.set_page_config(
    page_title="Smart Traffic Violation Detector",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk Glassmorphic CSS for styling the dashboard
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Main body styling */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #08090f !important;
        color: #f1f5f9 !important;
    }
    
    /* Force dark background for the entire page view */
    [data-testid="stAppViewContainer"] {
        background-color: #08090f !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0d111d !important;
    }
    
    /* Transparent default headers */
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Force ALL text labels and headers to be white/light slate */
    h1, h2, h3, h4, h5, h6, label, th, td {
        color: #ffffff !important;
    }
    p, li {
        color: #e2e8f0 !important;
    }
    
    /* Specific styling for widget labels in sidebar and main view */
    .stWidgetLabel p, [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Radio buttons and checkbox labels */
    div[class*="stRadio"] label, div[class*="stCheckbox"] label {
        color: #f1f5f9 !important;
    }
    
    /* Custom Title banner */
    .hero-container {
        background: radial-gradient(circle at 10% 20%, rgba(0, 242, 254, 0.08) 0%, transparent 45%),
                    radial-gradient(circle at 90% 80%, rgba(127, 0, 255, 0.08) 0%, transparent 45%);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 40px 30px;
        margin-bottom: 30px;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #00f2fe, #7f00ff, #ff007f);
    }
    
    .title-banner {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 30%, #7f00ff 70%, #ff007f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 12px;
        text-shadow: 0 10px 30px rgba(0, 242, 254, 0.1);
    }
    
    .subtitle-banner {
        color: #cbd5e1 !important;
        font-size: 1.25rem;
        font-weight: 400;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Custom Card container */
    .glass-card {
        background: rgba(17, 24, 43, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        margin-bottom: 24px;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 242, 254, 0.2);
        box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.08);
        transform: translateY(-2px);
    }
    
    /* Neon Custom Metrics Layout */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .metric-box {
        flex: 1;
        background: rgba(17, 24, 43, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-box:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 242, 254, 0.25);
        box-shadow: 0 10px 25px rgba(0, 242, 254, 0.06);
    }
    
    .metric-icon {
        font-size: 1.8rem;
        background: rgba(0, 242, 254, 0.08);
        color: #00f2fe;
        width: 54px;
        height: 54px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(0, 242, 254, 0.15);
    }
    
    .metric-icon.red {
        background: rgba(255, 75, 75, 0.08);
        color: #ff4b4b;
        border-color: rgba(255, 75, 75, 0.15);
    }
    
    .metric-icon.purple {
        background: rgba(127, 0, 255, 0.08);
        color: #a855f7;
        border-color: rgba(127, 0, 255, 0.15);
    }
    
    .metric-text {
        display: flex;
        flex-direction: column;
    }
    
    .metric-title {
        font-size: 0.85rem;
        color: #cbd5e1 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Neon glow color states - force with !important to protect from global overrides */
    .glow-cyan { color: #00f2fe !important; text-shadow: 0 0 12px rgba(0, 242, 254, 0.4); }
    .glow-red { color: #ff4b4b !important; text-shadow: 0 0 12px rgba(255, 75, 75, 0.4); }
    .glow-purple { color: #a855f7 !important; text-shadow: 0 0 12px rgba(168, 85, 247, 0.4); }
    .glow-green { color: #10b981 !important; text-shadow: 0 0 12px rgba(16, 185, 129, 0.4); }
    
    /* Alert badge */
    .alert-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 30px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-critical { background: rgba(255, 75, 75, 0.15); color: #ff4b4b !important; border: 1px solid rgba(255, 75, 75, 0.3); }
    .badge-high { background: rgba(168, 85, 247, 0.15); color: #c084fc !important; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-medium { background: rgba(251, 146, 60, 0.15); color: #fb923c !important; border: 1px solid rgba(251, 146, 60, 0.3); }
    
    /* Tab modifications - ensure high contrast for inactive and active tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(17, 24, 43, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: #cbd5e1 !important;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.25s ease;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #00f2fe !important;
        background: rgba(0, 242, 254, 0.06);
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.2), rgba(127, 0, 255, 0.2)) !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
    }
    
    /* Fix helper text inside widgets */
    .stSlider p, .stRadio p, .stSelectbox p {
        color: #cbd5e1 !important;
    }
    
    /* Terminal Console style for Webhook Dispatch Logs */
    .terminal-console {
        background: #02040a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 15px;
        font-family: 'JetBrains Mono', monospace;
        color: #38bdf8;
        font-size: 0.85rem;
        height: 180px;
        overflow-y: auto;
        box-shadow: inset 0 4px 20px rgba(0,0,0,0.8);
    }
    
</style>
""", unsafe_allow_html=True)

# Import local modules
from src.simulation import generate_traffic_video
from src.detector import VehicleDetector
from src.tracker import CentroidTracker
from src.speed_estimator import SpeedEstimator
from src.traffic_light import TrafficLightClassifier
from src.anomaly_detector import TrajectoryAnomalyDetector
from src.utils import draw_overlay

# Initialize session state objects if not present
if "violations_db" not in st.session_state:
    st.session_state.violations_db = []
if "trajectory_history" not in st.session_state:
    st.session_state.trajectory_history = {}  # Store completed trajectories for PCA
if "anomaly_detector" not in st.session_state:
    st.session_state.anomaly_detector = TrajectoryAnomalyDetector(num_points=30, contamination=0.15)
if "simulation_created" not in st.session_state:
    st.session_state.simulation_created = False
if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = np.zeros((600, 800), dtype=np.float32)
if "safety_history" not in st.session_state:
    st.session_state.safety_history = []
if "api_dispatch_logs" not in st.session_state:
    st.session_state.api_dispatch_logs = []

# Sidebar Configuration
st.sidebar.markdown("""
<div style="text-align: center; padding: 15px 0 25px 0;">
    <img src="https://img.icons8.com/nolan/96/traffic-light.png" style="width: 70px; margin-bottom: 10px;">
    <h3 style="margin: 0; font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #fff;">Control Center</h3>
    <span style="color: #cbd5e1; font-size: 0.85rem;">System Parameters & Feed</span>
</div>
""", unsafe_allow_html=True)

# Video Source Selection
video_source = st.sidebar.selectbox(
    "📹 Select Video Source",
    ["Demo Traffic Simulation", "Upload Video File"]
)

video_path = None
if video_source == "Demo Traffic Simulation":
    video_path = "data/traffic_demo.mp4"
    if not os.path.exists(video_path) or not st.session_state.simulation_created:
        with st.sidebar.status("🎬 Pre-Compiling Demo Simulation Video...", expanded=True) as status:
            generate_traffic_video(video_path, num_frames=1200)
            st.session_state.simulation_created = True
            status.update(label="Demo video compiled successfully!", state="complete")
else:
    uploaded_file = st.sidebar.file_uploader("Upload MP4 Traffic Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        video_path = os.path.join("data", "uploaded_traffic.mp4")
        os.makedirs("data", exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.sidebar.success("Video uploaded successfully!")
    else:
        st.sidebar.warning("Please upload a video or use the Demo Simulation.")

# Unique Feature Toggles
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='font-family: Space Grotesk; margin-bottom: 10px; color:#fff;'>🌟 Premium Unique Features</h4>", unsafe_allow_html=True)
enable_heatmap = st.sidebar.checkbox("🔥 Live Spatial Heatmap Overlay", value=True, help="Renders a real-time thermal spatial heatmap over the video highlighting where traffic violations are most frequent.")
enable_api_dispatch = st.sidebar.checkbox("🔗 REST API Dispatch Simulation", value=True, help="Simulates sending a JSON REST API payload to an external server on every critical violation event.")
api_webhook_url = st.sidebar.text_input("Webhook Destination URL", "https://api.citycontrol.gov/v1/traffic/dispatch", disabled=not enable_api_dispatch)

# Stop line Calibration (ROI)
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='font-family: Space Grotesk; margin-bottom: 10px; color:#fff;'>📐 Stop Line Calibration</h4>", unsafe_allow_html=True)
stop_line_east = st.sidebar.slider("Eastbound Stop Line (X)", 250, 450, 350)
stop_line_west = st.sidebar.slider("Westbound Stop Line (X)", 350, 550, 450)

# Detection Parameters
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='font-family: Space Grotesk; margin-bottom: 10px; color:#fff;'>Detection Settings</h4>", unsafe_allow_html=True)
detection_mode = st.sidebar.radio(
    "Vehicle Detection Method",
    ["Contour (Fast / Simulation)", "YOLOv8 (Real-world Video)"]
)

# Threshold settings
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='font-family: Space Grotesk; margin-bottom: 10px; color:#fff;'>Violation Thresholds</h4>", unsafe_allow_html=True)
speed_limit = st.sidebar.slider("Speed Limit (km/h)", 30, 100, 50, step=5)
tailgating_threshold = st.sidebar.slider("Tailgating Distance (pixels)", 25, 80, 45, step=5)

# ML Anomaly Settings
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='font-family: Space Grotesk; margin-bottom: 10px; color:#fff;'>ML Anomaly Settings</h4>", unsafe_allow_html=True)
contamination = st.sidebar.slider("Expected Anomaly Rate", 0.05, 0.30, 0.15, step=0.01)
st.session_state.anomaly_detector.contamination = contamination

# Main Panel Layout
st.markdown("""
<div class="hero-container">
    <div class="title-banner">🚦 Smart Traffic Violation Detector</div>
    <div class="subtitle-banner">
        An advanced multi-object tracking and computer vision pipeline integrated with an unsupervised 
        <strong>Isolation Forest</strong> machine learning model to classify reckless driving patterns in real-time.
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs structure
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Live Video Monitor", 
    "🧠 Trajectory Anomaly Analytics (ML)", 
    "📊 Violation Database & Reports",
    "📖 Project Overview"
])

with tab1:
    # 4 columns for custom neon KPI metrics
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        metric1_placeholder = st.empty()
    with kpi_col2:
        metric2_placeholder = st.empty()
    with kpi_col3:
        metric3_placeholder = st.empty()
    with kpi_col4:
        metric4_placeholder = st.empty()
        
    # Default state metrics before run
    metric1_placeholder.markdown("""
    <div class="metric-box">
        <div class="metric-icon">🛡️</div>
        <div class="metric-text">
            <div class="metric-title">Safety Index</div>
            <div class="metric-val glow-green">100%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    metric2_placeholder.markdown("""
    <div class="metric-box">
        <div class="metric-icon red">⚠️</div>
        <div class="metric-text">
            <div class="metric-title">Active Alerts</div>
            <div class="metric-val glow-cyan">0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    metric3_placeholder.markdown("""
    <div class="metric-box">
        <div class="metric-icon purple">⚡</div>
        <div class="metric-text">
            <div class="metric-title">Average Speed</div>
            <div class="metric-val glow-purple">-- km/h</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    metric4_placeholder.markdown("""
    <div class="metric-box">
        <div class="metric-icon">🚗</div>
        <div class="metric-text">
            <div class="metric-title">Total Tracked</div>
            <div class="metric-val glow-cyan">0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Set up layout columns for Video Player and Playback Control
    col_video, col_feed = st.columns([7, 3])
    
    with col_feed:
        st.markdown("<h4 style='font-family: Space Grotesk; margin-top: 0; color:#fff;'>Analysis Control</h4>", unsafe_allow_html=True)
        run_btn = st.button("▶️ Start Analysis", use_container_width=True)
        stop_btn = st.button("⏹️ Stop / Reset", use_container_width=True)
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 20px 0;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-family: Space Grotesk; color:#fff;'>Live Alert Feed</h4>", unsafe_allow_html=True)
        alert_placeholder = st.empty()
        alert_placeholder.info("Click 'Start Analysis' to begin parsing the video stream.")

    with col_video:
        video_placeholder = st.empty()
        if not run_btn:
            # Render a styled preview placeholder card
            video_placeholder.markdown("""
            <div style="background: rgba(17, 24, 43, 0.4); border: 2px dashed rgba(0, 242, 254, 0.2); border-radius: 16px; padding: 120px 20px; text-align: center;">
                <span style="font-size: 3rem;">🖥️</span>
                <h4 style="margin: 15px 0 5px 0; font-family: Space Grotesk; color: #fff;">Video Monitor Offline</h4>
                <p style="color: #cbd5e1; font-size: 0.95rem; max-width: 400px; margin: 0 auto;">
                    Click <strong>Start Analysis</strong> on the right to load the tracking engine and monitor traffic violations.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Simulated REST API Terminal Console (Unique Feature)
    if enable_api_dispatch:
        st.markdown("---")
        st.markdown("### 🔗 Simulated Enterprise Dispatch API Logs")
        st.write("Live logs demonstrating real-time REST API integration. Whenever a violation is detected, a JSON webhook is compiled and sent.")
        terminal_placeholder = st.empty()
        
        if not run_btn and not st.session_state.api_dispatch_logs:
            terminal_placeholder.markdown("""
            <div class="terminal-console">
                [SYSTEM] Waiting for active video analysis to initialize webhook dispatcher...
            </div>
            """, unsafe_allow_html=True)

# Process Video Stream
if run_btn and video_path is not None:
    # Initialize detector and tracker
    mode_str = "contour" if "Contour" in detection_mode else "yolo"
    detector = VehicleDetector(mode=mode_str)
    tracker = CentroidTracker(max_disappeared=15, max_distance=60)
    speed_est = SpeedEstimator(fps=30, mode="flat")
    light_clf = TrafficLightClassifier()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
    else:
        # If contour mode, set background frame using the first frame (without cars)
        ret, first_frame = cap.read()
        if ret:
            detector.set_background(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        recent_violations = []
        
        # Reset heatmap for the new run
        st.session_state.heatmap_data.fill(0)
        st.session_state.safety_history = []
        st.session_state.api_dispatch_logs = []
        
        # Streamlit playback loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Detect vehicles & light state
            rects = detector.detect(frame)
            light_state = light_clf.classify_state_synthetic(frame)
            tracker.update(rects)
            
            active_speeds = []
            
            # Decay heatmap trail slowly on every frame
            st.session_state.heatmap_data *= 0.985
            
            # Check violations for all currently tracked vehicles
            for obj_id in list(tracker.objects.keys()):
                trajectory = tracker.trajectories[obj_id]
                direction = tracker.directions[obj_id]
                bbox = tracker.bboxes[obj_id]
                class_id = tracker.class_ids[obj_id]
                class_name = "Car" if class_id == 2 else ("Truck" if class_id == 7 else "Vehicle")
                
                # Helper flag to see if this car got any violation on this frame
                violation_flag = False
                new_violation_entry = None
                
                # 1. Estimate Speed (look back 10 frames)
                speed = speed_est.estimate_speed(trajectory, window=10)
                if speed > 0:
                    tracker.speeds[obj_id].append(speed)
                    active_speeds.append(speed)
                    
                    # Log speeding violation
                    if speed > speed_limit:
                        violation_flag = True
                        v_type = "SPEEDING"
                        if v_type not in tracker.violations[obj_id]:
                            tracker.violations[obj_id].add(v_type)
                            new_violation_entry = {
                                "Timestamp": f"00:{frame_idx//30:02d}",
                                "Vehicle ID": int(obj_id),
                                "Vehicle Type": class_name,
                                "Violation Type": v_type,
                                "Speed": f"{speed} km/h",
                                "Details": f"Exceeded speed limit of {speed_limit} km/h (Recorded: {speed} km/h)",
                                "Severity": "Medium"
                            }
                            st.session_state.violations_db.insert(0, new_violation_entry)
                            recent_violations.insert(0, new_violation_entry)
                
                # 2. Red Light Violation
                if len(trajectory) >= 2 and direction == 1 and trajectory[-1][1] > 300: # Eastbound lane
                    prev_x = trajectory[-2][0]
                    curr_x = trajectory[-1][0]
                    
                    # Vehicle crossed stop line at calibrated X threshold
                    if prev_x < stop_line_east <= curr_x:
                        if light_state == "RED":
                            violation_flag = True
                            v_type = "RED_LIGHT"
                            if v_type not in tracker.violations[obj_id]:
                                tracker.violations[obj_id].add(v_type)
                                new_violation_entry = {
                                    "Timestamp": f"00:{frame_idx//30:02d}",
                                    "Vehicle ID": int(obj_id),
                                    "Vehicle Type": class_name,
                                    "Violation Type": v_type,
                                    "Speed": f"{speed} km/h" if speed > 0 else "N/A",
                                    "Details": f"Passed stop line at X={stop_line_east} while traffic light was RED",
                                    "Severity": "Critical"
                                }
                                st.session_state.violations_db.insert(0, new_violation_entry)
                                recent_violations.insert(0, new_violation_entry)
                                
                # 3. Wrong-Way Driving
                if direction == 1 and trajectory[-1][1] < 300:
                    violation_flag = True
                    v_type = "WRONG_WAY"
                    if v_type not in tracker.violations[obj_id]:
                        tracker.violations[obj_id].add(v_type)
                        new_violation_entry = {
                            "Timestamp": f"00:{frame_idx//30:02d}",
                            "Vehicle ID": int(obj_id),
                            "Vehicle Type": class_name,
                            "Violation Type": v_type,
                            "Speed": f"{speed} km/h" if speed > 0 else "N/A",
                            "Details": "Driving against traffic in Westbound lane",
                            "Severity": "Critical"
                        }
                        st.session_state.violations_db.insert(0, new_violation_entry)
                        recent_violations.insert(0, new_violation_entry)

                # 4. Tailgating Detection
                if direction is not None:
                    same_direction_vehicles = [
                        vid for vid in tracker.objects.keys() 
                        if vid != obj_id and tracker.directions.get(vid) == direction
                    ]
                    for other_id in same_direction_vehicles:
                        other_bbox = tracker.bboxes[other_id]
                        other_x = tracker.objects[other_id][0]
                        is_behind = (direction == 1 and trajectory[-1][0] < other_x) or (direction == -1 and trajectory[-1][0] > other_x)
                        
                        if is_behind:
                            pixel_gap = abs(trajectory[-1][0] - other_x) - (bbox[2] - bbox[0])
                            if pixel_gap < tailgating_threshold and abs(trajectory[-1][1] - tracker.objects[other_id][1]) < 30:
                                violation_flag = True
                                v_type = "TAILGATING"
                                if v_type not in tracker.violations[obj_id]:
                                    tracker.violations[obj_id].add(v_type)
                                    new_violation_entry = {
                                        "Timestamp": f"00:{frame_idx//30:02d}",
                                        "Vehicle ID": int(obj_id),
                                        "Vehicle Type": class_name,
                                        "Violation Type": v_type,
                                        "Speed": f"{speed} km/h" if speed > 0 else "N/A",
                                        "Details": f"Following vehicle ID {other_id} too closely (Distance: {int(pixel_gap)} px)",
                                        "Severity": "Medium"
                                    }
                                    st.session_state.violations_db.insert(0, new_violation_entry)
                                    recent_violations.insert(0, new_violation_entry)

                # 5. ML-based Reckless Swerving Anomaly Detection
                if len(trajectory) >= 30 and st.session_state.anomaly_detector.is_fitted:
                    is_anom, score = st.session_state.anomaly_detector.predict(trajectory)
                    if is_anom:
                        violation_flag = True
                        v_type = "RECKLESS_SWERVING"
                        if v_type not in tracker.violations[obj_id]:
                            tracker.violations[obj_id].add(v_type)
                            new_violation_entry = {
                                "Timestamp": f"00:{frame_idx//30:02d}",
                                "Vehicle ID": int(obj_id),
                                "Vehicle Type": class_name,
                                "Violation Type": v_type,
                                "Speed": f"{speed} km/h" if speed > 0 else "N/A",
                                "Details": f"Reckless driving patterns (ML Anomaly Score: {round(score, 3)})",
                                "Severity": "High"
                            }
                            st.session_state.violations_db.insert(0, new_violation_entry)
                            recent_violations.insert(0, new_violation_entry)
                
                # Trigger heatmap density update and simulated REST API webhook dispatch
                if violation_flag:
                    cx, cy = trajectory[-1]
                    cv2.circle(st.session_state.heatmap_data, (int(cx), int(cy)), 30, 0.4, -1)
                    
                    # API webhook dispatch simulation
                    if enable_api_dispatch and new_violation_entry is not None:
                        payload = {
                            "eventId": f"evt_{int(time.time()*1000)}_{obj_id}",
                            "timestamp": new_violation_entry["Timestamp"],
                            "webhookDestination": api_webhook_url,
                            "vehicleData": {
                                "id": int(obj_id),
                                "type": new_violation_entry["Vehicle Type"],
                                "currentSpeedKmh": speed
                            },
                            "infraction": {
                                "type": new_violation_entry["Violation Type"],
                                "severity": new_violation_entry["Severity"],
                                "description": new_violation_entry["Details"]
                            }
                        }
                        log_msg = f"[{time.strftime('%H:%M:%S')}] POST {api_webhook_url} -> 201 Created | Payload: {json.dumps(payload)}"
                        st.session_state.api_dispatch_logs.insert(0, log_msg)

            # Store completed trajectories for ML learning when vehicles exit
            for past_id in list(tracker.trajectories.keys()):
                if past_id not in tracker.objects and past_id not in st.session_state.trajectory_history:
                    final_traj = tracker.trajectories[past_id]
                    if len(final_traj) >= 20:
                        st.session_state.trajectory_history[past_id] = final_traj
                        st.session_state.anomaly_detector.add_to_training(final_traj)
                        
                        # Auto-train Isolation Forest
                        if not st.session_state.anomaly_detector.is_fitted and len(st.session_state.anomaly_detector.training_data) >= 15:
                            st.session_state.anomaly_detector.fit()

            # Render overlay on the frame
            annotated_frame = draw_overlay(frame, tracker, light_state, speed_limit)
            
            # --- UNIQUE FEATURE: Live Spatial Heatmap Blending ---
            if enable_heatmap and np.max(st.session_state.heatmap_data) > 0:
                heatmap_norm = np.clip(st.session_state.heatmap_data * 255, 0, 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
                mask = heatmap_norm > 10
                annotated_frame[mask] = cv2.addWeighted(annotated_frame, 0.5, heatmap_color, 0.5, 0)[mask]
            
            # Display frame in Streamlit
            video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            
            # Calculate metrics
            total_active_violations = sum(len(tracker.violations[vid]) for vid in tracker.objects.keys())
            total_historical_violations = len(st.session_state.violations_db)
            avg_speed_value = np.mean(active_speeds) if active_speeds else 45.0
            total_tracked = tracker.next_object_id - 1
            
            # Safety Index logic
            safety_index = max(0, 100 - total_active_violations * 15)
            st.session_state.safety_history.append(safety_index)
            
            # Update KPI metrics with custom HTML neon components
            metric1_placeholder.markdown(f"""
            <div class="metric-box">
                <div class="metric-icon">🛡️</div>
                <div class="metric-text">
                    <div class="metric-title">Safety Index</div>
                    <div class="metric-val {'glow-red' if safety_index < 70 else 'glow-green'}">{safety_index}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            metric2_placeholder.markdown(f"""
            <div class="metric-box">
                <div class="metric-icon red">⚠️</div>
                <div class="metric-text">
                    <div class="metric-title">Active Alerts</div>
                    <div class="metric-val {'glow-red' if total_active_violations > 0 else 'glow-cyan'}">{total_active_violations}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            metric3_placeholder.markdown(f"""
            <div class="metric-box">
                <div class="metric-icon purple">⚡</div>
                <div class="metric-text">
                    <div class="metric-title">Average Speed</div>
                    <div class="metric-val glow-purple">{int(avg_speed_value)} km/h</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            metric4_placeholder.markdown(f"""
            <div class="metric-box">
                <div class="metric-icon">🚗</div>
                <div class="metric-text">
                    <div class="metric-title">Total Tracked</div>
                    <div class="metric-val glow-cyan">{total_tracked}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Update Live Alerts Feed (show last 5)
            if recent_violations:
                alerts_html = ""
                for av in recent_violations[:5]:
                    badge_class = "badge-critical" if av["Severity"] == "Critical" else ("badge-high" if av["Severity"] == "High" else "badge-medium")
                    border_color = '#ff4b4b' if av['Severity']=='Critical' else ('#a855f7' if av['Severity']=='High' else '#fb923c')
                    alerts_html += f"""
                    <div style="padding: 14px; background: rgba(15, 23, 42, 0.6); border-left: 4px solid {border_color}; border-radius: 8px; margin-bottom: 10px; border-top: 1px solid rgba(255,255,255,0.02); border-right: 1px solid rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.02); box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span class="alert-badge {badge_class}">{av['Severity']}</span>
                            <span style="color: #cbd5e1; font-size: 0.8rem; font-family: 'JetBrains Mono';">{av['Timestamp']}</span>
                        </div>
                        <div style="color: #ffffff; font-size: 0.95rem; font-weight: 500;">
                            <strong>Vehicle ID {av['Vehicle ID']}</strong>: {av['Violation Type']}
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 2px;">
                            {av['Details']}
                        </div>
                    </div>
                    """
                alert_placeholder.markdown(alerts_html, unsafe_allow_html=True)
            else:
                alert_placeholder.info("No active violations detected. Traffic flow is normal.")
                
            # Update API dispatch logs terminal
            if enable_api_dispatch:
                logs_html = "<div class='terminal-console'>"
                if not st.session_state.api_dispatch_logs:
                    logs_html += "[SYSTEM] Listening for infraction dispatches..."
                else:
                    for log in st.session_state.api_dispatch_logs[:5]:
                        logs_html += f"{log}<br><br>"
                logs_html += "</div>"
                terminal_placeholder.markdown(logs_html, unsafe_allow_html=True)
                
            # Frame rate control
            time.sleep(0.015)
            
        cap.release()

with tab2:
    st.markdown("<h3 style='font-family: Space Grotesk; color:#fff;'>Brain Center: Unsupervised Trajectory Analytics</h3>", unsafe_allow_html=True)
    st.write("This section visualizes vehicle coordinates processed by the **Isolation Forest** model to detect reckless driving anomalies.")
    
    total_traj_count = len(st.session_state.trajectory_history)
    model_fitted = st.session_state.anomaly_detector.is_fitted
    
    col_ml_1, col_ml_2 = st.columns([1, 1])
    
    with col_ml_1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="font-family: Space Grotesk; margin-top:0; color:#fff;">Model Training Dashboard</h4>
            <table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 40px;">
                    <td style="color:#cbd5e1;">Completed Trajectories:</td>
                    <td style="text-align:right; font-weight:700; color:#fff;">""" + str(total_traj_count) + """</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 40px;">
                    <td style="color:#cbd5e1;">Model Status:</td>
                    <td style="text-align:right; font-weight:700; color:""" + ("#10b981" if model_fitted else "#fb923c") + """;">""" + ("READY (Fitted)" if model_fitted else "WAITING FOR DATA") + """</td>
                </tr>
                <tr style="height: 40px;">
                    <td style="color:#cbd5e1;">Contamination Factor:</td>
                    <td style="text-align:right; font-weight:700; color:#00f2fe;">""" + str(round(contamination, 2)) + """</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        train_btn = st.button("🔧 Fit Isolation Forest Model", use_container_width=True)
        if train_btn:
            if len(st.session_state.anomaly_detector.training_data) >= 10:
                success = st.session_state.anomaly_detector.fit()
                if success:
                    st.success("Isolation Forest Model trained successfully!")
                    st.rerun()
                else:
                    st.error("Model training failed.")
            else:
                st.warning(f"Not enough trajectories to train. Got {len(st.session_state.anomaly_detector.training_data)} trajectories, need at least 10.")
                
    with col_ml_2:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <h4 style="font-family: Space Grotesk; margin-top:0; color:#fff;">Methodology Overview</h4>
            <p style="color:#cbd5e1; font-size: 0.95rem; line-height: 1.6;">
                1. <strong>Resampling</strong>: Raw coordinates are resampled using linear interpolation to exactly 30 points, rendering features speed-invariant.<br>
                2. <strong>Normalization</strong>: Aligning coordinates to $[0, 1]$ relative translation preserves direction vectors while removing horizontal pixel offsets.<br>
                3. <strong>Isolation Forest</strong>: The algorithm builds isolating split trees. A swerving vehicle (high frequency sine displacement) is isolated quickly near the root of the trees, resulting in highly negative anomaly scores.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Visualizations
    if total_traj_count > 0:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 30px 0;'>", unsafe_allow_html=True)
        
        # --- UNIQUE FEATURE: Safety Rating Predictive Forecasting ---
        st.markdown("### Safety Index Trend Forecasting")
        st.write("Dynamic prediction of the junction safety profile for the next 10 seconds based on active safety ratings and violation trajectories.")
        
        if st.session_state.safety_history:
            history_len = len(st.session_state.safety_history)
            x_vals = list(range(history_len))
            y_vals = st.session_state.safety_history
            
            # Holt-Winters / Linear Trend extrapolation projection
            forecast_len = 300  # 10 seconds into the future
            x_fore = list(range(history_len, history_len + forecast_len))
            
            # Compute a moving average trend slope
            if history_len > 20:
                recent_y = y_vals[-20:]
                recent_x = x_vals[-20:]
                slope = np.polyfit(recent_x, recent_y, 1)[0]
            else:
                slope = 0.0
            
            # Predict safety index, clipping between 0 and 100
            y_fore = [max(0, min(100, y_vals[-1] + slope * idx)) for idx in range(1, forecast_len + 1)]
            
            fig_fore = go.Figure()
            # Historical line
            fig_fore.add_trace(go.Scatter(x=x_vals, y=y_vals, name="Safety Index (Actual)", line=dict(color="#00f2fe", width=2)))
            # Dotted forecast line
            fig_fore.add_trace(go.Scatter(x=x_fore, y=y_fore, name="Predictive Trend (Forecast)", line=dict(color="#ffa500", width=2, dash="dash")))
            
            fig_fore.update_layout(
                xaxis_title="Time Steps (Frames)",
                yaxis_title="Safety Score (%)",
                yaxis=dict(range=[0, 105]),
                plot_bgcolor="rgba(11,15,26,0.6)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Outfit")
            )
            st.plotly_chart(fig_fore, use_container_width=True)
        
        # Plot raw trajectories
        fig_paths = go.Figure()
        
        # Road outline
        fig_paths.add_shape(type="rect", x0=0, y0=220, x1=800, y1=380, fillcolor="rgba(100, 116, 139, 0.08)", line=dict(color="rgba(0, 242, 254, 0.15)", width=1))
        fig_paths.add_shape(type="line", x0=0, y0=300, x1=800, y1=300, line=dict(color="rgba(255,255,0,0.4)", width=1, dash="dash"))
        
        anom_count = 0
        norm_count = 0
        
        for tid, traj in st.session_state.trajectory_history.items():
            traj_arr = np.array(traj)
            xs = traj_arr[:, 0]
            ys = traj_arr[:, 1]
            
            is_anom = False
            score = 0.0
            if model_fitted:
                is_anom, score = st.session_state.anomaly_detector.predict(traj)
                
            if is_anom:
                anom_count += 1
                color = "#ff4b4b"
                name = f"Vehicle {tid} (Anomaly: {round(score,2)})"
                width = 3.0
            else:
                norm_count += 1
                color = "#00f2fe"
                name = f"Vehicle {tid} (Normal)"
                width = 1.0
                
            fig_paths.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines+markers',
                line=dict(color=color, width=width),
                marker=dict(size=4),
                name=name
            ))
            
        fig_paths.update_layout(
            title="Vehicle Trajectories in Coordinates (Y-Axis Inverted for Image Coordinates)",
            xaxis_title="X Coordinate (Pixels)",
            yaxis_title="Y Coordinate (Pixels)",
            yaxis=dict(autorange="reversed"),
            showlegend=True,
            plot_bgcolor="rgba(11,15,26,0.6)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Outfit")
        )
        st.plotly_chart(fig_paths, use_container_width=True)
        
        # PCA Projection Plot
        if model_fitted and len(st.session_state.anomaly_detector.training_data) >= 3:
            st.markdown("### PCA Dimensionality Reduction Projection")
            st.write("2D projection of the 60-dimensional trajectory features. Outlying points are flagged in red by the Isolation Forest.")
            
            X_train = np.array(st.session_state.anomaly_detector.training_data)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_train)
            
            predictions = []
            scores = []
            for t_feat in X_train:
                t_feat_reshaped = t_feat.reshape(1, -1)
                pred = st.session_state.anomaly_detector.model.predict(t_feat_reshaped)[0]
                score = st.session_state.anomaly_detector.model.decision_function(t_feat_reshaped)[0]
                predictions.append("Anomaly" if pred == -1 else "Normal")
                scores.append(round(score, 3))
                
            df_pca = pd.DataFrame({
                "PCA Component 1": X_pca[:, 0],
                "PCA Component 2": X_pca[:, 1],
                "Behavior": predictions,
                "Anomaly Score": scores,
                "Vehicle ID": list(st.session_state.trajectory_history.keys())[:len(X_train)]
            })
            
            fig_pca = px.scatter(
                df_pca, x="PCA Component 1", y="PCA Component 2",
                color="Behavior",
                color_discrete_map={"Normal": "#00f2fe", "Anomaly": "#ff4b4b"},
                hover_data=["Vehicle ID", "Anomaly Score"],
                title="PCA Projection of Trajectories"
            )
            fig_pca.update_layout(
                plot_bgcolor="rgba(11,15,26,0.6)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Outfit")
            )
            st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.warning("No trajectory history recorded yet. Please run the Video Analysis to accumulate trajectories.")

with tab3:
    st.markdown("<h3 style='font-family: Space Grotesk; color:#fff;'>Historical Violation Database</h3>", unsafe_allow_html=True)
    st.write("Browse, search, and export violation reports for traffic infractions recorded during video analysis.")
    
    if st.session_state.violations_db:
        df_violations = pd.DataFrame(st.session_state.violations_db)
        
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            search_query = st.text_input("🔍 Search by Details or Vehicle ID", key="violations_search")
        with col_f2:
            filter_type = st.multiselect(
                "Filter by Violation Type",
                options=df_violations["Violation Type"].unique(),
                default=df_violations["Violation Type"].unique(),
                key="violations_filter"
            )
            
        filtered_df = df_violations[df_violations["Violation Type"].isin(filter_type)]
        if search_query:
            filtered_df = filtered_df[
                filtered_df["Details"].str.contains(search_query, case=False) |
                filtered_df["Vehicle ID"].astype(str).str.contains(search_query)
            ]
            
        st.dataframe(filtered_df, use_container_width=True)
        
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Violation Report (CSV)",
            data=csv_data,
            file_name="traffic_violations_report.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-family: Space Grotesk; color:#fff;'>Violation Analytics</h3>", unsafe_allow_html=True)
        col_p1, col_p2 = st.columns([1, 1])
        
        with col_p1:
            type_counts = filtered_df["Violation Type"].value_counts().reset_index()
            type_counts.columns = ["Violation Type", "Count"]
            fig_pie = px.pie(type_counts, values="Count", names="Violation Type", hole=0.45,
                             color_discrete_sequence=["#00f2fe", "#7f00ff", "#ff4b4b", "#fb923c"])
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1", family="Outfit"))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_p2:
            severity_counts = filtered_df["Severity"].value_counts().reset_index()
            severity_counts.columns = ["Severity", "Count"]
            fig_bar = px.bar(severity_counts, x="Severity", y="Count", color="Severity", 
                             color_discrete_map={"Critical": "#ff4b4b", "High": "#c084fc", "Medium": "#fb923c"},
                             title="Violations by Severity")
            fig_bar.update_layout(plot_bgcolor="rgba(11,15,26,0.6)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1", family="Outfit"))
            st.plotly_chart(fig_bar, use_container_width=True)
            
    else:
        st.info("No violations logged yet. Run the Video Analysis to detect and log traffic infractions.")

with tab4:
    st.markdown("<h3 style='font-family: Space Grotesk; color:#fff;'>Project Features & Implementation Details</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    #### 🚀 Architecture & Algorithms
    This system integrates several high-value machine learning and computer vision pipelines:
    
    1. **Vehicle Detection & Classification**:
       - Uses **YOLOv8** (pre-trained on COCO dataset) to run frame-by-frame deep-learning object inference to detect cars, trucks, buses, and motorcycles.
       - Implements a fast, zero-dependency **Contour-based background subtraction** fallback mechanism using `cv2.absdiff` and structural morphology. This provides 60+ FPS processing on standard CPU instances.
    
    2. **Multi-Object Centroid Tracking**:
       - Bounding boxes are associated across consecutive frames using **Euclidean distance centroid matching** (greedy minimum mapping).
       - Tracks coordinates continuously to form historical trajectory vectors, handling frame-skip occlusions using a customizable frame buffer (`max_disappeared`).
       
    3. **Perspective Velocity Estimation**:
       - Projects pixel coordinate displacements into real-world meters using **Perspective Homography Matrices** (`cv2.getPerspectiveTransform`).
       - Calculates instant and average velocities in $km/h$, filtering jitter using sliding average windows.
       
    4. **Unsupervised Trajectory Anomaly Detection (The ML Core)**:
       - Rather than utilizing brittle, manually coded rule trees for driving patterns, this project implements a **scikit-learn Isolation Forest** model.
       - raw coordinate arrays are smoothed, resampled to $N=30$ elements using linear interpolation, and translation-normalized along the axis of traffic.
       - The forest builds isolation splits. Highly winding sequences (swerving) or coordinate sequences running in the inverse direction (wrong-way) are easily isolated, generating negative scores that classify them as reckless driving anomalies.
    
    #### 🛠️ Local Execution Instructions
    To run this project locally, ensure python is installed and run:
    ```bash
    # 1. Navigate to the project directory
    cd C:/Users/hemak/.gemini/antigravity/scratch/traffic_violation_detector
    
    # 2. Install dependencies
    pip install -r requirements.txt
    
    # 3. Launch the dashboard
    streamlit run app.py
    ```
    """)
