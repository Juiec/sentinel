import cv2
import numpy as np
import json
import os
import time
from collections import deque
from ultralytics import YOLO, YOLOE
from datetime import datetime, timezone

# ==========================================
# import backend function for generating alert log payloads
# ==========================================
from backend import generate_alert_log_payload

# ==========================================
# CLASS AND GROUP CONFIGURATION
# ==========================================
CONFIG_PATH = "classes.json"
TARGET_CLASSES = []
ID_GROUPS = {}

if os.path.exists(CONFIG_PATH) and os.path.getsize(CONFIG_PATH) > 0:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            TARGET_CLASSES = data.get("classes", [])
            ID_GROUPS = {k: set(v) for k, v in data.get("groups", {}).items()}
    except json.JSONDecodeError as e:
        print(f"Error parsing {CONFIG_PATH}: {e}")
else:
    print(f"Warning: {CONFIG_PATH} is missing or empty.")

# Shortcuts
CIGARETTE_IDS = ID_GROUPS.get("cigarette", set())
VAPE_IDS = ID_GROUPS.get("vape", set())
PET_IDS = ID_GROUPS.get("pet", set())
TENTS_IDS = ID_GROUPS.get("tent", set())
TRASH_IDS = ID_GROUPS.get("trash", set())      
VEHICLE_IDS = ID_GROUPS.get("vehicle", set())   
RAILING_IDS = ID_GROUPS.get("railing", set())   
GROUND_IDS = ID_GROUPS.get("floor", set())

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
YOLOE_MODEL_PATH = "models/yoloe-26l-seg.pt"
POSE_MODEL_PATH = "models/yolo26n-pose.pt"

CONF_YOLOE = 0.20
CONF_POSE = 0.50
COOLDOWN_SECONDS = 20.0

# Centralized Alert Configuration mapping (keeps code DRY)
ALERT_CONFIGS = {
    "SMOKING": {
        "buffer": 30, "min_hits": 10, 
        "color": (0, 140, 255), "emoji": "🚬", "label": "SMOKER DETECTED"
    },
    "FALL": {
        "buffer": 10, "min_hits": 3, 
        "color": (0, 0, 255), "emoji": "🤕", "label": "FALL DETECTED"
    },
    "SITTING_GROUND": {
        "buffer": 20, "min_hits": 8, 
        "color": (255, 165, 0), "emoji": "🪑", "label": "SITTING DETECTED"
    },
    "LEANING_RAILING": {
        "buffer": 15, "min_hits": 5, 
        "color": (128, 0, 128), "emoji": "🚧", "label": "RAILING LEAN"
    }
}

# ==========================================
# HELPER CLASSES & FUNCTIONS
# ==========================================
class IndividualAlarmManager:
    """Manages rolling windows and time-based cooldowns (in seconds) for each tracked person ID."""
    def __init__(self, buffer_size: int, min_hits: int, cooldown_seconds: float):
        self.buffer_size = buffer_size
        self.min_hits = min_hits
        self.cooldown_seconds = cooldown_seconds
        self.histories = {}
        self.cooldowns = {}

    def update(self, active_track_ids, target_track_ids):
        for track_id in active_track_ids:
            if track_id not in self.histories:
                self.histories[track_id] = deque([False] * self.buffer_size, maxlen=self.buffer_size)
            self.histories[track_id].append(track_id in target_track_ids)

        # Cleanup stale IDs
        stale_ids = [tid for tid in self.histories if tid not in active_track_ids]
        for tid in stale_ids:
            del self.histories[tid]
            self.cooldowns.pop(tid, None)

    def is_triggered(self, track_id) -> bool:
        return sum(self.histories.get(track_id, [])) >= self.min_hits

    def can_log_trigger(self, track_id) -> bool:
        current_time = time.time()
        if self.is_triggered(track_id):
            if current_time >= self.cooldowns.get(track_id, 0.0):
                self.cooldowns[track_id] = current_time + self.cooldown_seconds
                return True
        return False

# --- Pose Math Helpers ---
def get_avg_y(keypoints, indices):
    """Helper to extract valid Y coordinates and return their average."""
    valid_ys = [keypoints[i][1] for i in indices if keypoints[i][1] > 0]
    return sum(valid_ys) / len(valid_ys) if valid_ys else 0

def get_upper_y(keypoints):
    """Fallback logic for upper body Y coordinate."""
    nose_y = keypoints[0][1]
    return nose_y if nose_y > 0 else get_avg_y(keypoints, [5, 6]) # shoulders

# --- Heuristics ---
def calculate_dual_model_smoking_probability(yolo_conf, keypoints, person_box):
    obj_score = min(1.0, max(0.0, (yolo_conf - 0.20) / 0.15))
    pose_score = 0.5 
    
    if keypoints is not None and len(keypoints) >= 17:
        _, y1, _, y2 = person_box
        box_height = max(1.0, y2 - y1)
        nose = keypoints[0][:2]
        
        valid_wrists = [keypoints[i][:2] for i in (9, 10) if keypoints[i][0] > 0 and keypoints[i][1] > 0]
        
        if nose[0] > 0 and valid_wrists:
            min_dist = min([np.linalg.norm(nose - w) for w in valid_wrists])
            norm_dist = min_dist / box_height
            
            if norm_dist < 0.20:
                pose_score = 1.0  
            elif norm_dist > 0.50:
                pose_score = 0.0  
            else:
                pose_score = 1.0 - ((norm_dist - 0.20) / 0.30)

    visual_confidence = (obj_score * 0.70) + (pose_score * 0.30)
    return visual_confidence * 1.0 # Thermal Cap

def is_person_sitting_on_ground(keypoints, person_box, ground_boxes):
    if len(keypoints) < 17: return False
    x1, y1, x2, y2 = person_box
    box_height, box_width = y2 - y1, x2 - x1
    
    avg_hip_y = get_avg_y(keypoints, [11, 12])
    if not avg_hip_y or (avg_hip_y - y1) / box_height < 0.45:
        return False 

    if ground_boxes:
        person_center_x = (x1 + x2) / 2
        for gx1, gy1, gx2, gy2 in ground_boxes:
            if gx1 <= person_center_x <= gx2:
                avg_ankle_y = get_avg_y(keypoints, [15, 16])
                if avg_ankle_y and avg_ankle_y >= gy1 - 20: 
                    return True

    return box_height < box_width * 1.4

def is_person_leaning_over_railing(person_box, keypoints, railing_boxes, margin=20):
    if not railing_boxes or len(keypoints) < 17: return False
    px1, py1, px2, py2 = person_box
    
    upper_y = get_upper_y(keypoints)
    if upper_y == 0: return False
    
    person_center_x = (px1 + px2) / 2
    for rx1, ry1, rx2, ry2 in railing_boxes:
        if rx1 - margin <= person_center_x <= rx2 + margin and upper_y >= (ry1 - margin):
            return True
    return False

def is_person_fallen(keypoints, person_box):
    x1, y1, x2, y2 = person_box
    box_width, box_height = x2 - x1, y2 - y1
    if box_width > (box_height * 1.3): return True
    if len(keypoints) < 17: return False

    upper_y = get_upper_y(keypoints)
    hip_y = get_avg_y(keypoints, [11, 12])

    if upper_y > 0 and hip_y > 0 and upper_y >= (hip_y - (box_height * 0.15)):
        leg_ys = [keypoints[i][1] for i in (13, 14, 15, 16) if keypoints[i][1] > 0]
        if not leg_ys or max(leg_ys) <= (hip_y + (box_height * 0.30)):
            return True
    return False

def is_target_inside_person(target_box, person_box, margin=25):
    cx1, cy1, cx2, cy2 = target_box
    px1, py1, px2, py2 = person_box
    tx, ty = (cx1 + cx2) / 2, (cy1 + cy2) / 2
    return (px1 - margin <= tx <= px2 + margin) and (py1 - margin <= ty <= py2 + margin)

# --- UI Drawing ---
def draw_glowing_box(image, box_coords, color=(0, 0, 255), label=None):
    x1, y1, x2, y2 = box_coords
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.20, image, 0.80, 0, image)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(image, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 255), 1)

    if label:
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_bg_y1 = max(0, y1 - text_h - 10)
        cv2.rectangle(image, (x1, label_bg_y1), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ==========================================
# MAIN PIPELINE CLASS
# ==========================================
class PipelineManager:
    def __init__(self):
        print("Initializing local models on RTX (CUDA:0)...")
        self.detector = YOLOE(YOLOE_MODEL_PATH)
        self.detector.set_classes(TARGET_CLASSES)
        self.pose_model = YOLO(POSE_MODEL_PATH)
        
        # Dynamically create alarm managers using the centralized config
        self.alarms = {
            alert_type: IndividualAlarmManager(
                cfg["buffer"], cfg["min_hits"], COOLDOWN_SECONDS
            ) for alert_type, cfg in ALERT_CONFIGS.items()
        }

    def process_frame(self, frame):
        det_results = self.detector.predict(source=frame, conf=CONF_YOLOE, device=0, verbose=False)[0]
        pose_results = self.pose_model.track(source=frame, conf=CONF_POSE, persist=True, device=0, verbose=False)[0]
        
        # 1. Gather Scene Entities
        env = {"railing": [], "ground": [], "trash": [], "targets": []}
        
        if det_results.boxes:
            for box in det_results.boxes:
                cls_id = int(box.cls[0])
                coords = [int(c) for c in box.xyxy[0]]
                if cls_id in CIGARETTE_IDS or cls_id in VAPE_IDS:
                    env["targets"].append({"coords": coords, "conf": float(box.conf[0])})
                elif cls_id in RAILING_IDS: env["railing"].append(coords)
                elif cls_id in GROUND_IDS:  env["ground"].append(coords)
                elif cls_id in TRASH_IDS:   env["trash"].append(coords)

        tracked_persons = {}
        if pose_results.boxes and pose_results.boxes.id is not None:
            kpts_tensor = pose_results.keypoints.xy.cpu().numpy() if pose_results.keypoints is not None else None
            for i, (box, track_id) in enumerate(zip(pose_results.boxes, pose_results.boxes.id)):
                coords = [int(c) for c in box.xyxy[0]]
                tracked_persons[int(track_id)] = {
                    "box": tuple(coords),
                    "keypoints": kpts_tensor[i] if kpts_tensor is not None else []
                }

        # 2. Analyze Behaviors
        current_hits = {key: set() for key in self.alarms.keys()}
        
        for tid, data in tracked_persons.items():
            p_box, p_kpts = data["box"], data["keypoints"]

            if is_person_fallen(p_kpts, p_box):
                current_hits["FALL"].add(tid)
            if is_person_leaning_over_railing(p_box, p_kpts, env["railing"]):
                current_hits["LEANING_RAILING"].add(tid)
            # if is_person_sitting_on_ground(p_kpts, p_box, env["ground"]):
            #     current_hits["SITTING_GROUND"].add(tid)

            for t_data in env["targets"]:
                if is_target_inside_person(t_data["coords"], p_box):
                    if calculate_dual_model_smoking_probability(t_data["conf"], p_kpts, p_box) >= 0.50:
                        current_hits["SMOKING"].add(tid)

        # 3. Update Alarm Managers
        active_ids = set(tracked_persons.keys())
        for alert_type, manager in self.alarms.items():
            manager.update(active_ids, current_hits[alert_type])

        # 4. Evaluate & Render 
        timestamp = datetime.now().strftime('%H:%M:%S')
        canvas = frame.copy()
        triggered_alert_types = set()

        for alert_type, manager in self.alarms.items():
            cfg = ALERT_CONFIGS[alert_type]
            
            for tid, data in tracked_persons.items():
                if manager.is_triggered(tid):
                    triggered_alert_types.add(alert_type)
                    draw_glowing_box(canvas, data["box"], cfg["color"], cfg["label"])
                    
                    # Process Logging payload
                    if manager.can_log_trigger(tid):
                        history = list(manager.histories[tid])
                        acc = round(sum(history) / len(history) * 100, 1) if history else 0.0
                        
                        # Generate a clean canvas strictly for this specific payload
                        temp_canvas = frame.copy()
                        temp_canvas = pose_results.plot(img=temp_canvas) if pose_results else temp_canvas
                        if alert_type == "SMOKING":
                            temp_canvas = det_results.plot(img=temp_canvas)
                        draw_glowing_box(temp_canvas, data["box"], cfg["color"], f"{alert_type} ID: {tid}")
                        
                        generate_alert_log_payload(temp_canvas, tid, alert_type, cfg["emoji"])
                        print(f"[{timestamp}] {cfg['emoji']} [ALERT: {alert_type}] Track ID {tid} | Accuracy: {acc}%")

        # 5. Final Output Compilation
        if triggered_alert_types:
            alert_text = f"ALARM: {' & '.join(triggered_alert_types)} DETECTED!"
            cv2.putText(canvas, alert_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            return canvas
            
        annotated = det_results.plot()
        return pose_results.plot(img=annotated) if pose_results else annotated

# ==========================================
# RUNNER
# ==========================================
def run_local_pipeline():
    pipeline = PipelineManager()
    cap = cv2.VideoCapture(0)
    print("Starting live pipeline with YOLOE-26 + YOLO26-Pose Detection. Press 'q' to exit. Press 's' to save a manual screenshot.")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            final_frame = pipeline.process_frame(frame)
            cv2.imshow("Local YOLOE-26 + YOLO26-Pose Pipeline", final_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                date_subfolder = datetime.now().strftime('%Y-%m-%d')
                generate_alert_log_payload(
                    annotated_frame=final_frame, 
                    track_id="MANUAL", 
                    alert_type="MANUAL_SCREENSHOT", 
                    emoji="📸",
                    output_dir=r"logs\captures",
                    image_dir=os.path.join("images", date_subfolder),
                    json_dir=os.path.join("json", date_subfolder)
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Local pipeline shut down cleanly.")

if __name__ == "__main__":
    run_local_pipeline()