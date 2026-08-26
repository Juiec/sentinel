import cv2
import base64
import json
import os
from datetime import datetime, timezone

def generate_alert_log_payload(
    annotated_frame, 
    track_id, 
    alert_type="ALERT", 
    emoji="🚨",
    output_dir=r"logs\detections",
    image_dir="images",
    json_dir="json",
    facility_id=1,
    site_id=1,
    zone_id=1,
    building_id=1,
    floor_id=1,
    creator_id=0,
    request_category=0
):
    """
    Generates the alert payload with custom enterable IDs, encodes the frame 
    to Base64, and locally saves the resulting JSON file to disk.
    """
    # 0. Construct full paths combining the base output_dir with dynamic subdirectories
    full_image_dir = os.path.join(output_dir, image_dir)
    full_json_dir = os.path.join(output_dir, json_dir)
    
    # Ensure local storage directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(full_image_dir, exist_ok=True)
    os.makedirs(full_json_dir, exist_ok=True)

    # 1. Encode the OpenCV frame to JPEG format bytes in memory
    success, encoded_image = cv2.imencode('.jpg', annotated_frame)
    if not success:
        print("Error: Failed to encode frame for attachment.")
        return None
        
    image_bytes = encoded_image.tobytes()
    file_size = len(image_bytes)
    
    # 2. Convert image bytes into a Base64 string for FileContent
    base64_file_content = base64.b64encode(image_bytes).decode('utf-8')
    
    # 3. Generate ISO 8601 timestamp with milliseconds
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    
    # Clean string format for paths and filenames
    event_name = alert_type.lower().replace(" ", "_")
    file_timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:-3]

    # NEW: Save the photo as an independent .jpg file locally using the dynamic path
    image_filename = f"{event_name}_track_{track_id}_{file_timestamp_str}.jpg"
    image_save_path = os.path.join(full_image_dir, image_filename)
    cv2.imwrite(image_save_path, annotated_frame)
    print(f"📷 [IMAGE SAVED] {image_save_path}")
    
    # 4. Construct the target schema dictionary using passed arguments
    payload = {
        "FacilityId": facility_id,
        "SiteId": site_id,
        "ZoneId": zone_id,
        "TaskStartTime": timestamp,
        "BuildingId": building_id,
        "FloorId": floor_id,
        "TaskDescription": f"Automated Alert: {alert_type} detected for Track ID {track_id}",
        "CreatorId": creator_id,
        "RequestCategory": request_category,
        "Attachments": [
            {
                "Root": "local_storage",
                "Path": f"detections/{event_name}",
                "Name": image_filename,
                "Size": file_size,
                "CreatedDate": timestamp,
                "ModifiedDate": timestamp,
                "FileContent": base64_file_content  # Preserved base64 encoded image string
            }
        ]
    }
    
    # 5. Save the JSON payload directly to local disk using the dynamic path
    filename = f"{event_name}_id_{track_id}_{file_timestamp_str}.json"
    file_path = os.path.join(full_json_dir, filename)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"💾 [LOCAL STORAGE] Saved alert log to: {file_path}")
    except Exception as e:
        print(f"❌ [ERROR] Failed to write local JSON file: {e}")
        
    return payload