import os
import time
import torch
import cv2
from datetime import datetime
from ultralytics import YOLOE

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
MODEL_PATH = r"C:\yolo-env\models\yoloe-26l-seg.pt"
SAVE_DIR = r"C:\yolo-env\logs\captures"
CONF_THRESHOLD = 0.35
CAM_INDEX = 0

# ==========================================
# CONFIGURATION WITH NEGATIVES
# ==========================================
TARGET_CLASSES = ["bag", "backpack", "handbag", "suitcase", "luggage", "briefcase"]

# Classes frequently mistaken for targets or common background clutter
NEGATIVE_CLASSES = ["person", "clothing", "jacket", "chair", "box", "pillow", "shadow", "hanged jacket"]

# ==========================================
# UPDATED PIPELINE
# ==========================================
class YOLOEStreamPipeline:
    def __init__(self, model_path=MODEL_PATH, conf=CONF_THRESHOLD):
        self.conf = conf
        self.show_boxes = True
        self.device = 0 if torch.cuda.is_available() else "cpu"

        print(f"Loading YOLOE model weights from '{model_path}'...")
        self.model = YOLOE(model_path)

        # Set target + negative classes combined in the prompt set
        self.target_classes = TARGET_CLASSES
        self.negative_classes = NEGATIVE_CLASSES
        all_prompts = self.target_classes + self.negative_classes
        
        print(f"Setting vocabulary (Targets + Negatives): {all_prompts}")
        self.model.set_classes(all_prompts)

    def process_frame(self, frame):
        canvas = frame.copy()

        # Inference runs over ALL target and negative class prompts
        results = self.model.predict(
            source=frame, 
            conf=self.conf, 
            device=self.device, 
            verbose=False
        )[0]

        if not self.show_boxes:
            return canvas

        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else []

            for i in range(len(xyxy)):
                cls_idx = int(cls_ids[i])
                cls_name = results.names.get(cls_idx, "object")

                # FILTER: Only draw bounding boxes for TARGET classes
                if cls_name in self.target_classes:
                    x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                    c = float(confs[i])
                    label = f"{cls_name} {c:.2f}"
                    draw_glowing_box(canvas, (x1, y1, x2, y2), color=(0, 255, 0), label=label)

        return canvas

# ==========================================
# UI DRAWING HELPERS
# ==========================================
def draw_glowing_box(image, box_coords, color=(0, 255, 0), label=None):
    x1, y1, x2, y2 = box_coords
    
    # Translucent box background
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.20, image, 0.80, 0, image)
    
    # Outer double border (glowing effect)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(image, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 255), 1)

    # Text label background & string
    if label:
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_bg_y1 = max(0, y1 - text_h - 10)
        cv2.rectangle(image, (x1, label_bg_y1), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ==========================================
# RUNNER
# ==========================================
def run_pipeline():
    pipeline = YOLOEStreamPipeline()
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f"Error: Could not access webcam at index {CAM_INDEX}.")
        return

    print("\nStarting live YOLOE stream.")
    print("  'q' -> Exit")
    print("  's' -> Save screenshot")
    print("  't' -> Toggle box overlays\n")

    fps_start = time.time()
    fps_frames = 0
    current_fps = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            final_frame = pipeline.process_frame(frame)

            # Smooth FPS calculation
            fps_frames += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                current_fps = round(fps_frames / elapsed, 1)
                fps_start = time.time()
                fps_frames = 0

            # Render persistent FPS overlay on every frame
            cv2.putText(
                final_frame, 
                f"FPS: {current_fps}", 
                (15, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 0), 
                2
            )

            cv2.imshow("Live Object Detection (YOLOE)", final_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                os.makedirs(SAVE_DIR, exist_ok=True)
                ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                out_path = os.path.join(SAVE_DIR, f"bag_{ts}.jpg")
                cv2.imwrite(out_path, final_frame)
                print(f"Saved screenshot to: {out_path}")
            elif key == ord('t'):
                pipeline.show_boxes = not pipeline.show_boxes
                status = "ON" if pipeline.show_boxes else "OFF"
                print(f"Box visualization: {status}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Pipeline shut down cleanly.")


if __name__ == "__main__":
    run_pipeline()