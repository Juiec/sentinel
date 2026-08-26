import os
import time
import cv2
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
MODEL_PATH = r"C:\yolo-env\models\bagWn.pt"   # your trained bag weights
CONF_BAG = 0.38           # confidence threshold for the bag class
CAM_INDEX = 0             # webcam index (same as detection.py using cv2.VideoCapture(0))

# ==========================================
# MAIN PIPELINE CLASS
# ==========================================
class BagPipeline:
    def __init__(self):
        print("Initializing YOLO26 bag model...")
        self.model = YOLO(MODEL_PATH)
        self.track_viz = True      # live box overlay toggle (keypress 't')

    def process_frame(self, frame):
        results = self.model.predict(source=frame, conf=CONF_BAG, device=0, verbose=False)[0]

        canvas = frame.copy()
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()          # (N,4)
            confs = boxes.conf.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                c = float(confs[i])
                draw_glowing_box(canvas, (x1, y1, x2, y2), (0, 255, 0), "bag %.2f" % c)

        return canvas


# --- UI Drawing (copied pattern from detection.py) ---
def draw_glowing_box(image, box_coords, color=(0, 255, 0), label=None):
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
# RUNNER
# ==========================================
def run_bag_pipeline():
    pipeline = BagPipeline()
    cap = cv2.VideoCapture(CAM_INDEX)
    print("Starting live bag detection. Press 'q' to exit, 's' to save a screenshot, 't' to toggle boxes.")

    fps_t = time.time()
    fps_n = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            final_frame = pipeline.process_frame(frame)
            cv2.imshow("Live Bag Detection (YOLO26)", final_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                out = r"C:\yolo-env\logs\captures" + "\\bag_%s.jpg" % ts
                os.makedirs(r"C:\yolo-env\logs\captures", exist_ok=True)
                cv2.imwrite(out, final_frame)
                print("Saved screenshot: " + out)
            elif key == ord('t'):
                pipeline.track_viz = not pipeline.track_viz
                print("Box visualization: %s" % ("ON" if pipeline.track_viz else "OFF"))
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Bag detection pipeline shut down cleanly.")


if __name__ == "__main__":
    run_bag_pipeline()