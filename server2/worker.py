import os
import time
import cv2
from ultralytics import YOLO

# Directories
SHARED_DIR = "/mnt/shared"
RESIZED_DIR = os.path.join(SHARED_DIR, "resized")
OUTPUT_DIR = os.path.join(SHARED_DIR, "output", "boxes")
MODEL_DIR = "/models"

# Load YOLO models
MODELS = {
    "yolov8n-art": YOLO(os.path.join(MODEL_DIR, "yolov8n-art.pt")),
    "yolov8s-art": YOLO(os.path.join(MODEL_DIR, "yolov8s-art.pt")),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

processed = set()

while True:
    files = [f for f in os.listdir(RESIZED_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print("[Worker] No images found")
        time.sleep(2)
        continue
    print(f"[Worker] Found {len(files)} image(s): {files}")
    for filename in files:
        base = os.path.splitext(filename)[0]
        if base in processed:
            continue
        path = os.path.join(RESIZED_DIR, filename)
        image = cv2.imread(path)
        if image is None:
            continue
        for model_name, model in MODELS.items():
            results = model(image)
            annotated = image.copy()
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                model_name,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            out_file = os.path.join(OUTPUT_DIR, f"{base}_{model_name}.png")
            cv2.imwrite(out_file, annotated)
        processed.add(base)
    time.sleep(2)
