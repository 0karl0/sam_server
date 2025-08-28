import os
import time
import cv2
from ultralytics import YOLO

# Directories
# Shared directories
SHARED_DIR = "/mnt/shared"
RESIZED_DIR = os.path.join(SHARED_DIR, "resized")
OUTPUT_DIR = os.path.join(SHARED_DIR, "output", "boxes")

# Model directory: prefer /models (populated by Dockerfile),
# fall back to /mnt/shared/models for backwards compatibility.
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
if not os.path.isdir(MODEL_DIR):
    MODEL_DIR = os.path.join(SHARED_DIR, "models")


# Dynamically load all detection models found in MODEL_DI
MODELS: dict[str, YOLO] = {}
if os.path.isdir(MODEL_DIR):
    for fname in os.listdir(MODEL_DIR):
        if fname.lower().endswith((".pt", ".pth")):
            path = os.path.join(MODEL_DIR, fname)
            name = os.path.splitext(fname)[0]
            try:
                MODELS[name] = YOLO(path)
            except Exception as e:
                print(f"[Worker] Failed to load {fname}: {e}")
    if not MODELS:
        print(f"[Worker] No YOLO models found in {MODEL_DIR}")

else:
    print(f"[Worker] Model directory not found: {MODEL_DIR}")

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
            results = model(image)[0]
            annotated = image.copy()
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                label = results.names.get(cls_id, str(cls_id))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
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
