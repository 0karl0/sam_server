import os
import time
import cv2
import torch
import torchvision.transforms as T
from dataclasses import dataclass
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


@dataclass
class DetectionModel:
    name: str
    model: object
    kind: str  # 'yolo', 'detr', 'dfine'

    def predict(self, image):
        """Return list of (x1, y1, x2, y2, label)"""
        if self.kind in {"yolo", "dfine"}:  # D-FINE uses YOLO-style interface
            results = self.model(image)[0]
            out = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                label = results.names.get(cls_id, str(cls_id))
                out.append((x1, y1, x2, y2, label))
            return out
        if self.kind == "detr":
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(tensor)
            probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.7
            boxes = outputs["pred_boxes"][0, keep].cpu().numpy()
            class_ids = probas[keep].argmax(-1).cpu().numpy()
            h, w = image.shape[:2]
            out = []
            for (cx, cy, bw, bh), cls_id in zip(boxes, class_ids):
                x1 = int((cx - 0.5 * bw) * w)
                y1 = int((cy - 0.5 * bh) * h)
                x2 = int((cx + 0.5 * bw) * w)
                y2 = int((cy + 0.5 * bh) * h)
                out.append((x1, y1, x2, y2, str(cls_id)))
            return out
        return []


def load_models(model_dir: str, models: dict[str, DetectionModel] | None = None) -> dict[str, DetectionModel]:
    """Populate ``models`` with any weights found in ``model_dir``.

    Subsequent calls will only attempt to load weights that are not already
    present in ``models`` so that new files dropped into the directory are
    picked up automatically without restarting the worker.
    """

    if models is None:
        models = {}


    if not os.path.isdir(model_dir):
        print(f"[Worker] Model directory not found: {model_dir}")
        return models

    files = os.listdir(model_dir)
    print(f"[Worker] Scanning {model_dir} for weights: {files}")

    for fname in files:
        print(f"[Worker] Inspecting {fname}")
        if not fname.lower().endswith((".pt", ".pth")):
            continue

        name = os.path.splitext(fname)[0]
        if name in models:
            print(f"[Worker] Model already loaded: {name}")
            continue

        path = os.path.join(model_dir, fname)
        try:
            yolo = YOLO(path)
            models[name] = DetectionModel(name, yolo, "yolo")
            print(f"[Worker] Loaded YOLO model: {fname}")
            continue
        except Exception as e:
            print(f"[Worker] Failed to load {fname} with YOLO: {e}")

        lower = name.lower()
        if "detr" in lower:
            print(
                "[Worker] DETR weights detected; loading model definition "
                "facebookresearch/detr:detr_resnet50"
            )
            try:
                detr = torch.hub.load(
                    "facebookresearch/detr", "detr_resnet50", pretrained=False,
                    trust_repo=True,
                )
                state = torch.load(path, map_location="cpu")
                state = state.get("model", state)
                detr.load_state_dict(state)
                detr.eval()
                models[name] = DetectionModel(name, detr, "detr")
                print(f"[Worker] Loaded DETR model: {fname}")
            except Exception as e2:
                print(f"[Worker] Failed to load {fname} as DETR: {e2}")
            continue
        elif "dfine" in lower or "d-fine" in lower:
            print(
                "[Worker] D-FINE weights detected; loading model definition "
                "lyuwenyu/D-FINE:dfine_r18"
            )
            try:
                dfine = torch.hub.load(
                    "lyuwenyu/D-FINE", "dfine_r18", pretrained=False,
                    trust_repo=True,
                )

                state = torch.load(path, map_location="cpu")
                dfine.load_state_dict(state)
                dfine.eval()
                models[name] = DetectionModel(name, dfine, "dfine")
                print(f"[Worker] Loaded D-FINE model: {fname}")
            except Exception as e2:
                print(f"[Worker] Failed to load {fname} as D-FINE: {e2}")
            continue
        else:
            print(f"[Worker] No loader configured for {fname}")

    # Ensure built-in hub models are available even if no local weights are present
    if not any(m.kind == "detr" for m in models.values()):
        try:
            print("[Worker] Loading pretrained DETR via torch.hub")
            detr = torch.hub.load(
                "facebookresearch/detr", "detr_resnet50", pretrained=True, trust_repo=True
            )
            detr.eval()
            models["detr_resnet50"] = DetectionModel("detr_resnet50", detr, "detr")
        except Exception as e:
            print(f"[Worker] Failed to load DETR via torch.hub: {e}")

    if not any(m.kind == "dfine" for m in models.values()):
        try:
            print("[Worker] Loading pretrained D-FINE via torch.hub")
            dfine = torch.hub.load(
                "lyuwenyu/D-FINE", "dfine_r18", pretrained=True, trust_repo=True
            )
            dfine.eval()
            models["dfine_r18"] = DetectionModel("dfine_r18", dfine, "dfine")
        except Exception as e:
            print(f"[Worker] Failed to load D-FINE via torch.hub: {e}")

    if not models:
        print(f"[Worker] No detection models found in {model_dir}")
    return models



# Dynamically load all detection models found in MODEL_DIR
MODELS = load_models(MODEL_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Track which models have processed each image.  This allows newly added models
# to run on existing uploads instead of requiring users to re-upload files.
# {"image_stem": {"model_a", "model_b"}, ...}
processed: dict[str, set[str]] = {}

while True:
    # Load any newly added model weights before processing images
    load_models(MODEL_DIR, MODELS)

    files = [f for f in os.listdir(RESIZED_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print("[Worker] No images found")
        time.sleep(2)
        continue
    print(f"[Worker] Found {len(files)} image(s): {files}")
    for filename in files:
        base = os.path.splitext(filename)[0]
        path = os.path.join(RESIZED_DIR, filename)
        image = cv2.imread(path)
        if image is None:
            continue

        # Ensure there is a set to track processed models for this image
        done_models = processed.setdefault(base, set())

        for model_name, model in MODELS.items():
            if model_name in done_models:
                continue

            predictions = model.predict(image)
            annotated = image.copy()
            for x1, y1, x2, y2, label in predictions:
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

            # Remember that this model has processed this image
            done_models.add(model_name)
    time.sleep(2)
