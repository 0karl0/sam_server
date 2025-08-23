import os
import time
import json
import cv2
import gc
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# -------------------------
# Config / directories
# -------------------------
SHARED_DIR = "/mnt/shared"
RESIZED_DIR = os.path.join(SHARED_DIR, "resized")
MASKS_DIR = os.path.join(SHARED_DIR, "output", "masks")
SMALLS_DIR = os.path.join(SHARED_DIR, "output", "smalls")
CONFIG_FILE = os.path.join(SHARED_DIR, "config", "settings.json")
MODEL_PATH = os.path.join(SHARED_DIR, "models", "vit_l.pth")
PROCESSED_FILE = os.path.join(SHARED_DIR, "output", "processed.json")

AREA_THRESH = 1000  # pixel area below which masks are treated as "smalls"
MERGE_KERNEL = np.ones((5, 5), np.uint8)  # kernel for merging nearby masks


os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(SMALLS_DIR, exist_ok=True)


def load_processed_set():
    """Build a set of base filenames that have already been processed."""
    processed = set()
    # Load from persisted json if present
    if os.path.exists(PROCESSED_FILE):
        try:
            with open(PROCESSED_FILE, "r") as f:
                processed.update(json.load(f))
        except Exception:
            pass
    # Also include any masks that already exist on disk
    for fname in os.listdir(MASKS_DIR):
        if "_mask" in fname:
            base = fname.split("_mask")[0]
            processed.add(base)
    return processed


def save_processed_set(processed_set):
    """Persist processed base filenames to disk atomically."""
    tmp = PROCESSED_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(processed_set), f)
    os.replace(tmp, PROCESSED_FILE)


def load_processed_set():
    """Build a set of base filenames that have already been processed."""
    processed = set()
    # Load from persisted json if present
    if os.path.exists(PROCESSED_FILE):
        try:
            with open(PROCESSED_FILE, "r") as f:
                processed.update(json.load(f))
        except Exception:
            pass
    # Also include any masks that already exist on disk
    for fname in os.listdir(MASKS_DIR):
        if "_mask" in fname:
            base = fname.split("_mask")[0]
            processed.add(base)
    return processed


def save_processed_set(processed_set):
    """Persist processed base filenames to disk atomically."""
    tmp = PROCESSED_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(processed_set), f)
    os.replace(tmp, PROCESSED_FILE)

# -------------------------
# Load SAM model
# -------------------------
sam = sam_model_registry["vit_l"](checkpoint=MODEL_PATH)
sam.to("cpu")  # CPU-only

# -------------------------
# Helper functions
# -------------------------
def load_settings():
    """Load SAM settings from Server1 JSON file."""
    default = {
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "crop_n_layers": 1,
        "model_type": "vit_l"
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            settings = json.load(f)
        default.update(settings)
    return default

def generate_masks(image_path, settings):
    """Generate masks for a single image."""
    image = cv2.imread(image_path)
    if image is None:
        return []

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=settings["points_per_side"],
        pred_iou_thresh=settings["pred_iou_thresh"],
        stability_score_thresh=settings["stability_score_thresh"],
        crop_n_layers=settings["crop_n_layers"]
       # min_mask_region_area=1000
    )

    masks = mask_generator.generate(image)
    return masks, image

def save_masks(masks, image, base_name):
    """Merge nearby masks, split into large vs small, and save."""
    h, w = image.shape[:2]

    # Combine all masks into a single binary map (resized to image size)
    combined = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        if seg.shape != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        combined = cv2.bitwise_or(combined, seg)

    # Merge nearby regions
    merged = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, MERGE_KERNEL)

    # Label connected components
    num_labels, labels = cv2.connectedComponents(merged)
    big_idx = 0
    small_idx = 0
    for label in range(1, num_labels):
        comp = (labels == label).astype(np.uint8)
        area = int(comp.sum())
        out = comp * 255
        if area < AREA_THRESH:
            out_path = os.path.join(SMALLS_DIR, f"{base_name}_small{small_idx}.png")
            small_idx += 1
        else:
            out_path = os.path.join(MASKS_DIR, f"{base_name}_mask{big_idx}.png")
            big_idx += 1
        cv2.imwrite(out_path, out)
    """Save masks as PNGs to MASKS_DIR"""
    for idx, mask_dict in enumerate(masks):
        mask = mask_dict["segmentation"].astype(np.uint8) * 255
        mask_file = os.path.join(MASKS_DIR, f"{base_name}_mask{idx}.png")
        # Resize to original image size if needed
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        cv2.imwrite(mask_file, mask)

# -------------------------
# Watcher loop
# -------------------------
processed = load_processed_set()

while True:
    settings = load_settings()
    for f in os.listdir(RESIZED_DIR):
        if not f.endswith((".png", ".jpg", ".jpeg")):
            continue
        base = os.path.splitext(f)[0]
        if base in processed:
            continue
        file_path = os.path.join(RESIZED_DIR, f)
        start = time.process_time()
        print(f"[Worker] Processing {f} ...")
        try:
            masks, img = generate_masks(file_path, settings)
            save_masks(masks, img, base)
            processed.add(base)
            save_processed_set(processed)
            # Clean up memory
            gc.collect()
            end = time.process_time()
            total = end - start
            print(f'elapsed time: {total:.6f} seconds')
        except Exception as e:
            print(f"[Worker] Error processing {f}: {e}")
    time.sleep(2)
