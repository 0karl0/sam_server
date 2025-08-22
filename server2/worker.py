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
CONFIG_FILE = os.path.join(SHARED_DIR, "config", "settings.json")
MODEL_PATH = os.path.join(SHARED_DIR, "models", "vit_l.pth")
PROCESSED_FILE = os.path.join(SHARED_DIR, "output", "processed.json")

os.makedirs(MASKS_DIR, exist_ok=True)


def load_processed_list():
    """Read list of already-processed filenames from disk."""
    if os.path.exists(PROCESSED_FILE):
        try:
            with open(PROCESSED_FILE, "r") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_processed_list(processed_set):
    """Persist processed filenames to disk atomically."""
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
#        points_per_side=settings["points_per_side"],
#        pred_iou_thresh=settings["pred_iou_thresh"],
#        stability_score_thresh=settings["stability_score_thresh"],
#        crop_n_layers=settings["crop_n_layers"]
    )

    masks = mask_generator.generate(image)
    return masks, image

def save_masks(masks, image, base_name):
    """Save masks as PNGs to MASKS_DIR"""
    for idx, mask_dict in enumerate(masks):
        mask = mask_dict["segmentation"].astype(np.uint8) * 255
        mask_file = os.path.join(
            MASKS_DIR, f"{os.path.splitext(base_name)[0]}_mask{idx}.png"
        )
        # Resize to original image size if needed
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        cv2.imwrite(mask_file, mask)


def already_processed(filename, processed_set):
    """Return True if masks or a processed marker exist for the given image."""
    base = os.path.splitext(filename)[0]
    prefix = f"{base}_mask"
    if any(fname.startswith(prefix) for fname in os.listdir(MASKS_DIR)):
        return True
    return filename in processed_set

# -------------------------
# Watcher loop
# -------------------------
processed = load_processed_list()

while True:
    settings = load_settings()
    for f in os.listdir(RESIZED_DIR):
        if not f.endswith((".png", ".jpg", ".jpeg")):
            continue
        if already_processed(f, processed):
            continue
        file_path = os.path.join(RESIZED_DIR, f)
        start = time.process_time()
        print(f"[Worker] Processing {f} ...")
        try:
            masks, img = generate_masks(file_path, settings)
            save_masks(masks, img, f)
            processed.add(f)
            save_processed_list(processed)
            # Clean up memory
            gc.collect()
            end = time.process_time()
            total = end - start
            print(f'elapsed time: {total:.6f} seconds')
        except Exception as e:
            print(f"[Worker] Error processing {f}: {e}")
    time.sleep(2)
