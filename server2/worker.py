import os
import time
import json
import cv2
import gc
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
from rembg import remove, new_session

try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
    _REMBG_PROVIDERS = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ] if torch.cuda.is_available() else ["CPUExecutionProvider"]
except Exception:  # pragma: no cover - torch may not be installed
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False
    _REMBG_PROVIDERS = ["CPUExecutionProvider"]

# Rembg does not ship a model named "dis". Use the default "u2net" model
# to avoid runtime errors when initializing the background removal session.
_REMBG_SESSION = new_session("u2net", providers=_REMBG_PROVIDERS)

# -------------------------
# Config / directories
# -------------------------
SHARED_DIR = "/mnt/shared"
RESIZED_DIR = os.path.join(SHARED_DIR, "resized")
MASKS_DIR = os.path.join(SHARED_DIR, "output", "masks")
SMALLS_DIR = os.path.join(SHARED_DIR, "output", "smalls")
CONFIG_FILE = os.path.join(SHARED_DIR, "config", "settings.json")
BIRENET_MODEL_PATH = os.path.join(SHARED_DIR, "models", "birefnet_dis.pth")
MODEL_PATH = os.path.join(SHARED_DIR, "models", "vit_l.pth")
PROCESSED_FILE = os.path.join(SHARED_DIR, "output", "processed.json")

AREA_THRESH = 1000  # pixel area below which masks are treated as "smalls"
MERGE_KERNEL = np.ones((5, 5), np.uint8)  # kernel for merging nearby masks


os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(SMALLS_DIR, exist_ok=True)


def _is_mostly_one_color(image_bgr: np.ndarray, mask: np.ndarray, threshold: float = 15.0) -> bool:
    pixels = image_bgr[mask > 0]
    if pixels.size == 0:
        return False
    return float(pixels.std(axis=0).mean()) < threshold


def _refine_mask_with_rembg(image_bgr: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    result = remove(pil_img, session=_REMBG_SESSION)
    alpha = np.array(result)[..., 3]
    return (alpha > 0).astype(np.uint8)


try:
    from birefnet import BiRefNet  # type: ignore
    if _TORCH_AVAILABLE and os.path.exists(BIRENET_MODEL_PATH):
        _BIRENET_MODEL = BiRefNet()
        _BIRENET_MODEL.load_state_dict(
            torch.load(BIRENET_MODEL_PATH, map_location="cpu")
        )
        _BIRENET_MODEL.eval()
    else:  # pragma: no cover - model file missing
        _BIRENET_MODEL = None
except Exception:  # pragma: no cover - birefnet not installed
    _BIRENET_MODEL = None


def _refine_mask_with_birefnet(image_bgr: np.ndarray) -> np.ndarray:
    if _BIRENET_MODEL is None:
        raise RuntimeError("BiRefNet model not available")
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        pred = _BIRENET_MODEL(tensor)[0]
    mask = (pred.sigmoid().squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return mask


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
            if masks:
                largest = max(masks, key=lambda m: int(np.count_nonzero(m["segmentation"])))
                if _is_mostly_one_color(img, largest["segmentation"]):
                    try:
                        largest["segmentation"] = _refine_mask_with_birefnet(img).astype(bool)
                    except Exception:
                        largest["segmentation"] = _refine_mask_with_rembg(img).astype(bool)
                h, w = img.shape[:2]
                total_pixels = h * w
                for m in masks:
                    if np.count_nonzero(m["segmentation"]) > 0.9 * total_pixels:
                        m["segmentation"] = np.logical_not(m["segmentation"])
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
