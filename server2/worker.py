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

# -------------------------
# Config / directories
# -------------------------
SHARED_DIR = "/mnt/shared"
RESIZED_DIR = os.path.join(SHARED_DIR, "resized")
MASKS_DIR = os.path.join(SHARED_DIR, "output", "masks")
SMALLS_DIR = os.path.join(SHARED_DIR, "output", "smalls")
CROPS_DIR = os.path.join(SHARED_DIR, "output", "crops")
CONFIG_FILE = os.path.join(SHARED_DIR, "config", "settings.json")
MODEL_PATH = os.path.join(SHARED_DIR, "models", "vit_l.pth")
PROCESSED_FILE = os.path.join(SHARED_DIR, "output", "processed.json")

AREA_THRESH = 1000  # pixel area below which masks are treated as "smalls"
MERGE_KERNEL = np.ones((5, 5), np.uint8)  # kernel for merging nearby masks

# Load BirefNet session from the shared models directory if available to avoid
# downloading the model at runtime.
_BIREFNET_MODEL = os.path.join(SHARED_DIR, "models", "birefnet-dis.onnx")
_REMBG_SESSION = new_session(
    _BIREFNET_MODEL if os.path.exists(_BIREFNET_MODEL) else "birefnet-dis",
    providers=_REMBG_PROVIDERS,
)


os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(SMALLS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)


def _refine_mask_with_rembg(image_bgr: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    print("[Decision] running rembg remove")
    result = remove(pil_img, session=_REMBG_SESSION)
    alpha = np.array(result)[..., 3]
    print("[Decision] rembg remove complete")
    return (alpha > 0).astype(np.uint8)


def _refine_mask_with_birefnet(image_bgr: np.ndarray) -> np.ndarray:
    """Refine mask using the BirefNet session.

    This simply delegates to rembg with the preloaded BirefNet model. A
    separate helper makes it easy to catch errors and fall back to the generic
    rembg model if needed.
    """
    return _refine_mask_with_rembg(image_bgr)


def _is_line_drawing(image_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size
    color_std = float(image_bgr.std())
    result = edge_ratio > 0.05 and color_std < 25.0
    print(
        f"[Decision] _is_line_drawing: edge_ratio={edge_ratio:.4f}, color_std={color_std:.2f} -> {result}"
    )
    return result


def _is_mostly_one_color(image_bgr: np.ndarray, mask: np.ndarray, std_thresh: float = 5.0) -> bool:
    """Return True if the region defined by mask has little color variation."""
    if mask.shape != image_bgr.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    masked_pixels = image_bgr[mask.astype(bool)]
    if masked_pixels.size == 0:
        return False
    std = float(masked_pixels.std())
    result = std < std_thresh
    print(
        f"[Decision] _is_mostly_one_color: std={std:.2f}, thresh={std_thresh} -> {result}"
    )
    return result


def _crop_with_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask_u8)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask_u8
    return bgra[y:y+h, x:x+w]


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
    """Combine overlapping or nearby masks before saving.

    All masks generated by SAM are first resized to the original image size and
    OR-ed together. A morphological close operation then joins regions that
    either overlap or lie within a small distance of one another. Each connected
    component is written out as a single mask PNG. Smaller components are stored
    separately from larger ones based on ``AREA_THRESH``.
    """

    h, w = image.shape[:2]

    # Combine all masks into a single binary map (resized to image size)
    combined = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        if seg.shape != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        combined = cv2.bitwise_or(combined, seg)

    # Merge overlapping or nearby regions
    merged = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, MERGE_KERNEL)

    # Label connected components and save each as its own mask
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

# -------------------------
# Watcher loop
# -------------------------
processed = load_processed_set()

while True:
    settings = load_settings()
    files = [f for f in os.listdir(RESIZED_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print("[Worker] No pages found")
        time.sleep(2)
        continue
    print(f"[Worker] Found {len(files)} page(s): {files}")
    for f in files:
        base = os.path.splitext(f)[0]
        if base in processed:
            continue
        file_path = os.path.join(RESIZED_DIR, f)
        start = time.process_time()
        print(f"[Worker] Processing {f} ...")
        try:
            img = cv2.imread(file_path)
            if img is None:
                continue
            if _is_line_drawing(img):
                print("[Worker] Using rembg for line drawing")
                mask = _refine_mask_with_rembg(img)
                crop = _crop_with_mask(img, mask)
                mask_file = os.path.join(MASKS_DIR, f"{base}_mask0.png")
                cv2.imwrite(mask_file, mask.astype(np.uint8) * 255)
                if crop is not None:
                    crop_file = os.path.join(CROPS_DIR, f"{base}_mask0.png")
                    cv2.imwrite(crop_file, crop)
            else:
                print("[Worker] Using SAM for segmentation")
                masks, img = generate_masks(file_path, settings)


                if masks:
                    largest = max(masks, key=lambda m: int(np.count_nonzero(m["segmentation"])))
                    if _is_mostly_one_color(img, largest["segmentation"]):
                        try:
                            print("refining with birefnet")
                            largest["segmentation"] = _refine_mask_with_birefnet(img).astype(bool)
                        except Exception:
                            print("refining with rembg")
                            largest["segmentation"] = _refine_mask_with_rembg(img).astype(bool)
                    h, w = img.shape[:2]
                    total_pixels = h * w
                    center_y, center_x = h // 2, w // 2
                    for m in list(masks):
                        seg = m["segmentation"]
                        seg_resized = seg
                        if seg.shape != (h, w):
                            seg_resized = cv2.resize(
                                seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        area = np.count_nonzero(seg_resized)
                        if area > 0.9 * total_pixels:
                            if seg_resized[center_y, center_x]:
                                masks.remove(m)
                                continue
                            inverse = m.copy()
                            inverse["segmentation"] = np.logical_not(seg)
                            masks.append(inverse)
                save_masks(masks, img, base)
            processed.add(base)
            save_processed_set(processed)
            gc.collect()
            end = time.process_time()
            total = end - start
            print(f'elapsed time: {total:.6f} seconds')
        except Exception as e:
            print(f"[Worker] Error processing {f}: {e}")
    time.sleep(2)
