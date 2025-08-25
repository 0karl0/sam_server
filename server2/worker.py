import os
import time
import json
import gc
import argparse
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

try:
    import torch  # type: ignore
    print("importing torch")
    _TORCH_AVAILABLE = True
    _REMBG_PROVIDERS = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ] if torch.cuda.is_available() else ["CPUExecutionProvider"]
except Exception:  # pragma: no cover - torch may not be installed
    print("couldn't import torch")
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
PAGES_DIR = os.path.join(SHARED_DIR, "output", "pages")
CONFIG_FILE = os.path.join(SHARED_DIR, "config", "settings.json")
MODEL_PATH = os.path.join(SHARED_DIR, "models", "vit_l.pth")
PROCESSED_FILE = os.path.join(SHARED_DIR, "output", "processed.json")

AREA_THRESH = 1000  # pixel area below which masks are treated as "smalls"

# Global placeholders for heavy dependencies loaded lazily
sam = None
_REMBG_SESSION = None


def _lazy_load_dependencies():
    global cv2, np, sam_model_registry, SamAutomaticMaskGenerator
    global Image, remove, new_session, _REMBG_SESSION, sam


    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


    # Load BirefNet session from the shared models directory.
    os.environ.setdefault("U2NET_HOME", os.path.join(SHARED_DIR, "models"))
    _REMBG_SESSION = new_session("birefnet-dis", providers=_REMBG_PROVIDERS)

    # Load SAM model (CPU-only)
    sam = sam_model_registry["vit_l"](checkpoint=MODEL_PATH)
    sam.to("cpu")

def _refine_mask_with_rembg(image_bgr: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    print("[Decision] running rembg remove")
    result = remove(pil_img, session=_REMBG_SESSION)
    alpha = np.array(result)[..., 3]
    print("[Decision] rembg remove complete")
    return (alpha.any() > 0).astype(np.uint8)


def _refine_mask_with_birefnet(image_bgr: np.ndarray) -> np.ndarray:
    """Refine mask using the BirefNet session.

    This simply delegates to rembg with the preloaded BirefNet model. A
    separate helper makes it easy to catch errors and fall back to the generic
    rembg model if needed.
    """
    print("birefnet")
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
    mask = np.squeeze(mask)
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
    mask_u8 = (np.squeeze(mask) > 0).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask_u8)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask_u8
    return bgra[y:y+h, x:x+w]


def _detect_pages(image_bgr: np.ndarray) -> list[np.ndarray]:
    """Detect sheet-of-paper-like regions and return their masks."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float(np.count_nonzero(thresh)) / thresh.size
    inverted = False
    if white_ratio < 0.5:
        thresh = cv2.bitwise_not(thresh)
        inverted = True
    print(
        f"[Decision] _detect_pages: white_ratio={white_ratio:.2f}, inverted={inverted}"
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[Decision] _detect_pages: {len(contours)} contour(s) found")
    h, w = gray.shape
    img_area = h * w
    pages: list[np.ndarray] = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        is_page = area > 0.2 * img_area and len(approx) == 4
        print(
            f"[Decision] contour {idx}: area={area:.0f}, vertices={len(approx)} -> {'page' if is_page else 'discard'}"
        )
        if is_page:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, 255, -1)
            pages.append(mask)
    print(f"[Decision] _detect_pages: detected {len(pages)} page(s)")
    return pages


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

def generate_masks(image, settings):
    """Generate masks for a single image array."""
    print("using sam")
    if image is None:
        return [], image

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
    """Save each mask individually without merging.

    The masks generated by SAM are resized to the original image size and
    written out as separate PNG files. Smaller components are stored
    separately from larger ones based on ``AREA_THRESH``.
    """

    h, w = image.shape[:2]

    big_idx = 0
    small_idx = 0
    for m in masks:
        seg = np.squeeze(m["segmentation"]).astype(np.uint8)
        if seg.shape != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)

        area = int(seg.sum())
        out = seg * 255
        if area < AREA_THRESH:
            mask_path = os.path.join(SMALLS_DIR, f"{base_name}_small{small_idx}.png")
            crop_path = os.path.join(CROPS_DIR, f"{base_name}_small{small_idx}.png")
            small_idx += 1
        else:
            mask_path = os.path.join(MASKS_DIR, f"{base_name}_mask{big_idx}.png")
            crop_path = os.path.join(CROPS_DIR, f"{base_name}_mask{big_idx}.png")
            big_idx += 1
        cv2.imwrite(mask_path, out)
        crop = _crop_with_mask(image, seg)
        if crop is not None:
            cv2.imwrite(crop_path, crop)

# -------------------------
# Watcher loop
# -------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load heavy dependencies at startup instead of waiting for work",
    )
    args = parser.parse_args()

    os.makedirs(RESIZED_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(SMALLS_DIR, exist_ok=True)
    os.makedirs(CROPS_DIR, exist_ok=True)
    os.makedirs(PAGES_DIR, exist_ok=True)

    deps_loaded = False
    if args.preload:
        _lazy_load_dependencies()
        deps_loaded = True

    while True:
        files = [f for f in os.listdir(RESIZED_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            print("[Worker] No pages found")
            time.sleep(2)
            continue
        if not deps_loaded:
            _lazy_load_dependencies()
            deps_loaded = True
        settings = load_settings()
        processed = load_processed_set()
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
                page_masks = _detect_pages(img)
                if len(page_masks) == 0:
                    print(f"[Worker] No pages detected in {f}")
                    processed.add(base)
                    save_processed_set(processed)
                    continue
                print(f"[Worker] Detected {len(page_masks)} page(s) in {f}")
                for idx, page_mask in enumerate(page_masks):
                    page_base = f"{base}_page{idx}"
                    crop = _crop_with_mask(img, page_mask)
                    if crop is None:
                        continue
                    page_path = os.path.join(PAGES_DIR, f"{page_base}.png")
                    cv2.imwrite(page_path, crop)
                    page_img = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
                    if _is_line_drawing(page_img):
                        print(f"[Worker] {page_base}: Using rembg for line drawing")
                        mask = _refine_mask_with_rembg(page_img)
                        mask_file = os.path.join(MASKS_DIR, f"{page_base}_mask0.png")
                        cv2.imwrite(mask_file, mask.astype(np.uint8) * 255)
                        crop_file = os.path.join(CROPS_DIR, f"{page_base}_mask0.png")
                        crop_refined = _crop_with_mask(page_img, mask)
                        if crop_refined is None:
                            crop_refined = crop
                        cv2.imwrite(crop_file, crop_refined)
                    else:
                        print(f"[Worker] {page_base}: Using SAM for segmentation")
                        masks, page_img = generate_masks(page_img, settings)
                        if len(masks) > 0:
                            largest = max(
                                masks,
                                key=lambda m: int(np.count_nonzero(m["segmentation"]))
                            )
                            if _is_mostly_one_color(page_img, largest["segmentation"]):
                                try:
                                    print("refining with birefnet")
                                    largest["segmentation"] = _refine_mask_with_birefnet(page_img).astype(bool)
                                except Exception:
                                    print("refining with rembg")
                                    largest["segmentation"] = _refine_mask_with_rembg(page_img).astype(bool)
                            h, w = page_img.shape[:2]
                            total_pixels = h * w
                            center_y, center_x = h // 2, w // 2
                            for m in list(masks):
                                seg = np.squeeze(m["segmentation"]).astype(bool)
                                seg_resized = seg
                                if seg.shape != (h, w):
                                    seg_resized = cv2.resize(
                                        seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                                    ).astype(bool)
                                area = np.count_nonzero(seg_resized)
                                if area > 0.9 * total_pixels:
                                    if np.any(seg_resized[center_y, center_x]):
                                        masks.remove(m)
                                        continue
                                    inverse = m.copy()
                                    inverse["segmentation"] = np.logical_not(seg)
                                    masks.append(inverse)
                        save_masks(masks, page_img, page_base)
                processed.add(base)
                save_processed_set(processed)
                gc.collect()
                end = time.process_time()
                total = end - start
                print(f"elapsed time: {total:.6f} seconds")
            except Exception as e:
                print(f"[Worker] Error processing {f}: {e}")
        time.sleep(2)


if __name__ == "__main__":
    main()
