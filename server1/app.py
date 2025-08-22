import os
import json
import threading
import time
from typing import List, Dict

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import numpy as np
import cv2
from PIL import Image
import pillow_heif  # enables HEIC/HEIF decode in Pillow

# -------------------------
# Paths / Constants
# -------------------------
SHARED_DIR   = "/mnt/shared"
INPUT_DIR    = os.path.join(SHARED_DIR, "input")              # originals (PNG-normalized)
RESIZED_DIR  = os.path.join(SHARED_DIR, "resized")            # ≤1024 for SAM
MASKS_DIR    = os.path.join(SHARED_DIR, "output", "masks")    # from Server2
CROPS_DIR    = os.path.join(SHARED_DIR, "output", "crops")    # RGBA crops
CONFIG_DIR   = os.path.join(SHARED_DIR, "config")
SETTINGS_JSON = os.path.join(CONFIG_DIR, "settings.json")
CROPS_INDEX   = os.path.join(CROPS_DIR, "index.json")         # manifest linking crops to original

MAX_RESIZE = 1024  # longest side for SAM
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "heic", "heif"}

for d in [INPUT_DIR, RESIZED_DIR, MASKS_DIR, CROPS_DIR, CONFIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

app = Flask(__name__)

# -------------------------
# Helpers
# -------------------------
def normalize_to_png_and_save(pil_img: Image.Image, out_path_png: str, longest_side: int | None = None) -> None:
    """Optionally resize to longest_side, then save as PNG (preserve alpha if present)."""
    img = pil_img.convert("RGBA")
    if longest_side and max(img.size) > longest_side:
        img.thumbnail((longest_side, longest_side), Image.LANCZOS)
    img.save(out_path_png, "PNG")

def load_crops_index() -> Dict[str, List[str]]:
    if os.path.exists(CROPS_INDEX):
        try:
            with open(CROPS_INDEX, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_crops_index(index: Dict[str, List[str]]) -> None:
    tmp_path = CROPS_INDEX + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp_path, CROPS_INDEX)

def ensure_settings_defaults() -> dict:
    defaults = {
        "model_type": "vit_b",         # allow vit_b / vit_l / vit_h
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "crop_n_layers": 1
    }
    if os.path.exists(SETTINGS_JSON):
        try:
            with open(SETTINGS_JSON, "r") as f:
                data = json.load(f)
            defaults.update({k: data.get(k, v) for k, v in defaults.items()})
        except Exception:
            pass
    else:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(SETTINGS_JSON, "w") as f:
            json.dump(defaults, f, indent=2)
    return defaults

def make_rgba_crop(original_bgr: np.ndarray, mask_gray: np.ndarray) -> np.ndarray | None:
    """
    Apply mask as alpha channel to original and crop to bbox.
    Returns RGBA (H,W,4) or None if mask empty.
    """
    # Ensure mask matches original size
    if mask_gray.shape != original_bgr.shape[:2]:
        mask_gray = cv2.resize(mask_gray, (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Binary 0/255 uint8
    mask_u8 = (mask_gray > 0).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask_u8)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)

    # Convert to BGRA and apply alpha
    bgra = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask_u8

    crop = bgra[y:y+h, x:x+w]
    return crop

# -------------------------
# Upload & Settings
# -------------------------
@app.route("/", methods=["GET"])
def index():
    # Just render; the page fetches crops and settings via APIs
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """
    - Accepts most image types (incl. HEIC/HEIF).
    - Normalizes original to PNG in /input/<basename>.png
    - Produces resized PNG (≤1024) in /resized/<basename>.png
    """
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if not file or file.filename == "":
        return "No selected file", 400

    filename = secure_filename(file.filename)
    stem, ext = os.path.splitext(filename)
    ext = ext.lower().lstrip(".")

    if ext not in ALLOWED_EXT:
        return "Unsupported file type", 400

    # Read via Pillow (handles HEIC via pillow-heif)
    try:
        pil_img = Image.open(file.stream)
    except Exception as e:
        return f"Failed to open image: {e}", 400

    # Normalize original to PNG in /input
    input_png = os.path.join(INPUT_DIR, f"{stem}.png")
    normalize_to_png_and_save(pil_img, input_png, longest_side=None)  # keep original resolution

    # Also save a resized (≤1024) copy for SAM in /resized (same basename)
    resized_png = os.path.join(RESIZED_DIR, f"{stem}.png")
    normalize_to_png_and_save(pil_img, resized_png, longest_side=MAX_RESIZE)

    return jsonify({"status": "ok", "original": f"{stem}.png"})

# --- serve originals (normalized PNGs) ---
# Serve originals directly from /mnt/shared/input
@app.route("/input/<path:filename>", methods=["GET"])
def serve_input(filename):
    return send_from_directory(INPUT_DIR, filename)

# Albums: originals (from /input) + their crops (from crops index)
@app.route("/list_originals", methods=["GET"])
def list_originals():
    """
    Returns:
    [
      {
        "original": "penguin.png",
        "original_url": "/input/penguin.png",
        "crops": [
          {"file": "penguin_mask0.png", "url": "/crops/penguin_mask0.png"},
          ...
        ]
      },
      ...
    ]
    """
    index = load_crops_index()   # { "penguin.png": ["penguin_mask0.png", ...], ... }
    albums = []

    # Include every normalized original (PNG) in INPUT_DIR
    for f in sorted(os.listdir(INPUT_DIR)):
        if not f.lower().endswith(".png"):
            continue
        crop_files = index.get(f, [])
        crops = [{"file": c, "url": f"/crops/{c}"} for c in crop_files]
        albums.append({
            "original": f,
            "original_url": f"/input/{f}",
            "crops": crops
        })

    return jsonify(albums)




@app.route("/save_settings", methods=["POST"])
def save_settings():
    """
    Saves SAM settings to shared config for Server2.
    Expect JSON: { model_type, points_per_side, pred_iou_thresh, stability_score_thresh, crop_n_layers }
    """
    data = request.get_json(force=True, silent=True) or {}
    current = ensure_settings_defaults()
    current.update({
        "model_type": data.get("model_type", current["model_type"]),
        "points_per_side": int(data.get("points_per_side", current["points_per_side"])),
        "pred_iou_thresh": float(data.get("pred_iou_thresh", current["pred_iou_thresh"])),
        "stability_score_thresh": float(data.get("stability_score_thresh", current["stability_score_thresh"])),
        "crop_n_layers": int(data.get("crop_n_layers", current["crop_n_layers"])),
    })
    with open(SETTINGS_JSON, "w") as f:
        json.dump(current, f, indent=2)
    return jsonify({"status": "ok", "settings": current})

@app.route("/get_settings", methods=["GET"])
def get_settings():
    return jsonify(ensure_settings_defaults())

# -------------------------
# Serving crops & list API (with original association)
# -------------------------
@app.route("/crops/<path:filename>", methods=["GET"])
def serve_crop(filename):
    return send_from_directory(CROPS_DIR, filename)

@app.route("/list_crops", methods=["GET"])
def list_crops():
    """
    Returns JSON like:
    [
      { "file": "penguin_mask0.png", "url": "/crops/penguin_mask0.png", "original": "penguin.png" },
      ...
    ]
    """
    index = load_crops_index()
    items = []
    for original, crops in index.items():
        for c in crops:
            items.append({
                "file": c,
                "url": f"/crops/{c}",
                "original": original
            })
    return jsonify(items)

# -------------------------
# Mask watcher → cropper (runs in background)
# -------------------------
_processed_mask_files = set()

def process_mask_file(mask_path: str):
    """
    Given a mask PNG path like .../masks/<stem>_maskN.png
    - find /input/<stem>.png
    - create RGBA crop
    - save to /crops/<stem>_maskN.png
    - update index.json association
    """
    fname = os.path.basename(mask_path)
    if "_mask" not in fname:
        return

    base = fname.split("_mask")[0]                # stem
    original_png = os.path.join(INPUT_DIR, f"{base}.png")
    if not os.path.exists(original_png):
        # original not found; skip
        return

    # Load original + mask
    orig_bgr = cv2.imread(original_png, cv2.IMREAD_COLOR)
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if orig_bgr is None or mask_gray is None:
        return

    crop_rgba = make_rgba_crop(orig_bgr, mask_gray)
    if crop_rgba is None or crop_rgba.size == 0:
        return

    out_name = fname  # keep same naming pattern to match mask index (transparent crop image)
    out_path = os.path.join(CROPS_DIR, out_name)
    cv2.imwrite(out_path, crop_rgba)  # PNG with alpha

    # Update manifest
    index = load_crops_index()
    crops_for_original = index.get(f"{base}.png", [])
    if out_name not in crops_for_original:
        crops_for_original.append(out_name)
        index[f"{base}.png"] = crops_for_original
        save_crops_index(index)

def mask_watcher_loop():
    while True:
        try:
            for fname in os.listdir(MASKS_DIR):
                if not fname.lower().endswith(".png"):
                    continue
                fpath = os.path.join(MASKS_DIR, fname)
                if fpath in _processed_mask_files:
                    continue
                process_mask_file(fpath)
                _processed_mask_files.add(fpath)
        except Exception as e:
            # Keep the watcher alive even if one file causes an error
            print(f"[mask_watcher] error: {e}")
        time.sleep(0.5)  # light polling

# Start background watcher before serving
threading.Thread(target=mask_watcher_loop, daemon=True).start()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    ensure_settings_defaults()
    app.run(host="0.0.0.0", port=5050, threaded=True)
