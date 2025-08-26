import os
import time
import argparse
import cv2
import numpy as np

# -------------------------
# Config / directories
# -------------------------
SHARED_DIR = "/mnt/shared"
RESIZED_DIR = os.path.join(SHARED_DIR, "resized")
# Original-resolution images live here; used for full-image fallback
INPUT_DIR = os.path.join(SHARED_DIR, "input")
# Directory where this worker will emit mask PNGs for each detected page
MASKS_DIR = os.path.join(SHARED_DIR, "output", "masks")


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


def _resize_to_max(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Resize ``image`` so the longest side is ``max_side`` or smaller."""
    h, w = image.shape[:2]
    scale = min(max_side / float(max(h, w)), 1.0)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


def load_processed_set() -> set[str]:
    """Return set of base names already present in the masks directory."""
    processed: set[str] = set()
    for fname in os.listdir(MASKS_DIR):
        if not fname.lower().endswith(".png") or "_mask" not in fname:
            continue
        base = fname.split("_mask")[0]
        processed.add(base)
    return processed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    os.makedirs(RESIZED_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)

    while True:
        files = [
            f
            for f in os.listdir(RESIZED_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not files:
            #print("[Worker] No pages found")
            time.sleep(2)
            continue
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
                if not page_masks:
                    # Fallback: treat the entire original image as one mask
                    print(f"[Worker] No pages detected in {f}; using full image mask")
                    orig_path = os.path.join(INPUT_DIR, f)
                    orig_img = cv2.imread(orig_path)
                    if orig_img is not None:
                        mask_shape = orig_img.shape[:2]
                    else:
                        mask_shape = img.shape[:2]
                    page_masks = [np.ones(mask_shape, dtype=np.uint8) * 255]
                else:
                    print(f"[Worker] Detected {len(page_masks)} page(s) in {f}")

                for idx, mask in enumerate(page_masks):
                    mask_path = os.path.join(MASKS_DIR, f"{base}_mask{idx}.png")
                    cv2.imwrite(mask_path, mask)

                processed.add(base)
                end = time.process_time()
                total = end - start
                print(f"elapsed time: {total:.6f} seconds")
            except Exception as e:
                print(f"[Worker] Error processing {f}: {e}")
        time.sleep(2)


if __name__ == "__main__":
    main()
