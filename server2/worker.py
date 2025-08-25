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
PAGES_DIR = os.path.join(SHARED_DIR, "output", "pages")


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
    """Return set of base names already present in the pages directory."""
    processed = set()
    for fname in os.listdir(PAGES_DIR):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            processed.add(os.path.splitext(fname)[0])
    return processed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    os.makedirs(RESIZED_DIR, exist_ok=True)
    os.makedirs(PAGES_DIR, exist_ok=True)

    while True:
        files = [f for f in os.listdir(RESIZED_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
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
                if len(page_masks) == 0:
                    print(f"[Worker] No pages detected in {f}; moving original")
                    out_path = os.path.join(PAGES_DIR, f)
                    os.replace(file_path, out_path)
                else:
                    print(f"[Worker] Detected {len(page_masks)} page(s) in {f}")
                    combined = np.zeros(img.shape[:2], dtype=np.uint8)
                    for page_mask in page_masks:
                        combined = cv2.bitwise_or(combined, page_mask)
                    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    bgra[:, :, 3] = combined
                    out_path = os.path.join(PAGES_DIR, f)
                    cv2.imwrite(out_path, bgra)
                    os.remove(file_path)
                processed.add(base)
                end = time.process_time()
                total = end - start
                print(f"elapsed time: {total:.6f} seconds")
            except Exception as e:
                print(f"[Worker] Error processing {f}: {e}")
        time.sleep(2)


if __name__ == "__main__":
    main()
