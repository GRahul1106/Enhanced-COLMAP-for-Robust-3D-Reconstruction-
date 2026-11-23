"""
generate_variant_e.py

Generates Person Hall Variant E:
Conservative Quality-Guided Adaptive Image Enhancement.

Selection Criteria:
  - Contrast_StdDev < 52.0
  - Baseline SIFT Features < 10,000

Selected Images (Exactly 5):
  - IMG_1179.JPG
  - IMG_1282.JPG
  - IMG_1312.JPG
  - IMG_1316.JPG
  - IMG_1323.JPG

Enhancement:
  - CLAHE in LAB color space (L channel only)
  - clipLimit = 1.2
  - tileGridSize = (8, 8)
  - Remaining 325 images copied byte-for-byte unchanged.
"""

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = PROJECT_ROOT / "experiments" / "person_hall" / "baseline"
BASE_IMAGES_DIR = BASE_DIR / "images"
CONTRAST_CSV = BASE_DIR / "image_contrast.csv"
SIFT_CSV = BASE_DIR / "baseline_sift_counts.csv"

VAR_E_DIR = PROJECT_ROOT / "experiments" / "person_hall" / "variant_e"
OUT_IMAGES_DIR = VAR_E_DIR / "images"
PREVIEWS_DIR = VAR_E_DIR / "previews"
MANIFEST_PATH = VAR_E_DIR / "enhancement_manifest.json"
LOG_CSV_PATH = VAR_E_DIR / "enhancement_log.csv"
LOG_TXT_PATH = VAR_E_DIR / "enhancement.log"

EXPECTED_SELECTED = {
    "IMG_1179.JPG",
    "IMG_1282.JPG",
    "IMG_1312.JPG",
    "IMG_1316.JPG",
    "IMG_1323.JPG",
}
CLIP_LIMIT = 1.2
TILE_GRID = (8, 8)


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("variant_e_generation")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def apply_clahe_bgr(img_bgr: np.ndarray, clip: float = 1.2, grid=(8, 8)) -> np.ndarray:
    """Apply CLAHE to L channel of LAB color space, preserving A and B channels."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    l_enh = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    logger = setup_logging(LOG_TXT_PATH)
    logger.info("=" * 70)
    logger.info("PERSON HALL VARIANT E GENERATION - START")
    logger.info(f"Source Baseline Images: {BASE_IMAGES_DIR}")
    logger.info(f"Output Images: {OUT_IMAGES_DIR}")
    logger.info(f"Target CLAHE Settings: clipLimit={CLIP_LIMIT}, tileGridSize={TILE_GRID}")
    logger.info("=" * 70)

    # 1. Verify source baseline images
    all_base_files = sorted(list(BASE_IMAGES_DIR.glob("*.JPG")))
    if len(all_base_files) != 330:
        raise ValueError(f"Expected 330 baseline images, found {len(all_base_files)}")

    # 2. Load and verify selection metrics
    if not CONTRAST_CSV.exists() or not SIFT_CSV.exists():
        raise FileNotFoundError("Missing contrast or SIFT CSV in baseline directory.")

    contrast_data = {}
    with open(CONTRAST_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            contrast_data[r["Image"]] = float(r["Contrast_StdDev"])

    sift_data = {}
    with open(SIFT_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sift_data[r["Image"]] = int(r["SIFT_Features"])

    computed_selected = set()
    for name, c_val in contrast_data.items():
        s_val = sift_data.get(name, 999999)
        if c_val < 52.0 and s_val < 10000:
            computed_selected.add(name)

    logger.info(f"Computed candidate images satisfying dual criteria: {sorted(list(computed_selected))}")
    if computed_selected != EXPECTED_SELECTED:
        raise ValueError(
            f"Computed selection {computed_selected} does not match expected {EXPECTED_SELECTED}"
        )

    # 3. Create destination directories
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    enhanced_count = 0
    unchanged_count = 0

    logger.info("Processing images (enhancing 5 selected, copying 325 unchanged)...")
    from PIL import Image

    for img_p in all_base_files:
        name = img_p.name
        dst_p = OUT_IMAGES_DIR / name
        c_val = contrast_data[name]
        s_val = sift_data[name]

        if name in EXPECTED_SELECTED:
            # Read image ignoring orientation to match the raw sensor orientation and preserve EXIF
            img_bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if img_bgr is None:
                raise ValueError(f"Could not load image: {img_p}")
            orig_h, orig_w = img_bgr.shape[:2]

            enh_bgr = apply_clahe_bgr(img_bgr, CLIP_LIMIT, TILE_GRID)
            enh_h, enh_w = enh_bgr.shape[:2]
            if (orig_h, orig_w) != (enh_h, enh_w):
                raise ValueError(f"Dimension mismatch on {name}: {orig_h}x{orig_w} vs {enh_h}x{enh_w}")

            # Extract EXIF metadata from original file
            with Image.open(img_p) as pil_src:
                orig_exif = pil_src.info.get("exif")

            # Save enhanced image with PIL preserving EXIF
            enh_rgb = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)
            pil_enh = Image.fromarray(enh_rgb)
            if orig_exif:
                pil_enh.save(dst_p, "JPEG", quality=95, exif=orig_exif)
            else:
                pil_enh.save(dst_p, "JPEG", quality=95)

            enhanced_count += 1

            # Generate before/after preview (for preview display, we can use oriented view)
            disp_orig = cv2.imread(str(img_p))
            disp_enh = cv2.cvtColor(np.array(Image.open(dst_p)), cv2.COLOR_RGB2BGR) if disp_orig.shape != img_bgr.shape else enh_bgr
            h_disp, w_disp = disp_orig.shape[:2]
            target_h = 1080
            scale = target_h / float(h_disp)
            new_w = int(w_disp * scale)
            orig_s = cv2.resize(disp_orig, (new_w, target_h), interpolation=cv2.INTER_AREA)
            enh_s = cv2.resize(disp_enh, (new_w, target_h), interpolation=cv2.INTER_AREA)

            hdr = np.zeros((60, new_w * 2, 3), dtype=np.uint8)
            cv2.putText(
                hdr, f"ORIGINAL: {name} (Contrast={c_val:.2f}, SIFT={s_val})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )
            cv2.putText(
                hdr, f"ENHANCED [CLAHE clip={CLIP_LIMIT}]: {name}", (new_w + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
            )
            panel = np.vstack([hdr, np.hstack([orig_s, enh_s])])
            preview_p = PREVIEWS_DIR / f"preview_{name}"
            cv2.imwrite(str(preview_p), panel)

            manifest_records.append({
                "Image": name,
                "Action": "Enhanced",
                "Algorithm": f"CLAHE(clip={CLIP_LIMIT}, grid={TILE_GRID})",
                "Contrast_StdDev": c_val,
                "SIFT_Features": s_val,
                "Dimensions": f"{orig_w}x{orig_h}",
                "Preview": str(preview_p.relative_to(PROJECT_ROOT)),
            })
        else:
            # Copy bit-for-bit
            shutil.copy2(img_p, dst_p)
            unchanged_count += 1
            manifest_records.append({
                "Image": name,
                "Action": "Unchanged",
                "Algorithm": "None (Bit-for-bit copy)",
                "Contrast_StdDev": c_val,
                "SIFT_Features": s_val,
                "Dimensions": f"{img_bgr.shape[1] if 'img_bgr' in locals() and name in EXPECTED_SELECTED else 'Preserved'}",
                "Preview": "None",
            })

    logger.info(f"Image generation completed: {enhanced_count} enhanced, {unchanged_count} unchanged.")

    # 4. Strict Validation
    logger.info("Performing strict image-set validation...")
    out_files = sorted(list(OUT_IMAGES_DIR.glob("*.JPG")))
    if len(out_files) != 330:
        raise ValueError(f"Validation failed: expected 330 images in output, found {len(out_files)}")

    # Check 325 unchanged files for byte-identity
    byte_mismatches = []
    for img_p in all_base_files:
        if img_p.name not in EXPECTED_SELECTED:
            dst_p = OUT_IMAGES_DIR / img_p.name
            if compute_file_sha256(img_p) != compute_file_sha256(dst_p):
                byte_mismatches.append(img_p.name)

    if byte_mismatches:
        raise ValueError(f"Validation failed: {len(byte_mismatches)} unchanged images differ from baseline!")
    logger.info("Byte-identity verified for all 325 unchanged images: 100% exact match.")

    # Check 5 enhanced files for validity and unchanged dimensions
    for name in EXPECTED_SELECTED:
        src_p = BASE_IMAGES_DIR / name
        dst_p = OUT_IMAGES_DIR / name
        im_src = cv2.imread(str(src_p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        im_dst = cv2.imread(str(dst_p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if im_dst is None:
            raise ValueError(f"Enhanced image {name} failed to decode!")
        if im_src.shape != im_dst.shape:
            raise ValueError(f"Enhanced image {name} dimension mismatch: {im_src.shape} vs {im_dst.shape}")
        with Image.open(src_p) as pil_s, Image.open(dst_p) as pil_d:
            if pil_s.size != pil_d.size:
                raise ValueError(f"Enhanced image {name} PIL size mismatch: {pil_s.size} vs {pil_d.size}")
            if pil_s.getexif().get(274) != pil_d.getexif().get(274):
                raise ValueError(f"Enhanced image {name} EXIF orientation mismatch!")
        if compute_file_sha256(src_p) == compute_file_sha256(dst_p):
            raise ValueError(f"Enhanced image {name} is unexpectedly identical to source!")
    logger.info("All 5 enhanced images verified: valid decodability, exact dimension match, EXIF preserved, distinct content.")

    # 5. Write manifest and CSV log
    with open(LOG_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Image", "Action", "Algorithm", "Contrast_StdDev", "SIFT_Features", "Dimensions", "Preview"]
        )
        writer.writeheader()
        writer.writerows(manifest_records)

    manifest_summary = {
        "dataset": "person-hall",
        "variant": "variant_e_conservative_selection",
        "total_images": 330,
        "enhanced_images_count": enhanced_count,
        "unchanged_images_count": unchanged_count,
        "clip_limit": CLIP_LIMIT,
        "tile_grid_size": list(TILE_GRID),
        "selection_criteria": {
            "contrast_stddev_threshold": 52.0,
            "sift_feature_count_threshold": 10000,
        },
        "enhanced_images": sorted(list(EXPECTED_SELECTED)),
        "previews_directory": str(PREVIEWS_DIR.relative_to(PROJECT_ROOT)),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_summary, f, indent=2)

    logger.info(f"Manifest written to: {MANIFEST_PATH}")
    logger.info(f"Enhancement log written to: {LOG_CSV_PATH}")
    logger.info("=" * 70)
    logger.info("IMAGE GENERATION AND VALIDATION SUCCESSFUL")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
