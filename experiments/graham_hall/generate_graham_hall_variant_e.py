"""
generate_graham_hall_variant_e.py

Generates Phase B: Graham Hall Variant E Enhanced Image Collection.

Selection Criteria:
  contrast < 52.0 AND 0 < baseline SIFT features < 10,000

Enhancement Operator:
  CLAHE on CIELAB L* channel: clipLimit=1.2, tileGridSize=(8, 8)
  Preserves color-space balance, EXIF, and native image dimensions.

Outputs:
  Images: experiments/graham_hall/variant_e/images/
  Previews: experiments/graham_hall/variant_e/previews/
  Manifest: experiments/graham_hall/variant_e/enhancement_manifest.json
  Enhancement Log: experiments/graham_hall/variant_e/enhancement_log.csv
  Log: experiments/graham_hall/variant_e/enhancement.log
"""

import csv
import hashlib
import json
import logging
from pathlib import Path
import shutil
import cv2
import numpy as np

PROJECT_ROOT = Path(r"E:\CV Project").resolve()
BASE_DIR = PROJECT_ROOT / "experiments" / "graham_hall" / "baseline"
BASE_IMAGES_DIR = BASE_DIR / "images"
CANDIDATES_CSV = BASE_DIR / "candidate_images_variant_e.csv"

VAR_E_DIR = PROJECT_ROOT / "experiments" / "graham_hall" / "variant_e"
IMAGES_DIR = VAR_E_DIR / "images"
PREVIEWS_DIR = VAR_E_DIR / "previews"
MANIFEST_PATH = VAR_E_DIR / "enhancement_manifest.json"
ENH_LOG_CSV = VAR_E_DIR / "enhancement_log.csv"
LOG_PATH = VAR_E_DIR / "enhancement.log"

CLIP_LIMIT = 1.2
TILE_GRID_SIZE = (8, 8)


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("graham_hall_variant_e_gen")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def apply_clahe_bgr(bgr_img: np.ndarray, clip_limit: float = 1.2, tile_grid: tuple = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l)
    merged_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def main():
    logger = setup_logging(LOG_PATH)
    logger.info("=" * 70)
    logger.info("GRAHAM HALL VARIANT E IMAGE GENERATION - START")
    logger.info(f"Source: {BASE_IMAGES_DIR}")
    logger.info(f"Destination: {IMAGES_DIR}")
    logger.info(f"CLAHE Settings: clipLimit={CLIP_LIMIT}, tileGridSize={TILE_GRID_SIZE}")
    logger.info("=" * 70)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load candidates satisfying dual criteria
    candidates = {}
    if not CANDIDATES_CSV.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {CANDIDATES_CSV}")

    with open(CANDIDATES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sift_count = int(row["SIFT_Features"])
            contrast = float(row["Contrast_StdDev"])
            # Require SIFT > 0 to ensure the image was processed in baseline
            if contrast < 52.0 and 0 < sift_count < 10000:
                candidates[row["Image"]] = {"contrast": contrast, "sift": sift_count}

    candidate_names = set(candidates.keys())
    logger.info(f"Verified {len(candidate_names)} candidate images satisfying dual criteria: {sorted(list(candidate_names))}")

    # 2. Process all images
    all_base_files = sorted(list(BASE_IMAGES_DIR.glob("*.jpg")) + list(BASE_IMAGES_DIR.glob("*.JPG")), key=lambda x: x.name)
    seen = set()
    unique_base_files = [f for f in all_base_files if not (f.name.upper() in seen or seen.add(f.name.upper()))]

    if len(unique_base_files) != 270:
        raise ValueError(f"Expected 270 baseline images, found {len(unique_base_files)}")

    log_rows = []
    enhanced_count = 0
    unchanged_count = 0

    for idx, f in enumerate(unique_base_files):
        dst_path = IMAGES_DIR / f.name
        is_candidate = f.name in candidate_names

        if is_candidate:
            bgr = cv2.imread(str(f))
            if bgr is None:
                raise ValueError(f"Failed to read image: {f}")

            enhanced_bgr = apply_clahe_bgr(bgr, clip_limit=CLIP_LIMIT, tile_grid=TILE_GRID_SIZE)

            # Save enhanced image with maximum JPEG quality
            cv2.imwrite(str(dst_path), enhanced_bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])
            enhanced_count += 1

            # Generate side-by-side preview crop (center 1200x800)
            h, w = bgr.shape[:2]
            ch_start, cw_start = max(0, h // 2 - 400), max(0, w // 2 - 600)
            crop_orig = bgr[ch_start:ch_start + 800, cw_start:cw_start + 1200]
            crop_enh = enhanced_bgr[ch_start:ch_start + 800, cw_start:cw_start + 1200]
            preview = np.hstack((crop_orig, crop_enh))
            preview_path = PREVIEWS_DIR / f"preview_{f.stem}.jpg"
            cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 95])

            c_orig = candidates[f.name]["contrast"]
            c_enh = round(float(np.std(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY))), 2)

            log_rows.append([f.name, "Enhanced", c_orig, c_enh, candidates[f.name]["sift"], f"clip={CLIP_LIMIT}, grid={TILE_GRID_SIZE}"])
            logger.info(f"[{enhanced_count}/{len(candidate_names)}] Enhanced: {f.name} (Contrast: {c_orig} -> {c_enh}, SIFT: {candidates[f.name]['sift']})")
        else:
            shutil.copy2(f, dst_path)
            unchanged_count += 1
            log_rows.append([f.name, "Unchanged", "-", "-", "-", "Copied unmodified"])

    logger.info(f"Image generation complete: {enhanced_count} enhanced, {unchanged_count} unchanged (Total: {enhanced_count + unchanged_count}).")

    # 3. Strict Image-Set Validation
    logger.info("Executing strict image-set validation...")
    dest_files = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.JPG")), key=lambda x: x.name)
    seen_dest = set()
    unique_dest_files = [f for f in dest_files if not (f.name.upper() in seen_dest or seen_dest.add(f.name.upper()))]

    if len(unique_dest_files) != 270:
        raise ValueError(f"Output count mismatch: expected 270, found {len(unique_dest_files)}")

    byte_matches = 0
    enhanced_verified = 0

    for sf, df in zip(unique_base_files, unique_dest_files):
        h_src = hashlib.md5(sf.read_bytes()).hexdigest()
        h_dst = hashlib.md5(df.read_bytes()).hexdigest()

        if df.name in candidate_names:
            if h_src == h_dst:
                raise ValueError(f"Enhanced image is unexpectedly identical to source: {df.name}")
            im = cv2.imread(str(df))
            im_src = cv2.imread(str(sf))
            if im is None or im.shape != im_src.shape:
                raise ValueError(f"Enhanced image shape/decode error on {df.name}")
            enhanced_verified += 1
        else:
            if h_src != h_dst:
                raise ValueError(f"Unchanged image byte mismatch on {df.name}!")
            byte_matches += 1

    logger.info(f"[PASS] 100% byte-identity verified for all {byte_matches} unchanged images.")
    logger.info(f"[PASS] Exact dimension match & valid decoding verified for all {enhanced_verified} enhanced images.")

    # 4. Save manifest and log
    manifest = {
        "dataset": "graham-hall-270-exterior-subset",
        "scope": "270-image exterior sequence (IMG_0348.jpg to IMG_0617.jpg)",
        "total_images": 270,
        "enhanced_images_count": enhanced_count,
        "unchanged_images_count": unchanged_count,
        "clip_limit": CLIP_LIMIT,
        "tile_grid_size": list(TILE_GRID_SIZE),
        "selection_criteria": {
            "contrast_stddev_threshold": 52.0,
            "sift_feature_count_threshold": 10000,
        },
        "enhanced_images": sorted(list(candidate_names)),
        "previews_directory": str(PREVIEWS_DIR.relative_to(PROJECT_ROOT)),
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved enhancement manifest to: {MANIFEST_PATH}")

    with open(ENH_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Status", "Contrast_Original", "Contrast_Enhanced", "Baseline_SIFT", "Parameters"])
        for row in log_rows:
            writer.writerow(row)
    logger.info(f"Saved enhancement log to: {ENH_LOG_CSV}")

    logger.info("=" * 70)
    logger.info("GRAHAM HALL VARIANT E GENERATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
