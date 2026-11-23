"""
Phase 6 — Ablation Enhancement Generator

Generates three CLAHE-intensity variants for South Building:
  Variant A: clipLimit=2.0 (current, for re-validation)
  Variant B: clipLimit=1.5 (mild)
  Variant C: clipLimit=1.2 (conservative)

All variants use the same 28 selected images (Contrast_StdDev < 60.0)
and the same tileGridSize=(8,8). Unchanged images are copied identically
across all variants.

Each variant is saved to a distinct output directory.
Original images and existing experiment results are never modified.

Usage:
    python experiments/phase6/run_ablation_enhancement.py
"""

import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "south-building"
QUALITY_CSV = PROJECT_ROOT / "experiments" / "quality_analysis" / "south-building" / "image_quality.csv"
PHASE6_DIR = PROJECT_ROOT / "experiments" / "phase6"

# Ablation configurations: (name, clipLimit, tileGridSize)
ABLATION_VARIANTS = [
    ("variant_a_clip2.0", 2.0, (8, 8)),
    ("variant_b_clip1.5", 1.5, (8, 8)),
    ("variant_c_clip1.2", 1.2, (8, 8)),
]

# Enhancement thresholds (same as Phase 3A / Phase 5)
BLUR_THRESHOLD = 80.0
CONTRAST_THRESHOLD = 60.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ablation_enhancement")
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


# ---------------------------------------------------------------------------
# CLAHE implementation
# ---------------------------------------------------------------------------

def apply_clahe_bgr(img_bgr: np.ndarray, clip_limit: float, tile_grid_size: Tuple[int, int]) -> np.ndarray:
    """Apply CLAHE in LAB color space (Luminance channel only)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Quality CSV loading
# ---------------------------------------------------------------------------

def load_quality_csv(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load per-image quality metrics."""
    metrics = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["Image"]] = {
                "Blur": float(row["Blur_LaplacianVar"]),
                "Contrast": float(row["Contrast_StdDev"]),
                "Brightness": float(row["Brightness_Mean"]),
                "Noise": float(row["Noise_Sigma"]),
            }
    return metrics


# ---------------------------------------------------------------------------
# Side-by-side preview
# ---------------------------------------------------------------------------

def create_side_by_side_preview(
    orig_path: Path, enh_path: Path, variant_label: str, save_path: Path
) -> None:
    """Generate a side-by-side comparison image."""
    orig = cv2.imread(str(orig_path))
    enh = cv2.imread(str(enh_path))
    if orig is None or enh is None:
        return

    h, w = orig.shape[:2]
    target_h = 1080
    scale = target_h / float(h)
    new_w = int(w * scale)

    orig_small = cv2.resize(orig, (new_w, target_h), interpolation=cv2.INTER_AREA)
    enh_small = cv2.resize(enh, (new_w, target_h), interpolation=cv2.INTER_AREA)

    header_h = 60
    header = np.zeros((header_h, new_w * 2, 3), dtype=np.uint8)
    cv2.putText(header, f"ORIGINAL: {orig_path.name}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, f"ENHANCED [{variant_label}]: {enh_path.name}", (new_w + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    side_by_side = np.hstack([orig_small, enh_small])
    preview_img = np.vstack([header, side_by_side])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), preview_img)


# ---------------------------------------------------------------------------
# Multi-variant comparison strip
# ---------------------------------------------------------------------------

def create_multi_variant_strip(
    orig_path: Path,
    variant_paths: List[Tuple[str, Path]],
    save_path: Path
) -> None:
    """Generate a horizontal strip: Original | Variant A | Variant B | Variant C."""
    orig = cv2.imread(str(orig_path))
    if orig is None:
        return

    h, w = orig.shape[:2]
    target_h = 720
    scale = target_h / float(h)
    new_w = int(w * scale)

    panels = [cv2.resize(orig, (new_w, target_h), interpolation=cv2.INTER_AREA)]
    labels = ["ORIGINAL"]

    for label, vp in variant_paths:
        img = cv2.imread(str(vp))
        if img is not None:
            panels.append(cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA))
            labels.append(label)

    header_h = 50
    total_w = new_w * len(panels)
    header = np.zeros((header_h, total_w, 3), dtype=np.uint8)
    for i, lbl in enumerate(labels):
        color = (255, 255, 255) if i == 0 else (0, 255, 255)
        cv2.putText(header, lbl, (i * new_w + 10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    strip = np.hstack(panels)
    result = np.vstack([header, strip])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    master_log = setup_logging(PHASE6_DIR / "ablation_enhancement.log")
    master_log.info("=" * 70)
    master_log.info("PHASE 6: ABLATION ENHANCEMENT GENERATOR — START")
    master_log.info("=" * 70)

    # Load quality metrics
    quality_map = load_quality_csv(QUALITY_CSV)

    # Find source images
    src_images_dir = DATASET_DIR / "images"
    image_files = sorted([
        f for f in src_images_dir.glob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    master_log.info(f"Source images: {len(image_files)} in {src_images_dir}")
    master_log.info(f"Quality CSV: {QUALITY_CSV}")
    master_log.info(f"Blur threshold: {BLUR_THRESHOLD}")
    master_log.info(f"Contrast threshold: {CONTRAST_THRESHOLD}")

    # Determine which images need CLAHE (same selection for all variants)
    clahe_images = []
    for img_path in image_files:
        metrics = quality_map.get(img_path.name)
        if metrics and metrics["Contrast"] < CONTRAST_THRESHOLD:
            clahe_images.append(img_path.name)

    master_log.info(f"Images selected for CLAHE: {len(clahe_images)}")

    # Process each variant
    variant_image_dirs = {}  # variant_name -> images_dir

    for variant_name, clip_limit, tile_grid_size in ABLATION_VARIANTS:
        master_log.info(f"\n{'='*70}")
        master_log.info(f"Processing: {variant_name} (clipLimit={clip_limit}, tileGrid={tile_grid_size})")
        master_log.info(f"{'='*70}")

        variant_dir = PHASE6_DIR / variant_name
        variant_images_dir = variant_dir / "images"
        variant_previews_dir = variant_dir / "previews"
        variant_images_dir.mkdir(parents=True, exist_ok=True)
        variant_previews_dir.mkdir(parents=True, exist_ok=True)

        variant_image_dirs[variant_name] = variant_images_dir

        log_rows = []
        preview_samples = []
        counts = {"unchanged": 0, "clahe_only": 0, "total": len(image_files)}
        start_time = time.time()

        for img_path in image_files:
            img_name = img_path.name
            dst_path = variant_images_dir / img_name
            metrics = quality_map.get(img_name)

            if metrics is None or metrics["Contrast"] >= CONTRAST_THRESHOLD:
                # Unchanged — copy original
                shutil.copy2(img_path, dst_path)
                counts["unchanged"] += 1
                log_rows.append({
                    "Image": img_name,
                    "Blur": metrics["Blur"] if metrics else "N/A",
                    "Contrast": metrics["Contrast"] if metrics else "N/A",
                    "Noise_Sigma": metrics["Noise"] if metrics else "N/A",
                    "Category": "unchanged",
                    "clipLimit": "N/A",
                    "Operations": "None (Copied Unchanged)"
                })
            else:
                # CLAHE enhancement
                img_bgr = cv2.imread(str(img_path))
                enhanced = apply_clahe_bgr(img_bgr, clip_limit, tile_grid_size)
                cv2.imwrite(str(dst_path), enhanced)
                counts["clahe_only"] += 1

                log_rows.append({
                    "Image": img_name,
                    "Blur": metrics["Blur"],
                    "Contrast": metrics["Contrast"],
                    "Noise_Sigma": metrics["Noise"],
                    "Category": "clahe_only",
                    "clipLimit": clip_limit,
                    "Operations": f"CLAHE(clip={clip_limit}, tile={tile_grid_size})"
                })

                # Collect preview samples (up to 5)
                if len(preview_samples) < 5:
                    preview_samples.append((img_path, dst_path))

        elapsed = time.time() - start_time

        # Save enhancement log CSV
        log_csv_path = variant_dir / "enhancement_log.csv"
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Image", "Blur", "Contrast", "Noise_Sigma",
                "Category", "clipLimit", "Operations"
            ])
            writer.writeheader()
            writer.writerows(log_rows)

        # Save enhancement summary JSON
        summary = {
            "variant": variant_name,
            "dataset": "south-building",
            "parameters": {
                "clip_limit": clip_limit,
                "tile_grid_size": list(tile_grid_size),
                "blur_threshold": BLUR_THRESHOLD,
                "contrast_threshold": CONTRAST_THRESHOLD,
            },
            "counts": counts,
            "enhancement_time_s": round(elapsed, 1),
            "clahe_images": clahe_images,
            "verifications": {
                "input_count": len(image_files),
                "output_count": len(list(variant_images_dir.glob("*"))),
                "match": len(image_files) == len(list(variant_images_dir.glob("*")))
            }
        }
        with open(variant_dir / "enhancement_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Generate previews
        for orig_p, enh_p in preview_samples:
            preview_path = variant_previews_dir / f"preview_{orig_p.name}"
            create_side_by_side_preview(orig_p, enh_p, variant_name, preview_path)

        # Verify output count
        output_count = len(list(variant_images_dir.glob("*")))
        status = "PASS" if output_count == len(image_files) else "FAIL"

        master_log.info(f"  Variant: {variant_name}")
        master_log.info(f"  clipLimit: {clip_limit}")
        master_log.info(f"  tileGridSize: {tile_grid_size}")
        master_log.info(f"  CLAHE images: {counts['clahe_only']}")
        master_log.info(f"  Unchanged: {counts['unchanged']}")
        master_log.info(f"  Output images: {output_count}")
        master_log.info(f"  Enhancement time: {elapsed:.1f}s")
        master_log.info(f"  Verification: {status}")

    # Generate multi-variant comparison strips for representative images
    master_log.info("\nGenerating multi-variant comparison strips...")
    comparison_dir = PHASE6_DIR / "comparison_previews"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Use first 5 CLAHE images for comparison
    representative = clahe_images[:5]
    for img_name in representative:
        orig_path = src_images_dir / img_name
        variant_paths = []
        for vname, _, _ in ABLATION_VARIANTS:
            vp = variant_image_dirs[vname] / img_name
            variant_paths.append((vname.replace("variant_", "").upper(), vp))

        strip_path = comparison_dir / f"compare_{img_name}"
        create_multi_variant_strip(orig_path, variant_paths, strip_path)
        master_log.info(f"  Saved comparison strip: {strip_path.name}")

    master_log.info("=" * 70)
    master_log.info("PHASE 6: ABLATION ENHANCEMENT GENERATOR — COMPLETE")
    master_log.info("=" * 70)


if __name__ == "__main__":
    main()
