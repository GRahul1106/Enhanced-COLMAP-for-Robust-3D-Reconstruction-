"""
adaptive_enhancement.py

Phase 3A: Quality-Guided Adaptive Image Enhancement for COLMAP 3D Reconstruction.

Applies selective image processing:
  - CLAHE (in LAB color space) for low-contrast images (Contrast < contrast_threshold)
  - Unsharp Mask Sharpening for blurry images (Blur < blur_threshold)
  - CLAHE + Sharpening for images satisfying both conditions
  - Direct copy (Unchanged) for normal quality images

No denoising is applied in Phase 3A. Output images, logs, summary, and previews
are written to:
    experiments/enhanced/<dataset_name>/

Usage:
    python src/enhanced/adaptive_enhancement.py \
        --dataset "datasets/gerrard-hall" \
        --quality-csv "experiments/quality_analysis/gerrard-hall/image_quality.csv" \
        --output "experiments/enhanced/gerrard-hall" \
        --blur-threshold 80.0 \
        --contrast-threshold 60.0
"""

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("adaptive_enhancement")
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


def apply_clahe_bgr(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE in LAB color space (Luminance channel only)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(img_bgr: np.ndarray, alpha: float = 0.5, sigma: float = 1.0) -> np.ndarray:
    """Apply Unsharp Masking for controlled high-frequency edge sharpening."""
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + alpha, blurred, -alpha, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def load_quality_csv(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load per-image quality metrics from CSV into a dictionary keyed by image name."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Quality metrics CSV not found at: {csv_path}")

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


def create_side_by_side_preview(orig_path: Path, enh_path: Path, category: str, save_path: Path) -> None:
    """Generate a side-by-side comparison image [ Original | Enhanced ] with text label banner."""
    orig = cv2.imread(str(orig_path))
    enh = cv2.imread(str(enh_path))

    if orig is None or enh is None:
        return

    # Resize to max height 1080 for lightweight preview
    h, w = orig.shape[:2]
    target_h = 1080
    scale = target_h / float(h)
    new_w = int(w * scale)

    orig_small = cv2.resize(orig, (new_w, target_h), interpolation=cv2.INTER_AREA)
    enh_small = cv2.resize(enh, (new_w, target_h), interpolation=cv2.INTER_AREA)

    # Add header text banner
    header_h = 60
    header = np.zeros((header_h, new_w * 2, 3), dtype=np.uint8)

    cv2.putText(header, f"ORIGINAL: {orig_path.name}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, f"ENHANCED [{category.upper()}]: {enh_path.name}", (new_w + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    side_by_side = np.hstack([orig_small, enh_small])
    preview_img = np.vstack([header, side_by_side])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), preview_img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3A: Adaptive Quality-Guided Enhancement")
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset folder (e.g. datasets/gerrard-hall)")
    parser.add_argument("--quality-csv", required=True, type=Path, help="Path to image_quality.csv")
    parser.add_argument("--output", required=True, type=Path, help="Output folder (e.g. experiments/enhanced/gerrard-hall)")
    parser.add_argument("--blur-threshold", type=float, default=80.0, help="Laplacian variance threshold for blur")
    parser.add_argument("--contrast-threshold", type=float, default=60.0, help="StdDev threshold for contrast")

    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    quality_csv_path = args.quality_csv.resolve()
    output_dir = args.output.resolve()

    output_images_dir = output_dir / "images"
    previews_dir = output_dir / "previews"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / "enhancement.log")

    logger.info("=" * 70)
    logger.info("PHASE 3A: ADAPTIVE IMAGE ENHANCEMENT — START")
    logger.info("=" * 70)
    logger.info(f"Dataset:            {dataset_dir}")
    logger.info(f"Quality CSV:        {quality_csv_path}")
    logger.info(f"Output Images:      {output_images_dir}")
    logger.info(f"Blur Threshold:     {args.blur_threshold}")
    logger.info(f"Contrast Threshold: {args.contrast_threshold}")

    quality_map = load_quality_csv(quality_csv_path)

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    src_images_dir = dataset_dir / "images"
    image_files = sorted([f for f in src_images_dir.glob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])

    log_rows = []
    category_samples: Dict[str, List[Tuple[Path, Path]]] = {
        "clahe_only": [],
        "sharpen_only": [],
        "clahe_and_sharpen": [],
    }

    counts = {
        "unchanged_count": 0,
        "clahe_only_count": 0,
        "sharpen_only_count": 0,
        "both_clahe_and_sharpened_count": 0,
        "total_images": len(image_files)
    }

    for idx, img_path in enumerate(image_files, 1):
        img_name = img_path.name
        metrics = quality_map.get(img_name)
        dst_path = output_images_dir / img_name

        if metrics is None:
            logger.warning(f"No metrics found for {img_name}, copying unchanged.")
            shutil.copy2(img_path, dst_path)
            counts["unchanged_count"] += 1
            log_rows.append({
                "Image": img_name,
                "Blur": "N/A",
                "Contrast": "N/A",
                "Noise_Sigma": "N/A",
                "Category": "unchanged",
                "Operations_Applied": "None (Copied Unchanged - Missing Metrics)",
                "Output_Path": str(dst_path)
            })
            continue

        blur_val = metrics["Blur"]
        contrast_val = metrics["Contrast"]
        noise_val = metrics["Noise"]

        is_low_contrast = contrast_val < args.contrast_threshold
        is_blurry = blur_val < args.blur_threshold

        # Mutually exclusive categorization
        if is_low_contrast and is_blurry:
            category = "clahe_and_sharpen"
            counts["both_clahe_and_sharpened_count"] += 1
            ops = ["CLAHE", "Sharpen"]
        elif is_low_contrast:
            category = "clahe_only"
            counts["clahe_only_count"] += 1
            ops = ["CLAHE"]
        elif is_blurry:
            category = "sharpen_only"
            counts["sharpen_only_count"] += 1
            ops = ["Sharpen"]
        else:
            category = "unchanged"
            counts["unchanged_count"] += 1
            ops = []

        # Process image
        if not ops:
            shutil.copy2(img_path, dst_path)
            ops_str = "None (Copied Unchanged)"
        else:
            img_bgr = cv2.imread(str(img_path))
            if "CLAHE" in ops:
                img_bgr = apply_clahe_bgr(img_bgr)
            if "Sharpen" in ops:
                img_bgr = apply_unsharp_mask(img_bgr)

            cv2.imwrite(str(dst_path), img_bgr)
            ops_str = " + ".join(ops)

            # Record sample for preview generation
            if len(category_samples[category]) < 3:
                category_samples[category].append((img_path, dst_path))

        log_rows.append({
            "Image": img_name,
            "Blur": blur_val,
            "Contrast": contrast_val,
            "Noise_Sigma": noise_val,
            "Category": category,
            "Operations_Applied": ops_str,
            "Output_Path": str(dst_path)
        })

    # Save log CSV
    log_csv_path = output_dir / "enhancement_log.csv"
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "Blur", "Contrast", "Noise_Sigma", "Category", "Operations_Applied", "Output_Path"])
        writer.writeheader()
        writer.writerows(log_rows)

    # Category sum calculation
    category_sum = (counts["unchanged_count"] + counts["clahe_only_count"] +
                    counts["sharpen_only_count"] + counts["both_clahe_and_sharpened_count"])

    # Output files count
    output_files = list(output_images_dir.glob("*"))

    # Save summary JSON
    summary_json_path = output_dir / "enhancement_summary.json"
    summary_data = {
        "dataset": dataset_dir.name,
        "thresholds": {
            "blur_threshold": args.blur_threshold,
            "contrast_threshold": args.contrast_threshold,
        },
        "counts": counts,
        "verifications": {
            "input_image_count": len(image_files),
            "output_image_count": len(output_files),
            "csv_row_count": len(log_rows),
            "category_sum": category_sum,
            "is_input_equal_output": len(image_files) == len(output_files),
            "is_input_equal_csv_rows": len(image_files) == len(log_rows),
            "is_category_sum_valid": category_sum == len(image_files),
            "verification_passed": (
                len(image_files) == len(output_files) and
                len(image_files) == len(log_rows) and
                category_sum == len(image_files)
            )
        }
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Generate side-by-side previews
    logger.info("Generating side-by-side comparison previews...")
    preview_count = 0
    for cat, samples in category_samples.items():
        for orig_p, enh_p in samples:
            preview_save_path = previews_dir / f"preview_{cat}_{orig_p.name}"
            create_side_by_side_preview(orig_p, enh_p, cat, preview_save_path)
            preview_count += 1

    logger.info(f"Generated {preview_count} side-by-side preview panels in {previews_dir}")

    # Output count verification assertions
    if not summary_data["verifications"]["verification_passed"]:
        logger.error("VERIFICATION FAILURE: Validation checks failed!")
        logger.error(f"  Input Count: {len(image_files)}, Output Count: {len(output_files)}, CSV Rows: {len(log_rows)}, Category Sum: {category_sum}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("PHASE 3A: ADAPTIVE ENHANCEMENT COMPLETE")
    logger.info(f"  VERIFICATION STATUS: SUCCESSFUL ✅")
    logger.info(f"  Input Images:           {len(image_files)}")
    logger.info(f"  Output Images:          {len(output_files)}")
    logger.info(f"  CSV Log Rows:           {len(log_rows)}")
    logger.info(f"  Category Sum:           {category_sum}")
    logger.info(f"  Unchanged Images:       {counts['unchanged_count']}")
    logger.info(f"  CLAHE Only:             {counts['clahe_only_count']}")
    logger.info(f"  Sharpened Only:         {counts['sharpen_only_count']}")
    logger.info(f"  Both CLAHE & Sharpened: {counts['both_clahe_and_sharpened_count']}")
    logger.info(f"  Previews Generated:     {preview_count}")
    logger.info(f"  Log CSV:                {log_csv_path}")
    logger.info(f"  Summary JSON:           {summary_json_path}")
    logger.info(f"  Previews Directory:     {previews_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
