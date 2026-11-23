"""
analyze_image_quality.py

Performs Image Quality Assessment (IQA) on a dataset of images without modifying
any input files. Evaluates key image metrics for 3D reconstruction suitability:

1. Blur / Sharpness: Laplacian variance (high variance = sharp, low = blurry)
2. Contrast: Standard deviation of pixel intensities
3. Brightness / Exposure: Mean pixel intensity
4. Noise: Fast noise standard deviation estimation using Laplacian/Immerkaer residual operator

Generates outputs under experiments/quality_analysis/<dataset_name>/:
  - image_quality.csv       Per-image quality metrics table
  - quality_summary.json    Dataset summary stats (mean, std, min, percentiles, max)
  - plots/                  Visual distribution plots and histograms

Usage:
    python src/enhanced/analyze_image_quality.py \
        --dataset "datasets/gerrard-hall" \
        --output  "experiments/quality_analysis/gerrard-hall"
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    """Configure logger to write to both stdout and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("image_quality_analysis")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-runs
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
# Core Quality Metric Calculations
# ---------------------------------------------------------------------------

def estimate_blur(gray: np.ndarray) -> float:
    """
    Calculate image sharpness/blur via Laplacian variance.
    High value -> sharp image; Low value -> blurry image.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def estimate_contrast(gray: np.ndarray) -> float:
    """
    Calculate contrast as the standard deviation of pixel intensities.
    High value -> high contrast; Low value -> flat/low contrast.
    """
    return float(np.std(gray))


def estimate_brightness(gray: np.ndarray) -> float:
    """
    Calculate mean intensity as exposure/brightness measure (0 to 255).
    Low (<60) -> underexposed/dark; High (>190) -> overexposed/bright.
    """
    return float(np.mean(gray))


def estimate_noise(gray: np.ndarray) -> float:
    """
    Estimate additive Gaussian noise sigma using Immerkaer's fast filter method:
    Immerkaer, J. "Fast Noise Variance Estimation", Computer Vision and Image
    Understanding, 1996.

    Uses kernel N = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]].
    """
    H, W = gray.shape
    if H < 3 or W < 3:
        return 0.0

    kernel = np.array([[1, -2, 1],
                       [-2, 4, -2],
                       [1, -2, 1]], dtype=np.float64)

    sigma = np.sum(np.abs(cv2.filter2D(gray.astype(np.float64), -1, kernel)))
    sigma *= np.sqrt(0.5 * np.pi) / (6.0 * (W - 2) * (H - 2))
    return float(sigma)


def analyze_single_image(img_path: Path) -> Dict[str, float]:
    """Load an image and compute all quality metrics."""
    # Read image in grayscale mode directly for efficiency
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image file: {img_path}")

    height, width = gray.shape

    blur = estimate_blur(gray)
    contrast = estimate_contrast(gray)
    brightness = estimate_brightness(gray)
    noise = estimate_noise(gray)

    return {
        "Image": img_path.name,
        "Width": width,
        "Height": height,
        "Blur_LaplacianVar": round(blur, 2),
        "Contrast_StdDev": round(contrast, 2),
        "Brightness_Mean": round(brightness, 2),
        "Noise_Sigma": round(noise, 2),
    }


# ---------------------------------------------------------------------------
# Dataset Scanning Helper
# ---------------------------------------------------------------------------

def locate_images(dataset_dir: Path, logger: logging.Logger) -> Tuple[Path, List[Path]]:
    """
    Find all images inside dataset_dir/images (or subdirectories).
    """
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    images_dir = dataset_dir / "images"
    if not images_dir.exists() or not images_dir.is_dir():
        if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in dataset_dir.iterdir() if f.is_file()):
            images_dir = dataset_dir
        else:
            raise FileNotFoundError(f"No images directory found under {dataset_dir}")

    # Search recursively for image files
    image_files = sorted([
        f for f in images_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        raise FileNotFoundError(f"No valid image files found in {images_dir}")

    logger.info(f"Located {len(image_files)} images in: {images_dir}")
    return images_dir, image_files


# ---------------------------------------------------------------------------
# Visualization Generator
# ---------------------------------------------------------------------------

def generate_distribution_plots(results: List[Dict], plots_dir: Path, logger: logging.Logger) -> None:
    """Generate distribution histograms and boxplots for all metrics."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("Blur_LaplacianVar", "Blur (Laplacian Variance)", "higher = sharper", "teal"),
        ("Contrast_StdDev", "Contrast (Standard Deviation)", "higher = more contrast", "coral"),
        ("Brightness_Mean", "Brightness (Mean Intensity)", "range [0, 255]", "goldenrod"),
        ("Noise_Sigma", "Noise Estimation (Sigma)", "lower = cleaner", "mediumpurple"),
    ]

    # Individual metric histograms
    for key, title, subtitle, color in metrics:
        values = [r[key] for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=20, color=color, edgecolor="black", alpha=0.75)
        ax.axvline(np.mean(values), color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {np.mean(values):.2f}")
        ax.axvline(np.median(values), color="blue", linestyle="dotted", linewidth=1.5, label=f"Median: {np.median(values):.2f}")

        ax.set_title(f"Dataset Distribution: {title}\n({subtitle})", fontsize=12, fontweight="bold")
        ax.set_xlabel(key, fontsize=10)
        ax.set_ylabel("Image Count", fontsize=10)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        plot_path = plots_dir / f"{key.lower()}_distribution.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.debug(f"Saved plot: {plot_path.name}")

    # Overview 2x2 grid plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Image Quality Overview — Gerrard Hall Baseline Analysis", fontsize=14, fontweight="bold")

    for idx, (key, title, _, color) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [r[key] for r in results]
        ax.hist(values, bins=20, color=color, edgecolor="black", alpha=0.75)
        ax.axvline(np.mean(values), color="red", linestyle="--", label=f"Mean ({np.mean(values):.1f})")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(key, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    overview_path = plots_dir / "quality_overview_matrix.png"
    fig.savefig(overview_path, dpi=150)
    plt.close(fig)
    logger.info(f"Overview plot matrix generated: {overview_path.name}")


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Image Quality Analysis for COLMAP Datasets"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to dataset directory (e.g. datasets/gerrard-hall)"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output directory (e.g. experiments/quality_analysis/gerrard-hall)"
    )

    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "quality_analysis.log"
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("PHASE 2: IMAGE QUALITY ANALYSIS — START")
    logger.info("=" * 70)
    logger.info(f"Dataset Path: {dataset_dir}")
    logger.info(f"Output Path:  {output_dir}")

    _, image_files = locate_images(dataset_dir, logger)

    results: List[Dict] = []
    for idx, img_path in enumerate(image_files, 1):
        try:
            metrics = analyze_single_image(img_path)
            results.append(metrics)
            if idx % 20 == 0 or idx == len(image_files):
                logger.info(f"Analyzed [{idx}/{len(image_files)}] images...")
        except Exception as e:
            logger.error(f"Failed to analyze {img_path.name}: {e}")

    if not results:
        logger.error("No images were analyzed successfully.")
        sys.exit(1)

    # 1. Write image_quality.csv
    csv_path = output_dir / "image_quality.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Saved CSV results: {csv_path}")

    # 2. Compute summary statistics and write quality_summary.json
    summary_stats = {
        "dataset_name": dataset_dir.name,
        "total_images": len(results),
        "metrics_summary": {}
    }

    numeric_keys = ["Blur_LaplacianVar", "Contrast_StdDev", "Brightness_Mean", "Noise_Sigma"]
    for key in numeric_keys:
        vals = [r[key] for r in results]
        summary_stats["metrics_summary"][key] = {
            "mean": round(float(np.mean(vals)), 2),
            "std": round(float(np.std(vals)), 2),
            "min": round(float(np.min(vals)), 2),
            "p25": round(float(np.percentile(vals, 25)), 2),
            "p50_median": round(float(np.median(vals)), 2),
            "p75": round(float(np.percentile(vals, 75)), 2),
            "max": round(float(np.max(vals)), 2),
        }

    json_path = output_dir / "quality_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    logger.info(f"Saved Summary JSON: {json_path}")

    # 3. Generate visual distribution plots
    plots_dir = output_dir / "plots"
    generate_distribution_plots(results, plots_dir, logger)

    logger.info("=" * 70)
    logger.info("PHASE 2: IMAGE QUALITY ANALYSIS — COMPLETE")
    logger.info(f"Results Directory: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
