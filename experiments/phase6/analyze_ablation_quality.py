"""
Phase 6 — Image Quality Statistics Comparison

Compares quality metrics (SIFT keypoints, pixel stats, histograms) across:
  - Original images
  - Variant A (clipLimit=2.0)
  - Variant B (clipLimit=1.5)
  - Variant C (clipLimit=1.2)

For the 28 CLAHE-selected images only, measures:
  1. Mean/std pixel intensity change
  2. Histogram comparison (correlation, chi-square)
  3. SIFT keypoint count per image
  4. Summary statistics table

Saves results to experiments/phase6/quality_comparison/

Usage:
    python experiments/phase6/analyze_ablation_quality.py
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "south-building" / "images"
PHASE6_DIR = PROJECT_ROOT / "experiments" / "phase6"
OUTPUT_DIR = PHASE6_DIR / "quality_comparison"

VARIANTS = [
    ("original", DATASET_DIR),
    ("variant_a_clip2.0", PHASE6_DIR / "variant_a_clip2.0" / "images"),
    ("variant_b_clip1.5", PHASE6_DIR / "variant_b_clip1.5" / "images"),
    ("variant_c_clip1.2", PHASE6_DIR / "variant_c_clip1.2" / "images"),
]


def load_clahe_image_list() -> List[str]:
    """Load list of CLAHE-selected images from variant A summary."""
    summary_path = PHASE6_DIR / "variant_a_clip2.0" / "enhancement_summary.json"
    with open(summary_path, "r") as f:
        data = json.load(f)
    return data["clahe_images"]


def compute_image_stats(img_path: Path) -> Dict:
    """Compute quality statistics for a single image."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Basic stats
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    brightness = mean_intensity

    # Laplacian variance (blur/sharpness)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # Histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_normalized = hist / hist.sum()

    # SIFT keypoints
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    sift_count = len(keypoints)

    # Contrast (standard deviation)
    contrast = std_intensity

    return {
        "mean_intensity": round(mean_intensity, 2),
        "std_intensity": round(std_intensity, 2),
        "blur_laplacian_var": round(blur_score, 2),
        "contrast_stddev": round(contrast, 2),
        "sift_keypoints": sift_count,
        "histogram": hist_normalized,
        "dimensions": f"{img_bgr.shape[1]}x{img_bgr.shape[0]}",
    }


def histogram_correlation(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute histogram correlation between two normalized histograms."""
    return float(cv2.compareHist(
        hist1.astype(np.float32).reshape(-1, 1),
        hist2.astype(np.float32).reshape(-1, 1),
        cv2.HISTCMP_CORREL
    ))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clahe_images = load_clahe_image_list()
    print(f"Analyzing {len(clahe_images)} CLAHE-selected images across {len(VARIANTS)} variants")

    # Collect per-image, per-variant stats
    all_results = []  # list of dicts with image, variant, stats

    for img_name in clahe_images:
        row = {"Image": img_name}
        for variant_name, variant_dir in VARIANTS:
            img_path = variant_dir / img_name
            if not img_path.exists():
                print(f"  WARNING: {img_path} not found")
                continue

            stats = compute_image_stats(img_path)
            if stats is None:
                continue

            prefix = variant_name.replace("variant_", "").replace(".", "")
            row[f"{prefix}_mean"] = stats["mean_intensity"]
            row[f"{prefix}_std"] = stats["std_intensity"]
            row[f"{prefix}_blur"] = stats["blur_laplacian_var"]
            row[f"{prefix}_contrast"] = stats["contrast_stddev"]
            row[f"{prefix}_sift"] = stats["sift_keypoints"]

            # Histogram correlation vs original
            if variant_name != "original":
                orig_path = DATASET_DIR / img_name
                orig_stats = compute_image_stats(orig_path)
                if orig_stats:
                    corr = histogram_correlation(orig_stats["histogram"], stats["histogram"])
                    row[f"{prefix}_hist_corr"] = round(corr, 4)

        all_results.append(row)
        print(f"  Analyzed: {img_name}")

    # Save detailed CSV
    if all_results:
        csv_path = OUTPUT_DIR / "quality_comparison.csv"
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSaved detailed comparison: {csv_path}")

    # Compute summary statistics
    summary = {}
    for variant_name, _ in VARIANTS:
        prefix = variant_name.replace("variant_", "").replace(".", "")
        sift_key = f"{prefix}_sift"
        contrast_key = f"{prefix}_contrast"
        blur_key = f"{prefix}_blur"
        mean_key = f"{prefix}_mean"

        sift_vals = [r[sift_key] for r in all_results if sift_key in r]
        contrast_vals = [r[contrast_key] for r in all_results if contrast_key in r]
        blur_vals = [r[blur_key] for r in all_results if blur_key in r]
        mean_vals = [r[mean_key] for r in all_results if mean_key in r]

        summary[variant_name] = {
            "sift_keypoints": {
                "mean": round(np.mean(sift_vals), 1) if sift_vals else None,
                "std": round(np.std(sift_vals), 1) if sift_vals else None,
                "min": int(np.min(sift_vals)) if sift_vals else None,
                "max": int(np.max(sift_vals)) if sift_vals else None,
                "total": int(np.sum(sift_vals)) if sift_vals else None,
            },
            "contrast_stddev": {
                "mean": round(np.mean(contrast_vals), 2) if contrast_vals else None,
                "std": round(np.std(contrast_vals), 2) if contrast_vals else None,
            },
            "blur_laplacian_var": {
                "mean": round(np.mean(blur_vals), 2) if blur_vals else None,
            },
            "mean_intensity": {
                "mean": round(np.mean(mean_vals), 2) if mean_vals else None,
            },
        }

        # Histogram correlation summary
        corr_key = f"{prefix}_hist_corr"
        corr_vals = [r[corr_key] for r in all_results if corr_key in r]
        if corr_vals:
            summary[variant_name]["hist_correlation_vs_original"] = {
                "mean": round(np.mean(corr_vals), 4),
                "min": round(np.min(corr_vals), 4),
                "max": round(np.max(corr_vals), 4),
            }

    summary_path = OUTPUT_DIR / "quality_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    # Generate comparison plots
    _generate_plots(all_results, summary, OUTPUT_DIR)

    print("\nQuality comparison analysis complete!")


def _generate_plots(results: List[Dict], summary: Dict, output_dir: Path):
    """Generate comparison visualizations."""

    # 1. SIFT keypoint comparison bar chart
    variant_labels = ["Original", "Clip=2.0", "Clip=1.5", "Clip=1.2"]
    prefixes = ["original", "a_clip20", "b_clip15", "c_clip12"]
    colors = ["#4A90E2", "#F5A623", "#50E3C2", "#BD10E0"]

    sift_means = []
    sift_stds = []
    for prefix in prefixes:
        key = f"{prefix}_sift"
        vals = [r[key] for r in results if key in r]
        sift_means.append(np.mean(vals) if vals else 0)
        sift_stds.append(np.std(vals) if vals else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(variant_labels))
    bars = ax.bar(x, sift_means, yerr=sift_stds, capsize=5, color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Mean SIFT Keypoints per Image", fontsize=12)
    ax.set_title("SIFT Keypoint Count: Original vs CLAHE Variants\n(28 CLAHE-selected South Building images)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, mean_val in zip(bars, sift_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{mean_val:.0f}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "sift_keypoints_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Per-image SIFT comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    image_names = [r["Image"].replace("P1180", "").replace(".JPG", "") for r in results]
    x = np.arange(len(image_names))
    width = 0.2

    for i, (prefix, label, color) in enumerate(zip(prefixes, variant_labels, colors)):
        key = f"{prefix}_sift"
        vals = [r.get(key, 0) for r in results]
        ax.bar(x + i * width - 1.5 * width, vals, width, label=label, color=color, alpha=0.8)

    ax.set_ylabel("SIFT Keypoints")
    ax.set_title("Per-Image SIFT Keypoint Count Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(image_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "sift_per_image_comparison.png", dpi=150)
    plt.close(fig)

    # 3. Contrast change comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (prefix, label, color) in enumerate(zip(prefixes, variant_labels, colors)):
        key = f"{prefix}_contrast"
        vals = [r.get(key, 0) for r in results]
        ax.bar(x + i * width - 1.5 * width, vals, width, label=label, color=color, alpha=0.8)

    ax.axhline(y=60, color="red", linestyle="--", linewidth=1.5, label="Threshold (60)")
    ax.set_ylabel("Contrast (StdDev)")
    ax.set_title("Contrast After Enhancement: Original vs CLAHE Variants")
    ax.set_xticks(x)
    ax.set_xticklabels(image_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "contrast_comparison.png", dpi=150)
    plt.close(fig)

    print(f"  Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
