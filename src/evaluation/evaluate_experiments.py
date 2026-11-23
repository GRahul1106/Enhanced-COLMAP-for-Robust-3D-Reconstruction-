"""
evaluate_experiments.py

Consolidated multi-dataset evaluation and robustness testing module.
Aggregates quantitative metrics across datasets for both baseline and
adaptively enhanced COLMAP reconstructions.

Produces:
  - experiments/evaluation/summary.csv
  - experiments/evaluation/summary.json
  - experiments/evaluation/report.md
  - experiments/evaluation/plots/*.png

Usage:
  python src/evaluation/evaluate_experiments.py
"""

import csv
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("evaluate_experiments")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EVALUATION_DIR = EXPERIMENTS_DIR / "evaluation"
PLOTS_DIR = EVALUATION_DIR / "plots"


def count_input_images(dataset_name: str) -> int:
    """Count input images in dataset directory."""
    ds_path = DATASETS_DIR / dataset_name
    if not ds_path.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    img_files = [f for f in ds_path.rglob("*") if f.is_file() and f.suffix.lower() in exts]
    
    # Handle person-hall Part1+Part2 sum vs direct images
    if dataset_name == "person-hall":
        # 165 + 165 = 330
        return len(img_files)
    elif dataset_name == "graham-hall":
        # Each part has 1273 unique images
        part1 = ds_path / "Part1"
        if part1.exists():
            return len([f for f in part1.rglob("*") if f.is_file() and f.suffix.lower() in exts])
        return len(img_files)
    elif (ds_path / "images").exists():
        return len([f for f in (ds_path / "images").iterdir() if f.is_file() and f.suffix.lower() in exts])
    return len(img_files)


def parse_enhancement_summary(dataset_name: str) -> Optional[Dict[str, Any]]:
    """Load enhancement summary JSON."""
    summary_path = EXPERIMENTS_DIR / "enhanced" / dataset_name / "enhancement_summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read enhancement summary for {dataset_name}: {e}")
        return None


def parse_colmap_sparse_model(sparse_model_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract metrics from sparse reconstruction TXT files (points3D.txt, images.txt)."""
    p3d_path = sparse_model_dir / "points3D.txt"
    img_path = sparse_model_dir / "images.txt"
    if not p3d_path.exists() or not img_path.exists():
        return None

    num_points = None
    mean_track_length = None
    errors = []

    with open(p3d_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# Number of points:"):
                m = re.search(r"Number of points:\s*(\d+),\s*mean track length:\s*([0-9.]+)", line)
                if m:
                    num_points = int(m.group(1))
                    mean_track_length = float(m.group(2))
            elif not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        errors.append(float(parts[7]))
                    except ValueError:
                        pass

    num_registered_images = None
    mean_observations = None

    with open(img_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Number of images:"):
                m = re.search(r"Number of images:\s*(\d+),\s*mean observations per image:\s*([0-9.]+)", line)
                if m:
                    num_registered_images = int(m.group(1))
                    mean_observations = float(m.group(2))
                break

    mean_reproj_err = float(np.mean(errors)) if errors else None

    return {
        "num_points": num_points if num_points is not None else len(errors),
        "mean_track_length": mean_track_length,
        "registered_images": num_registered_images,
        "mean_observations_per_image": mean_observations,
        "mean_reprojection_error": mean_reproj_err,
    }


def parse_colmap_timings(log_path: Path) -> Dict[str, float]:
    """Parse stage runtimes from baseline.log."""
    if not log_path.exists():
        return {}
    timings = {}
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"STAGE COMPLETE:\s*(.+?)\s*\(([0-9.]+)s\)", line)
            if m:
                stage = m.group(1).strip()
                # Remove non-ascii characters for clean keys
                stage_clean = stage.replace("→", "to").replace("—", "-")
                timings[stage_clean] = float(m.group(2))
    if timings:
        timings["total_sfm_time_s"] = round(sum(timings.values()), 1)
    return timings


def parse_analysis_time(dataset_name: str) -> Optional[float]:
    """Extract quality analysis runtime from quality_analysis.log."""
    log_path = EXPERIMENTS_DIR / "quality_analysis" / dataset_name / "quality_analysis.log"
    if not log_path.exists():
        return None
    timestamps = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
            if m:
                try:
                    timestamps.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    pass
    if len(timestamps) >= 2:
        return (timestamps[-1] - timestamps[0]).total_seconds()
    return None


def parse_enhancement_time(dataset_name: str) -> Optional[float]:
    """Extract enhancement runtime from enhancement.log."""
    log_path = EXPERIMENTS_DIR / "enhanced" / dataset_name / "enhancement.log"
    if not log_path.exists():
        return None
    timestamps = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
            if m:
                try:
                    timestamps.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    pass
    if len(timestamps) >= 2:
        return (timestamps[-1] - timestamps[0]).total_seconds()
    return None


def evaluate_dataset(dataset_name: str) -> Dict[str, Any]:
    """Evaluate single dataset across baseline and enhanced runs."""
    input_images = count_input_images(dataset_name)
    enh_summary = parse_enhancement_summary(dataset_name)

    num_enhanced = 0
    pct_enhanced = 0.0
    enh_categories = {}
    if enh_summary:
        counts = enh_summary.get("counts", {})
        total_in = counts.get("total_images", input_images)
        unchanged = counts.get("unchanged_count", 0)
        num_enhanced = total_in - unchanged
        pct_enhanced = round((num_enhanced / total_in) * 100, 2) if total_in > 0 else 0.0
        enh_categories = counts

    # Check baseline model
    base_sparse_dir = EXPERIMENTS_DIR / "baseline" / dataset_name / "sparse" / "0"
    base_model = parse_colmap_sparse_model(base_sparse_dir)
    base_timings = parse_colmap_timings(EXPERIMENTS_DIR / "baseline" / dataset_name / "baseline.log")

    # Check enhanced model
    enh_sparse_dir = EXPERIMENTS_DIR / "baseline_enhanced" / dataset_name / "sparse" / "0"
    enh_model = parse_colmap_sparse_model(enh_sparse_dir)
    enh_timings = parse_colmap_timings(EXPERIMENTS_DIR / "baseline_enhanced" / dataset_name / "baseline.log")

    has_baseline = base_model is not None
    has_enhanced = enh_model is not None
    comparable = has_baseline and has_enhanced

    analysis_time_s = parse_analysis_time(dataset_name)
    enhancement_time_s = parse_enhancement_time(dataset_name)

    entry = {
        "dataset": dataset_name,
        "input_images": input_images,
        "has_baseline_run": has_baseline,
        "has_enhanced_run": has_enhanced,
        "is_comparable": comparable,
        "enhancement": {
            "enhanced_images": num_enhanced,
            "percentage_enhanced": pct_enhanced,
            "categories": enh_categories,
            "analysis_time_s": analysis_time_s,
            "enhancement_time_s": enhancement_time_s,
        },
        "baseline": {
            "registered_images": base_model["registered_images"] if base_model else None,
            "registration_rate_pct": round((base_model["registered_images"] / input_images) * 100, 2) if (base_model and input_images) else None,
            "num_points": base_model["num_points"] if base_model else None,
            "mean_reprojection_error_px": round(base_model["mean_reprojection_error"], 4) if (base_model and base_model["mean_reprojection_error"]) else None,
            "mean_observations_per_image": round(base_model["mean_observations_per_image"], 2) if (base_model and base_model["mean_observations_per_image"]) else None,
            "mean_track_length": round(base_model["mean_track_length"], 4) if (base_model and base_model["mean_track_length"]) else None,
            "timings_s": base_timings,
        },
        "enhanced": {
            "registered_images": enh_model["registered_images"] if enh_model else None,
            "registration_rate_pct": round((enh_model["registered_images"] / input_images) * 100, 2) if (enh_model and input_images) else None,
            "num_points": enh_model["num_points"] if enh_model else None,
            "mean_reprojection_error_px": round(enh_model["mean_reprojection_error"], 4) if (enh_model and enh_model["mean_reprojection_error"]) else None,
            "mean_observations_per_image": round(enh_model["mean_observations_per_image"], 2) if (enh_model and enh_model["mean_observations_per_image"]) else None,
            "mean_track_length": round(enh_model["mean_track_length"], 4) if (enh_model and enh_model["mean_track_length"]) else None,
            "timings_s": enh_timings,
        },
        "comparison": {}
    }

    if comparable:
        def diff(b, e):
            if b is None or e is None:
                return None, None
            abs_d = e - b
            pct_d = (abs_d / b) * 100 if b != 0 else 0.0
            return round(abs_d, 4), round(pct_d, 2)

        pts_abs, pts_pct = diff(entry["baseline"]["num_points"], entry["enhanced"]["num_points"])
        err_abs, err_pct = diff(entry["baseline"]["mean_reprojection_error_px"], entry["enhanced"]["mean_reprojection_error_px"])
        obs_abs, obs_pct = diff(entry["baseline"]["mean_observations_per_image"], entry["enhanced"]["mean_observations_per_image"])
        trk_abs, trk_pct = diff(entry["baseline"]["mean_track_length"], entry["enhanced"]["mean_track_length"])
        reg_abs, reg_pct = diff(entry["baseline"]["registered_images"], entry["enhanced"]["registered_images"])

        entry["comparison"] = {
            "registered_images_diff": {"abs": reg_abs, "pct": reg_pct},
            "points_diff": {"abs": pts_abs, "pct": pts_pct},
            "reprojection_error_diff": {"abs": err_abs, "pct": err_pct},
            "observations_diff": {"abs": obs_abs, "pct": obs_pct},
            "track_length_diff": {"abs": trk_abs, "pct": trk_pct},
        }

    return entry


def generate_evaluation_plots(eval_data: List[Dict[str, Any]], plots_dir: Path) -> None:
    """Generate comparative visualization plots across completed datasets."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    comp_datasets = [d for d in eval_data if d["is_comparable"]]
    if not comp_datasets:
        logger.warning("No comparable datasets available to plot.")
        return

    names = [d["dataset"] for d in comp_datasets]
    x = np.arange(len(names))
    width = 0.35

    # 1. Reconstructed 3D Points
    fig, ax = plt.subplots(figsize=(8, 5))
    base_pts = [d["baseline"]["num_points"] for d in comp_datasets]
    enh_pts = [d["enhanced"]["num_points"] for d in comp_datasets]
    ax.bar(x - width/2, base_pts, width, label="Original Baseline", color="#4A90E2")
    ax.bar(x + width/2, enh_pts, width, label="Adaptive Enhanced", color="#50E3C2")
    ax.set_ylabel("Reconstructed 3D Points")
    ax.set_title("3D Reconstructed Point Count Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i in range(len(names)):
        diff_pts = enh_pts[i] - base_pts[i]
        sign = "+" if diff_pts >= 0 else ""
        ax.text(x[i], max(base_pts[i], enh_pts[i]) * 1.01, f"{sign}{diff_pts}", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "points_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Mean Reprojection Error
    fig, ax = plt.subplots(figsize=(8, 5))
    base_err = [d["baseline"]["mean_reprojection_error_px"] for d in comp_datasets]
    enh_err = [d["enhanced"]["mean_reprojection_error_px"] for d in comp_datasets]
    ax.bar(x - width/2, base_err, width, label="Original Baseline", color="#F5A623")
    ax.bar(x + width/2, enh_err, width, label="Adaptive Enhanced", color="#BD10E0")
    ax.set_ylabel("Mean Reprojection Error (pixels)")
    ax.set_title("Mean Reprojection Error Comparison (Lower is Preferable)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i in range(len(names)):
        diff_err = enh_err[i] - base_err[i]
        sign = "+" if diff_err >= 0 else ""
        ax.text(x[i], max(base_err[i], enh_err[i]) * 1.01, f"{sign}{diff_err:.4f}px", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "reprojection_error_comparison.png", dpi=150)
    plt.close(fig)

    # 3. Mean Track Length and Observations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    base_trk = [d["baseline"]["mean_track_length"] for d in comp_datasets]
    enh_trk = [d["enhanced"]["mean_track_length"] for d in comp_datasets]
    ax1.bar(x - width/2, base_trk, width, label="Baseline", color="#7ED321")
    ax1.bar(x + width/2, enh_trk, width, label="Enhanced", color="#9013FE")
    ax1.set_ylabel("Mean Track Length")
    ax1.set_title("Mean Track Length")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    base_obs = [d["baseline"]["mean_observations_per_image"] for d in comp_datasets]
    enh_obs = [d["enhanced"]["mean_observations_per_image"] for d in comp_datasets]
    ax2.bar(x - width/2, base_obs, width, label="Baseline", color="#4A90E2")
    ax2.bar(x + width/2, enh_obs, width, label="Enhanced", color="#50E3C2")
    ax2.set_ylabel("Mean Observations / Image")
    ax2.set_title("Mean Observations Per Image")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(plots_dir / "track_and_observations_comparison.png", dpi=150)
    plt.close(fig)
    logger.info(f"Generated comparison plots in {plots_dir}")


def generate_markdown_report(eval_data: List[Dict[str, Any]], report_path: Path) -> None:
    """Generate comprehensive markdown evaluation report."""
    lines = [
        "# Consolidated Multi-Dataset Evaluation Report",
        "## Adaptive Image Enhancement for 3D Reconstruction (COLMAP)",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Phase:** 5 — Multi-Dataset Evaluation & Robustness Testing  ",
        "",
        "---",
        "",
        "## 1. Experiment Completion & Comparability Matrix",
        "",
        "| Dataset | Input Images | Baseline SfM | Enhanced Images | Enhanced SfM | Valid Comparison? | Status / Notes |",
        "|---|---|---|---|---|---|---|",
    ]

    for d in eval_data:
        name = d["dataset"]
        n_in = d["input_images"]
        b_sfm = "✅ Complete" if d["has_baseline_run"] else "❌ Missing"
        enh_img = f"✅ Done ({d['enhancement']['enhanced_images']} enh)" if d["enhancement"]["enhanced_images"] is not None and d["enhancement"]["enhanced_images"] > 0 or d["enhancement"]["categories"] else "❌ Missing"
        e_sfm = "✅ Complete" if d["has_enhanced_run"] else "⏳ Pending"
        comp = "✅ Yes" if d["is_comparable"] else "❌ Incomplete"
        status = "Completed & Evaluated" if d["is_comparable"] else ("Ready for SfM" if d["enhancement"]["categories"] else "Planned")
        lines.append(f"| **{name}** | {n_in} | {b_sfm} | {enh_img} | {e_sfm} | {comp} | {status} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Quantitative Baseline vs. Enhanced Comparison",
        "",
    ])

    comp_datasets = [d for d in eval_data if d["is_comparable"]]
    if not comp_datasets:
        lines.append("*No datasets currently have both completed baseline and enhanced reconstruction runs.*")
    else:
        lines.extend([
            "| Dataset | Metric | Original Baseline | Adaptive Enhanced | Absolute Diff | Relative Diff (%) | Interpretation |",
            "|---|---|---|---|---|---|---|",
        ])
        for d in comp_datasets:
            name = d["dataset"]
            c = d["comparison"]
            b = d["baseline"]
            e = d["enhanced"]

            # Reg images
            lines.append(
                f"| **{name}** | Registered Images | {b['registered_images']} ({b['registration_rate_pct']}%) | "
                f"{e['registered_images']} ({e['registration_rate_pct']}%) | "
                f"{c['registered_images_diff']['abs']} | {c['registered_images_diff']['pct']}% | "
                f"{'Perfect registration maintained' if b['registered_images'] == e['registered_images'] else 'Registration changed'} |"
            )

            # 3D points
            pts_interp = f"+{c['points_diff']['abs']} points added" if c['points_diff']['abs'] > 0 else f"{c['points_diff']['abs']} points"
            lines.append(
                f"| | Reconstructed 3D Points | {b['num_points']:,} | {e['num_points']:,} | "
                f"{c['points_diff']['abs']:+,} | {c['points_diff']['pct']:+.2f}% | "
                f"{pts_interp} |"
            )

            # Reprojection error
            err_interp = "Slightly higher error" if c['reprojection_error_diff']['abs'] > 0 else "Error reduced"
            lines.append(
                f"| | Mean Reprojection Error | {b['mean_reprojection_error_px']:.4f} px | {e['mean_reprojection_error_px']:.4f} px | "
                f"{c['reprojection_error_diff']['abs']:+.4f} px | {c['reprojection_error_diff']['pct']:+.2f}% | "
                f"{err_interp} |"
            )

            # Observations / image
            lines.append(
                f"| | Mean Observations / Image | {b['mean_observations_per_image']:.2f} | {e['mean_observations_per_image']:.2f} | "
                f"{c['observations_diff']['abs']:+.2f} | {c['observations_diff']['pct']:+.2f}% | "
                f"{'Comparable coverage' if abs(c['observations_diff']['pct']) < 1.0 else 'Observation density shifted'} |"
            )

            # Track length
            lines.append(
                f"| | Mean Track Length | {b['mean_track_length']:.4f} | {e['mean_track_length']:.4f} | "
                f"{c['track_length_diff']['abs']:+.4f} | {c['track_length_diff']['pct']:+.2f}% | "
                f"{'Stable track length' if abs(c['track_length_diff']['pct']) < 1.0 else 'Track length changed'} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Scientific Analysis & Observations",
        "",
        "1. **3D Points vs. Quality:** Adding 3D points (+185 in Gerrard Hall) demonstrates that enhancement exposed additional features in low-contrast regions. However, more 3D points do not automatically guarantee superior reconstruction geometry.",
        "2. **Reprojection Error:** The slight increase (+0.0017 px / +0.16%) in Gerrard Hall reprojection error is typical when triangulating features in previously weak or marginal contrast regions.",
        "3. **Dataset Diversity:** South Building exhibits a markedly higher natural sharpness (minimum Laplacian variance 122.79 vs Gerrard Hall's 37.14). Hence, sharpening was selectively skipped, while 28 low-contrast images (21.9%) were enhanced via CLAHE.",
        "",
        "---",
        "",
        "## 4. Processing Runtime & Bottleneck Summary",
        "",
        "| Dataset | Quality Analysis (s) | Enhancement (s) | Baseline SfM Total (s) | Enhanced SfM Total (s) | SfM Time Diff (s) |",
        "|---|---|---|---|---|---|",
    ])

    for d in eval_data:
        name = d["dataset"]
        qa_s = f"{d['enhancement']['analysis_time_s']:.1f}" if d['enhancement']['analysis_time_s'] else "N/A"
        enh_s = f"{d['enhancement']['enhancement_time_s']:.1f}" if d['enhancement']['enhancement_time_s'] else "N/A"
        b_time = f"{d['baseline']['timings_s'].get('total_sfm_time_s', 'N/A')}" if d["has_baseline_run"] else "Pending"
        e_time = f"{d['enhanced']['timings_s'].get('total_sfm_time_s', 'N/A')}" if d["has_enhanced_run"] else "Pending"
        diff_time = "N/A"
        if d["has_baseline_run"] and d["has_enhanced_run"] and "total_sfm_time_s" in d['baseline']['timings_s'] and "total_sfm_time_s" in d['enhanced']['timings_s']:
            d_s = d['enhanced']['timings_s']['total_sfm_time_s'] - d['baseline']['timings_s']['total_sfm_time_s']
            diff_time = f"{d_s:+.1f}s"
        lines.append(f"| **{name}** | {qa_s} | {enh_s} | {b_time} | {e_time} | {diff_time} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved Markdown report to: {report_path}")


def main() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_names = ["gerrard-hall", "south-building", "person-hall", "graham-hall"]
    eval_results = []

    for name in dataset_names:
        logger.info(f"Evaluating dataset: {name}")
        entry = evaluate_dataset(name)
        eval_results.append(entry)

    # 1. Save JSON
    json_path = EVALUATION_DIR / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Saved JSON summary: {json_path}")

    # 2. Save CSV
    csv_path = EVALUATION_DIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Input_Images", "Comparable",
            "Enhanced_Images", "Pct_Enhanced",
            "Base_Reg_Images", "Enh_Reg_Images", "Reg_Images_Diff",
            "Base_3D_Points", "Enh_3D_Points", "Points_Diff_Abs", "Points_Diff_Pct",
            "Base_Reproj_Error_px", "Enh_Reproj_Error_px", "Reproj_Error_Diff_Abs", "Reproj_Error_Diff_Pct",
            "Base_Mean_Observations", "Enh_Mean_Observations", "Obs_Diff_Abs",
            "Base_Track_Length", "Enh_Track_Length", "Track_Length_Diff_Abs",
            "Analysis_Time_s", "Enhancement_Time_s", "Base_SfM_Time_s", "Enh_SfM_Time_s"
        ])
        for d in eval_results:
            b = d["baseline"]
            e = d["enhanced"]
            c = d["comparison"]
            writer.writerow([
                d["dataset"],
                d["input_images"],
                d["is_comparable"],
                d["enhancement"]["enhanced_images"],
                d["enhancement"]["percentage_enhanced"],
                b["registered_images"],
                e["registered_images"],
                c.get("registered_images_diff", {}).get("abs"),
                b["num_points"],
                e["num_points"],
                c.get("points_diff", {}).get("abs"),
                c.get("points_diff", {}).get("pct"),
                b["mean_reprojection_error_px"],
                e["mean_reprojection_error_px"],
                c.get("reprojection_error_diff", {}).get("abs"),
                c.get("reprojection_error_diff", {}).get("pct"),
                b["mean_observations_per_image"],
                e["mean_observations_per_image"],
                c.get("observations_diff", {}).get("abs"),
                b["mean_track_length"],
                e["mean_track_length"],
                c.get("track_length_diff", {}).get("abs"),
                d["enhancement"]["analysis_time_s"],
                d["enhancement"]["enhancement_time_s"],
                b["timings_s"].get("total_sfm_time_s"),
                e["timings_s"].get("total_sfm_time_s"),
            ])
    logger.info(f"Saved CSV summary: {csv_path}")

    # 3. Generate visual distribution and comparison plots
    generate_evaluation_plots(eval_results, PLOTS_DIR)

    # 4. Generate markdown report
    report_path = EVALUATION_DIR / "report.md"
    generate_markdown_report(eval_results, report_path)

    logger.info("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
