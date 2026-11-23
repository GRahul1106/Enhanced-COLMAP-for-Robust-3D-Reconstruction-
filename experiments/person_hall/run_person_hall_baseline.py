"""
run_person_hall_baseline.py

Executes Phase A: Person Hall Baseline COLMAP Reconstruction.

Pipeline Stages:
1. feature_extractor (--ImageReader.single_camera 1)
2. exhaustive_matcher
3. mapper
4. model_converter (TXT)

Outputs:
  Database: experiments/person_hall/baseline/database.db
  Sparse Model: experiments/person_hall/baseline/sparse/0/
  Log: experiments/person_hall/baseline/baseline.log
  Timings & Metrics: experiments/person_hall/baseline/baseline_summary.json
  SIFT Counts: experiments/person_hall/baseline/baseline_sift_counts.csv
  Variant E Candidate Images: experiments/person_hall/baseline/candidate_images_variant_e.csv
"""

import csv
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Dict, List, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = PROJECT_ROOT / "experiments" / "person_hall" / "baseline"
IMAGES_DIR = BASE_DIR / "images"
DATABASE_PATH = BASE_DIR / "database.db"
SPARSE_DIR = BASE_DIR / "sparse"
SPARSE_MODEL_DIR = SPARSE_DIR / "0"
LOG_PATH = BASE_DIR / "baseline.log"
COLMAP_BAT = PROJECT_ROOT / "colmap" / "COLMAP.bat"
COLMAP_EXE = PROJECT_ROOT / "colmap" / "bin" / "colmap.exe"
COLMAP_BIN_DIR = COLMAP_EXE.parent
COLMAP_PLUGINS_DIR = PROJECT_ROOT / "colmap" / "plugins"
CONTRAST_CSV = BASE_DIR / "image_contrast.csv"


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("person_hall_baseline")
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


def run_colmap_stage(subcommand: str, args: list, logger: logging.Logger, stage_name: str) -> float:
    cmd = [str(COLMAP_EXE), subcommand] + args

    env = os.environ.copy()
    env["PATH"] = str(COLMAP_BIN_DIR) + os.pathsep + env.get("PATH", "")
    env["QT_PLUGIN_PATH"] = str(COLMAP_PLUGINS_DIR)

    cmd_display = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    logger.info("=" * 70)
    logger.info(f"STAGE: {stage_name}")
    logger.info(f"Command: {cmd_display}")
    logger.info("=" * 70)

    start_time = time.time()

    if not COLMAP_EXE.exists():
        raise FileNotFoundError(f"colmap.exe not found at: {COLMAP_EXE}")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    elapsed = time.time() - start_time

    if proc.stdout.strip():
        for line in proc.stdout.splitlines():
            logger.debug(f"  [stdout] {line}")
    if proc.stderr.strip():
        for line in proc.stderr.splitlines():
            logger.debug(f"  [stderr] {line}")

    if proc.returncode != 0:
        logger.error(f"STAGE FAILED: {stage_name} (exit code {proc.returncode})")
        if proc.stdout.strip():
            for line in proc.stdout.splitlines():
                logger.error(f"  {line}")
        if proc.stderr.strip():
            for line in proc.stderr.splitlines():
                logger.error(f"  {line}")
        raise RuntimeError(f"COLMAP '{subcommand}' failed with exit code {proc.returncode}")

    logger.info(f"STAGE COMPLETE: {stage_name} ({elapsed:.1f}s)")
    return elapsed


def parse_sparse_model(sparse_dir: Path) -> Dict:
    p3d_file = sparse_dir / "points3D.txt"
    img_file = sparse_dir / "images.txt"

    if not p3d_file.exists() or not img_file.exists():
        return {}

    num_points = 0
    mean_track_length = 0.0
    errors = []

    with open(p3d_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# Number of points:"):
                import re
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

    num_reg_images = 0
    mean_obs = 0.0
    with open(img_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Number of images:"):
                import re
                m = re.search(r"Number of images:\s*(\d+),\s*mean observations per image:\s*([0-9.]+)", line)
                if m:
                    num_reg_images = int(m.group(1))
                    mean_obs = float(m.group(2))
                break

    mean_err = float(np.mean(errors)) if errors else 0.0
    median_err = float(np.median(errors)) if errors else 0.0

    return {
        "registered_images": num_reg_images,
        "num_3d_points": num_points if num_points else len(errors),
        "mean_reprojection_error_px": round(mean_err, 4),
        "median_reprojection_error_px": round(median_err, 4),
        "mean_observations_per_image": round(mean_obs, 2),
        "mean_track_length": round(mean_track_length, 4),
    }


def extract_database_metrics(db_path: Path) -> Dict:
    if not db_path.exists():
        return {}

    db_size_bytes = db_path.stat().st_size
    db_size_mb = round(db_size_bytes / (1024 * 1024), 2)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Image SIFT counts
    cursor.execute("""
        SELECT images.name, keypoints.rows
        FROM images
        JOIN keypoints ON images.image_id = keypoints.image_id
        ORDER BY images.name ASC;
    """)
    sift_rows = cursor.fetchall()

    sift_dict = {name: rows for name, rows in sift_rows}

    # Two-view verified inliers and verified pair count
    cursor.execute("SELECT SUM(rows), COUNT(*) FROM two_view_geometries WHERE rows > 0;")
    res = cursor.fetchone()
    two_view_inliers = res[0] if res and res[0] is not None else 0
    verified_pairs_count = res[1] if res and res[1] is not None else 0

    # Total raw matches and matched pair count
    cursor.execute("SELECT SUM(rows), COUNT(*) FROM matches WHERE rows > 0;")
    res_m = cursor.fetchone()
    raw_matches = res_m[0] if res_m and res_m[0] is not None else 0
    raw_pairs_count = res_m[1] if res_m and res_m[1] is not None else 0

    conn.close()

    return {
        "database_size_bytes": db_size_bytes,
        "database_size_mb": db_size_mb,
        "two_view_inliers": two_view_inliers,
        "verified_pairs_count": verified_pairs_count,
        "raw_matches": raw_matches,
        "raw_pairs_count": raw_pairs_count,
        "sift_dict": sift_dict,
    }


def main():
    logger = setup_logging(LOG_PATH)
    logger.info("=" * 70)
    logger.info("PERSON HALL BASELINE RECONSTRUCTION - START")
    logger.info(f"Images: {IMAGES_DIR}")
    logger.info(f"Output Database: {DATABASE_PATH}")
    logger.info(f"Output Sparse: {SPARSE_DIR}")
    logger.info(f"Log: {LOG_PATH}")
    logger.info("=" * 70)

    # Re-verify images directory
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")

    images_list = sorted(list(IMAGES_DIR.glob("*.JPG")))
    if len(images_list) != 330:
        raise ValueError(f"Expected 330 images, but found {len(images_list)}")

    logger.info(f"Validated input images count: {len(images_list)}")

    timings = {}
    start_total = time.time()

    # Stage 1: Feature Extraction
    timings["feature_extraction_s"] = run_colmap_stage(
        subcommand="feature_extractor",
        args=[
            "--database_path", str(DATABASE_PATH),
            "--image_path", str(IMAGES_DIR),
            "--ImageReader.single_camera", "1",
        ],
        logger=logger,
        stage_name="Feature Extraction (SIFT)",
    )

    # Stage 2: Exhaustive Feature Matching
    timings["exhaustive_matching_s"] = run_colmap_stage(
        subcommand="exhaustive_matcher",
        args=[
            "--database_path", str(DATABASE_PATH),
        ],
        logger=logger,
        stage_name="Exhaustive Feature Matching",
    )

    # Stage 3: Sparse Reconstruction (mapper)
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)
    timings["sparse_mapping_s"] = run_colmap_stage(
        subcommand="mapper",
        args=[
            "--database_path", str(DATABASE_PATH),
            "--image_path", str(IMAGES_DIR),
            "--output_path", str(SPARSE_DIR),
        ],
        logger=logger,
        stage_name="Sparse Reconstruction / SfM (mapper)",
    )

    if not SPARSE_MODEL_DIR.exists():
        raise RuntimeError(f"Mapper produced no reconstruction model at {SPARSE_MODEL_DIR}")

    # Stage 4: Model Conversion (binary -> TXT) for all models
    model_dirs = sorted([d for d in SPARSE_DIR.iterdir() if d.is_dir()])
    if not model_dirs:
        raise RuntimeError(f"Mapper produced no reconstruction model directory inside {SPARSE_DIR}")

    timings["model_conversion_s"] = 0.0
    for m_dir in model_dirs:
        logger.info(f"Converting model at {m_dir} to TXT...")
        t_conv = run_colmap_stage(
            subcommand="model_converter",
            args=[
                "--input_path", str(m_dir),
                "--output_path", str(m_dir),
                "--output_type", "TXT",
            ],
            logger=logger,
            stage_name=f"Model Conversion (binary -> TXT) [{m_dir.name}]",
        )
        timings["model_conversion_s"] += t_conv

    total_elapsed = time.time() - start_total
    timings["total_sfm_time_s"] = round(total_elapsed, 1)

    logger.info("Extracting database and model metrics...")
    db_metrics = extract_database_metrics(DATABASE_PATH)
    sparse_metrics = parse_sparse_model(SPARSE_MODEL_DIR)

    # Save SIFT counts to CSV
    sift_dict = db_metrics.pop("sift_dict", {})
    sift_csv_path = BASE_DIR / "baseline_sift_counts.csv"
    with open(sift_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "SIFT_Features"])
        for img_name in sorted(sift_dict.keys()):
            writer.writerow([img_name, sift_dict[img_name]])
    logger.info(f"Saved SIFT feature counts for {len(sift_dict)} images to {sift_csv_path}")

    # Load contrast metrics and check dual criterion
    contrast_dict = {}
    if CONTRAST_CSV.exists():
        with open(CONTRAST_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                contrast_dict[r["Image"]] = float(r["Contrast_StdDev"])

    # Evaluate Variant E candidate criteria: Contrast < 52.0 AND SIFT < 10000
    candidate_images = []
    for img_p in images_list:
        name = img_p.name
        contrast_val = contrast_dict.get(name, None)
        sift_val = sift_dict.get(name, None)
        is_candidate = (
            contrast_val is not None
            and contrast_val < 52.0
            and sift_val is not None
            and sift_val < 10000
        )
        if is_candidate:
            candidate_images.append({
                "Image": name,
                "Contrast_StdDev": contrast_val,
                "SIFT_Features": sift_val,
            })

    candidate_csv_path = BASE_DIR / "candidate_images_variant_e.csv"
    with open(candidate_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "Contrast_StdDev", "SIFT_Features"])
        writer.writeheader()
        writer.writerows(candidate_images)
    logger.info(f"Saved {len(candidate_images)} Variant E candidate images to {candidate_csv_path}")

    # Build final summary
    summary = {
        "dataset": "person-hall",
        "total_images": len(images_list),
        "sparse_model": sparse_metrics,
        "database_metrics": db_metrics,
        "timings": timings,
        "variant_e_candidates_count": len(candidate_images),
        "variant_e_candidates": candidate_images,
    }

    summary_json_path = BASE_DIR / "baseline_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("PERSON HALL BASELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Registered Images: {sparse_metrics.get('registered_images')} / {len(images_list)}")
    logger.info(f"3D Points: {sparse_metrics.get('num_3d_points')}")
    logger.info(f"Mean Reprojection Error: {sparse_metrics.get('mean_reprojection_error_px')} px")
    logger.info(f"Median Reprojection Error: {sparse_metrics.get('median_reprojection_error_px')} px")
    logger.info(f"Mean Obs / Image: {sparse_metrics.get('mean_observations_per_image')}")
    logger.info(f"Mean Track Length: {sparse_metrics.get('mean_track_length')}")
    logger.info(f"Database Size: {db_metrics.get('database_size_mb')} MB")
    logger.info(f"Verified Two-View Inliers: {db_metrics.get('two_view_inliers')}")
    logger.info(f"Candidate Images for Variant E (Contrast < 52.0 & SIFT < 10,000): {len(candidate_images)}")
    logger.info(f"Total SFM Runtime: {timings.get('total_sfm_time_s')}s")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
