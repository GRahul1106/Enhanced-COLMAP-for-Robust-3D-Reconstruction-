"""
run_person_hall_variant_e.py

Executes Phase B: Person Hall Variant E Reconstruction.

Pipeline Stages:
1. Feature Extraction:
   - 5 enhanced images extracted via COLMAP SiftGPU into temporary db
   - Assembled into isolated Variant E database alongside the 325 byte-identical views
   - Single unified Camera #1 (SIMPLE_RADIAL 5616x3744) preserved across all 330 images
2. exhaustive_matcher (all 54,285 pairs)
3. mapper (incremental SfM)
4. model_converter (binary -> TXT)

Outputs:
  Database: experiments/person_hall/variant_e/database.db
  Sparse Model: experiments/person_hall/variant_e/sparse/0/
  Log: experiments/person_hall/variant_e/variant_e.log
  Timings & Metrics: experiments/person_hall/variant_e/variant_e_summary.json
  SIFT Counts: experiments/person_hall/variant_e/variant_e_sift_counts.csv
  Comparison: experiments/person_hall/variant_e/comparison_with_baseline.json
"""

import csv
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Dict, List, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = PROJECT_ROOT / "experiments" / "person_hall" / "baseline"
BASE_DB_PATH = BASE_DIR / "database.db"
BASE_SUMMARY_PATH = BASE_DIR / "baseline_summary.json"

VAR_E_DIR = PROJECT_ROOT / "experiments" / "person_hall" / "variant_e"
IMAGES_DIR = VAR_E_DIR / "images"
DATABASE_PATH = VAR_E_DIR / "database.db"
SPARSE_DIR = VAR_E_DIR / "sparse"
SPARSE_MODEL_DIR = SPARSE_DIR / "0"
LOG_PATH = VAR_E_DIR / "variant_e.log"
COLMAP_BAT = PROJECT_ROOT / "colmap" / "COLMAP.bat"
COLMAP_EXE = PROJECT_ROOT / "colmap" / "bin" / "colmap.exe"
COLMAP_BIN_DIR = COLMAP_EXE.parent
COLMAP_PLUGINS_DIR = PROJECT_ROOT / "colmap" / "plugins"

SELECTED_IMAGES = [
    "IMG_1179.JPG",
    "IMG_1282.JPG",
    "IMG_1312.JPG",
    "IMG_1316.JPG",
    "IMG_1323.JPG",
]


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("person_hall_variant_e")
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

    # Two-view verified inliers and pair count
    cursor.execute("SELECT SUM(rows), COUNT(*) FROM two_view_geometries WHERE rows > 0;")
    res = cursor.fetchone()
    two_view_inliers = res[0] if res and res[0] is not None else 0
    verified_pairs_count = res[1] if res and res[1] is not None else 0

    # Total raw matches and pair count
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
    logger.info("PERSON HALL VARIANT E RECONSTRUCTION - START")
    logger.info(f"Images: {IMAGES_DIR}")
    logger.info(f"Output Database: {DATABASE_PATH}")
    logger.info(f"Output Sparse: {SPARSE_DIR}")
    logger.info(f"Log: {LOG_PATH}")
    logger.info("=" * 70)

    # 1. Re-verify images directory
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")

    images_list = sorted(list(IMAGES_DIR.glob("*.JPG")))
    if len(images_list) != 330:
        raise ValueError(f"Expected 330 images, but found {len(images_list)}")

    logger.info(f"Validated input images count: {len(images_list)}")

    # 2. Reset database and sparse directories safely
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()
        logger.warning(f"Removed existing database: {DATABASE_PATH}")
    if SPARSE_DIR.exists():
        shutil.rmtree(SPARSE_DIR)
        logger.warning(f"Removed existing sparse directory: {SPARSE_DIR}")

    timings = {}
    start_total = time.time()

    # Stage 1: Feature Extraction
    # Extract SIFT features on the 5 enhanced images into a temporary db via COLMAP SiftGPU
    temp_enh_db = VAR_E_DIR / "temp_enh_features.db"
    if temp_enh_db.exists():
        temp_enh_db.unlink()

    enh_list_txt = VAR_E_DIR / "enh_list.txt"
    with open(enh_list_txt, "w") as f:
        f.write("\n".join(SELECTED_IMAGES))

    logger.info("Running feature extraction on the 5 enhanced images...")
    t_enh_extract = run_colmap_stage(
        subcommand="feature_extractor",
        args=[
            "--database_path", str(temp_enh_db),
            "--image_path", str(IMAGES_DIR),
            "--image_list_path", str(enh_list_txt),
            "--ImageReader.single_camera", "1",
        ],
        logger=logger,
        stage_name="Feature Extraction (SIFT) [5 Enhanced Images]",
    )
    enh_list_txt.unlink()

    # Assemble isolated Variant E database:
    # Copy baseline database (which contains Camera 1 and all 325 identical views' features)
    logger.info("Assembling isolated Variant E database from baseline reference...")
    shutil.copy2(BASE_DB_PATH, DATABASE_PATH)

    conn = sqlite3.connect(str(DATABASE_PATH))
    c = conn.cursor()
    # Reset matches and two_view_geometries for fresh reconstruction
    c.execute("DELETE FROM matches")
    c.execute("DELETE FROM two_view_geometries")

    # Update the 5 enhanced images with their freshly extracted SIFT keypoints & descriptors
    conn_enh = sqlite3.connect(str(temp_enh_db))
    c_enh = conn_enh.cursor()

    for name in SELECTED_IMAGES:
        c.execute("SELECT image_id FROM images WHERE name = ?", (name,))
        base_id = c.fetchone()[0]

        c_enh.execute(
            "SELECT keypoints.rows, keypoints.cols, keypoints.data FROM images JOIN keypoints ON images.image_id = keypoints.image_id WHERE images.name = ?",
            (name,)
        )
        kp_row = c_enh.fetchone()

        c_enh.execute(
            "SELECT descriptors.type, descriptors.rows, descriptors.cols, descriptors.data FROM images JOIN descriptors ON images.image_id = descriptors.image_id WHERE images.name = ?",
            (name,)
        )
        desc_row = c_enh.fetchone()

        c.execute("UPDATE keypoints SET rows = ?, cols = ?, data = ? WHERE image_id = ?", (kp_row[0], kp_row[1], kp_row[2], base_id))
        c.execute("UPDATE descriptors SET type = ?, rows = ?, cols = ?, data = ? WHERE image_id = ?", (desc_row[0], desc_row[1], desc_row[2], desc_row[3], base_id))

    conn.commit()
    c.execute("VACUUM")
    conn.close()
    conn_enh.close()

    temp_enh_db.unlink()
    timings["feature_extraction_s"] = t_enh_extract

    # Verify single camera and full 330 images in database
    conn_v = sqlite3.connect(str(DATABASE_PATH))
    c_v = conn_v.cursor()
    c_v.execute("SELECT COUNT(*) FROM images")
    db_img_cnt = c_v.fetchone()[0]
    c_v.execute("SELECT COUNT(*) FROM keypoints")
    db_kp_cnt = c_v.fetchone()[0]
    c_v.execute("SELECT COUNT(*) FROM cameras")
    db_cam_cnt = c_v.fetchone()[0]
    c_v.execute("SELECT DISTINCT camera_id FROM images")
    cam_ids = [r[0] for r in c_v.fetchall()]
    conn_v.close()

    logger.info(f"Database Feature Extraction Verified: Images={db_img_cnt}/330, Keypoints={db_kp_cnt}/330, Cameras={db_cam_cnt}, Camera IDs={cam_ids}")
    if db_img_cnt != 330 or db_kp_cnt != 330 or db_cam_cnt != 1 or cam_ids != [1]:
        raise RuntimeError(f"Database verification failed: images={db_img_cnt}, keypoints={db_kp_cnt}, cameras={db_cam_cnt}, camera_ids={cam_ids}")

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
    sift_csv_path = VAR_E_DIR / "variant_e_sift_counts.csv"
    with open(sift_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "SIFT_Features"])
        for img_name in sorted(sift_dict.keys()):
            writer.writerow([img_name, sift_dict[img_name]])
    logger.info(f"Saved SIFT feature counts for {len(sift_dict)} images to {sift_csv_path}")

    # Build final summary
    summary = {
        "dataset": "person-hall",
        "variant": "variant_e_conservative_selection",
        "total_images": len(images_list),
        "sparse_model": sparse_metrics,
        "database_metrics": db_metrics,
        "timings": timings,
    }

    summary_json_path = VAR_E_DIR / "variant_e_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Load baseline summary for side-by-side comparison
    comparison = {}
    if BASE_SUMMARY_PATH.exists():
        with open(BASE_SUMMARY_PATH, "r", encoding="utf-8") as f:
            base_sum = json.load(f)

        base_sparse = base_sum.get("sparse_model", {})
        enh_sparse = sparse_metrics

        base_db = base_sum.get("database_metrics", {})
        enh_db = db_metrics

        base_timings = base_sum.get("timings", {})
        enh_timings = timings

        comparison = {
            "registered_images": {
                "baseline": base_sparse.get("registered_images"),
                "variant_e": enh_sparse.get("registered_images"),
                "delta": enh_sparse.get("registered_images", 0) - base_sparse.get("registered_images", 0),
            },
            "num_3d_points": {
                "baseline": base_sparse.get("num_3d_points"),
                "variant_e": enh_sparse.get("num_3d_points"),
                "delta_abs": enh_sparse.get("num_3d_points", 0) - base_sparse.get("num_3d_points", 0),
                "delta_pct": round(
                    ((enh_sparse.get("num_3d_points", 0) - base_sparse.get("num_3d_points", 0)) / base_sparse.get("num_3d_points", 1)) * 100, 3
                ) if base_sparse.get("num_3d_points") else 0.0,
            },
            "mean_reprojection_error_px": {
                "baseline": base_sparse.get("mean_reprojection_error_px"),
                "variant_e": enh_sparse.get("mean_reprojection_error_px"),
                "delta_abs": round(enh_sparse.get("mean_reprojection_error_px", 0) - base_sparse.get("mean_reprojection_error_px", 0), 4),
            },
            "median_reprojection_error_px": {
                "baseline": base_sparse.get("median_reprojection_error_px"),
                "variant_e": enh_sparse.get("median_reprojection_error_px"),
                "delta_abs": round(enh_sparse.get("median_reprojection_error_px", 0) - base_sparse.get("median_reprojection_error_px", 0), 4),
            },
            "mean_observations_per_image": {
                "baseline": base_sparse.get("mean_observations_per_image"),
                "variant_e": enh_sparse.get("mean_observations_per_image"),
                "delta_abs": round(enh_sparse.get("mean_observations_per_image", 0) - base_sparse.get("mean_observations_per_image", 0), 2),
            },
            "mean_track_length": {
                "baseline": base_sparse.get("mean_track_length"),
                "variant_e": enh_sparse.get("mean_track_length"),
                "delta_abs": round(enh_sparse.get("mean_track_length", 0) - base_sparse.get("mean_track_length", 0), 4),
            },
            "two_view_inliers": {
                "baseline": base_db.get("two_view_inliers"),
                "variant_e": enh_db.get("two_view_inliers"),
                "delta_abs": enh_db.get("two_view_inliers", 0) - base_db.get("two_view_inliers", 0),
                "delta_pct": round(
                    ((enh_db.get("two_view_inliers", 0) - base_db.get("two_view_inliers", 0)) / base_db.get("two_view_inliers", 1)) * 100, 3
                ) if base_db.get("two_view_inliers") else 0.0,
            },
            "verified_pairs_count": {
                "baseline": base_db.get("verified_pairs_count"),
                "variant_e": enh_db.get("verified_pairs_count"),
                "delta_abs": enh_db.get("verified_pairs_count", 0) - base_db.get("verified_pairs_count", 0),
            },
            "database_size_mb": {
                "baseline": base_db.get("database_size_mb"),
                "variant_e": enh_db.get("database_size_mb"),
                "delta_mb": round(enh_db.get("database_size_mb", 0) - base_db.get("database_size_mb", 0), 2),
            },
            "total_sfm_time_s": {
                "baseline": base_timings.get("total_sfm_time_s"),
                "variant_e": enh_timings.get("total_sfm_time_s"),
                "delta_s": round(enh_timings.get("total_sfm_time_s", 0) - base_timings.get("total_sfm_time_s", 0), 1),
            },
            "stage_runtimes_s": {
                "feature_extraction": {
                    "baseline": base_timings.get("feature_extraction_s"),
                    "variant_e": enh_timings.get("feature_extraction_s"),
                },
                "exhaustive_matching": {
                    "baseline": base_timings.get("exhaustive_matching_s"),
                    "variant_e": enh_timings.get("exhaustive_matching_s"),
                },
                "sparse_mapping": {
                    "baseline": base_timings.get("sparse_mapping_s"),
                    "variant_e": enh_timings.get("sparse_mapping_s"),
                },
                "model_conversion": {
                    "baseline": base_timings.get("model_conversion_s"),
                    "variant_e": enh_timings.get("model_conversion_s"),
                },
            },
        }

        comp_json_path = VAR_E_DIR / "comparison_with_baseline.json"
        with open(comp_json_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison JSON saved to: {comp_json_path}")

    logger.info("=" * 70)
    logger.info("PERSON HALL VARIANT E COMPLETED SUCCESSFULLY")
    logger.info(f"Registered Images: {sparse_metrics.get('registered_images')} / {len(images_list)}")
    logger.info(f"3D Points: {sparse_metrics.get('num_3d_points')}")
    logger.info(f"Mean Reprojection Error: {sparse_metrics.get('mean_reprojection_error_px')} px")
    logger.info(f"Median Reprojection Error: {sparse_metrics.get('median_reprojection_error_px')} px")
    logger.info(f"Mean Obs / Image: {sparse_metrics.get('mean_observations_per_image')}")
    logger.info(f"Mean Track Length: {sparse_metrics.get('mean_track_length')}")
    logger.info(f"Database Size: {db_metrics.get('database_size_mb')} MB")
    logger.info(f"Verified Two-View Inliers: {db_metrics.get('two_view_inliers')}")
    logger.info(f"Total SFM Runtime: {timings.get('total_sfm_time_s')}s")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
