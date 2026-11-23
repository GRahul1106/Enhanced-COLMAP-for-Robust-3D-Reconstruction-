"""
run_graham_hall_variant_e.py

Executes Phase B: Graham Hall 270-Image Exterior Subset Variant E Reconstruction.

Pipeline Stages:
  1. Feature Extraction:
     - SIFT extraction on the 28 enhanced images into temporary database
     - Assembled into isolated Variant E database alongside the 230 processed baseline views
     - Unified Camera #1 (SIMPLE_RADIAL 5616x3744) preserved across all views
  2. colmap exhaustive_matcher (all pairs)
  3. colmap mapper (incremental SfM)
  4. colmap model_converter (binary -> TXT)
  5. Metric extraction & automated comparison with baseline

Outputs:
  Database: experiments/graham_hall/variant_e/database.db
  Sparse Model: experiments/graham_hall/variant_e/sparse/0/
  Log: experiments/graham_hall/variant_e/variant_e.log
  Summary: experiments/graham_hall/variant_e/variant_e_summary.json
  SIFT Counts: experiments/graham_hall/variant_e/variant_e_sift_counts.csv
  Comparison: experiments/graham_hall/variant_e/comparison_with_baseline.json
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

PROJECT_ROOT = Path(r"E:\CV Project").resolve()
BASE_DIR = PROJECT_ROOT / "experiments" / "graham_hall" / "baseline"
BASE_DB_PATH = BASE_DIR / "database.db"
BASE_SUMMARY_PATH = BASE_DIR / "baseline_summary.json"

VAR_E_DIR = PROJECT_ROOT / "experiments" / "graham_hall" / "variant_e"
IMAGES_DIR = VAR_E_DIR / "images"
DATABASE_PATH = VAR_E_DIR / "database.db"
SPARSE_DIR = VAR_E_DIR / "sparse"
SPARSE_MODEL_DIR = SPARSE_DIR / "0"
LOG_PATH = VAR_E_DIR / "variant_e.log"
MANIFEST_PATH = VAR_E_DIR / "enhancement_manifest.json"

COLMAP_EXE = PROJECT_ROOT / "colmap" / "bin" / "colmap.exe"
COLMAP_BIN_DIR = COLMAP_EXE.parent
COLMAP_PLUGINS_DIR = PROJECT_ROOT / "colmap" / "plugins"


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("graham_hall_variant_e")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def stream_colmap_stage(subcommand: str, args: list, logger: logging.Logger, stage_name: str) -> float:
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
    t_start_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    logger.info(f"Stage Start Time: {t_start_iso}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )

    for line in iter(proc.stdout.readline, ""):
        line_str = line.strip()
        if line_str:
            logger.debug(f"  [colmap] {line_str}")
            if any(k in line_str for k in [
                "=>", "Registering image", "Bundle adjustment", "Retriangulation",
                "Number of registered images", "Number of points", "Elapsed time",
                "Initialization", "Extracting", "Writing", "Processing block"
            ]):
                logger.info(f"  {line_str}")

    proc.stdout.close()
    return_code = proc.wait()
    elapsed = time.time() - start_time
    t_end_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    logger.info(f"Stage End Time: {t_end_iso} | Elapsed: {elapsed:.1f}s | Exit Code: {return_code}")

    if return_code != 0:
        logger.error(f"STAGE FAILED: {stage_name} (Exit code {return_code})")
        raise RuntimeError(f"COLMAP '{subcommand}' failed with exit code {return_code}")

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
        "registration_rate_pct": round((num_reg_images / 270.0) * 100.0, 2),
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

    cursor.execute("""
        SELECT images.name, keypoints.rows
        FROM images
        JOIN keypoints ON images.image_id = keypoints.image_id
        ORDER BY images.name ASC;
    """)
    sift_rows = cursor.fetchall()
    sift_dict = {name: rows for name, rows in sift_rows}

    cursor.execute("SELECT SUM(rows), COUNT(*) FROM two_view_geometries WHERE rows > 0;")
    res = cursor.fetchone()
    two_view_inliers = res[0] if res and res[0] is not None else 0
    verified_pairs_count = res[1] if res and res[1] is not None else 0

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
    logger.info("PHASE 7: GRAHAM HALL 270-IMAGE VARIANT E RECONSTRUCTION")
    logger.info(f"Images Directory: {IMAGES_DIR}")
    logger.info(f"Database Path: {DATABASE_PATH}")
    logger.info(f"Sparse Output: {SPARSE_DIR}")
    logger.info(f"Log Path: {LOG_PATH}")
    logger.info("=" * 70)

    # 1. Load enhanced manifest
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Enhancement manifest missing: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    enhanced_images = manifest["enhanced_images"]
    logger.info(f"Loaded {len(enhanced_images)} enhanced images from manifest.")

    # 2. Reset database and sparse directories safely
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()
        logger.warning(f"Reset existing database: {DATABASE_PATH}")
    if SPARSE_DIR.exists():
        shutil.rmtree(SPARSE_DIR)
        logger.warning(f"Reset existing sparse directory: {SPARSE_DIR}")
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)

    timings = {}
    total_start = time.time()

    # Stage 1: Feature Extraction on the 28 Enhanced Images
    temp_enh_db = VAR_E_DIR / "temp_enh.db"
    if temp_enh_db.exists():
        temp_enh_db.unlink()

    enh_list_file = VAR_E_DIR / "enh_list.txt"
    with open(enh_list_file, "w", encoding="utf-8") as f:
        f.write("\n".join(enhanced_images))

    logger.info(f"Extracting SIFT features on the {len(enhanced_images)} enhanced images...")
    t_enh_extract = stream_colmap_stage(
        subcommand="feature_extractor",
        args=[
            "--database_path", str(temp_enh_db),
            "--image_path", str(IMAGES_DIR),
            "--image_list_path", str(enh_list_file),
            "--ImageReader.single_camera", "1",
        ],
        logger=logger,
        stage_name=f"Feature Extraction (SIFT) [{len(enhanced_images)} Enhanced Images]",
    )
    enh_list_file.unlink()

    # Copy baseline database as master reference and update the 28 enhanced views
    logger.info("Assembling isolated Variant E database from baseline master...")
    shutil.copy2(BASE_DB_PATH, DATABASE_PATH)

    conn = sqlite3.connect(str(DATABASE_PATH))
    c = conn.cursor()
    c.execute("DELETE FROM matches;")
    c.execute("DELETE FROM two_view_geometries;")

    conn_enh = sqlite3.connect(str(temp_enh_db))
    c_enh = conn_enh.cursor()

    for name in enhanced_images:
        c.execute("SELECT image_id FROM images WHERE name = ?;", (name,))
        res = c.fetchone()
        if not res:
            logger.warning(f"Image {name} was not in baseline DB images table. Skipping DB keypoint injection.")
            continue
        base_id = res[0]

        c_enh.execute(
            "SELECT keypoints.rows, keypoints.cols, keypoints.data FROM images JOIN keypoints ON images.image_id = keypoints.image_id WHERE images.name = ?;",
            (name,)
        )
        kp_row = c_enh.fetchone()

        c_enh.execute(
            "SELECT descriptors.type, descriptors.rows, descriptors.cols, descriptors.data FROM images JOIN descriptors ON images.image_id = descriptors.image_id WHERE images.name = ?;",
            (name,)
        )
        desc_row = c_enh.fetchone()

        if kp_row and desc_row:
            c.execute("UPDATE keypoints SET rows = ?, cols = ?, data = ? WHERE image_id = ?;", (kp_row[0], kp_row[1], kp_row[2], base_id))
            c.execute("UPDATE descriptors SET type = ?, rows = ?, cols = ?, data = ? WHERE image_id = ?;", (desc_row[0], desc_row[1], desc_row[2], desc_row[3], base_id))

    conn.commit()
    c.execute("VACUUM;")
    conn.close()
    conn_enh.close()
    temp_enh_db.unlink()

    timings["feature_extraction_incremental_s"] = round(t_enh_extract, 1)

    # Verify DB state
    conn_chk = sqlite3.connect(str(DATABASE_PATH))
    c_chk = conn_chk.cursor()
    c_chk.execute("SELECT COUNT(*) FROM images;")
    db_img_cnt = c_chk.fetchone()[0]
    c_chk.execute("SELECT COUNT(*) FROM cameras;")
    db_cam_cnt = c_chk.fetchone()[0]
    conn_chk.close()
    logger.info(f"Variant E database prepared: {db_img_cnt} images, {db_cam_cnt} camera.")

    # Stage 2: Exhaustive Feature Matching
    timings["exhaustive_matching_s"] = stream_colmap_stage(
        subcommand="exhaustive_matcher",
        args=[
            "--database_path", str(DATABASE_PATH),
        ],
        logger=logger,
        stage_name="Exhaustive Feature Matching",
    )

    # Stage 3: Sparse Reconstruction (mapper)
    timings["sparse_mapping_s"] = stream_colmap_stage(
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

    # Stage 4: Model Conversion (binary -> TXT)
    timings["model_conversion_s"] = stream_colmap_stage(
        subcommand="model_converter",
        args=[
            "--input_path", str(SPARSE_MODEL_DIR),
            "--output_path", str(SPARSE_MODEL_DIR),
            "--output_type", "TXT",
        ],
        logger=logger,
        stage_name="Model Conversion (binary -> TXT)",
    )

    total_elapsed = time.time() - total_start
    timings["total_sfm_time_s"] = round(total_elapsed, 1)

    # Stage 5: Metrics Extraction
    logger.info("Extracting database and sparse model metrics...")
    db_metrics = extract_database_metrics(DATABASE_PATH)
    sparse_metrics = parse_sparse_model(SPARSE_MODEL_DIR)

    sift_dict = db_metrics.pop("sift_dict", {})
    sift_csv = VAR_E_DIR / "variant_e_sift_counts.csv"
    with open(sift_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "SIFT_Features"])
        for name in sorted(sift_dict.keys()):
            writer.writerow([name, sift_dict[name]])
    logger.info(f"Saved SIFT feature counts for {len(sift_dict)} images to: {sift_csv}")

    summary = {
        "dataset": "graham-hall-270-exterior-subset",
        "scope": "270-image exterior sequence (IMG_0348.jpg to IMG_0617.jpg)",
        "total_images": 270,
        "enhanced_images_count": len(enhanced_images),
        "unchanged_images_count": 270 - len(enhanced_images),
        "sparse_model": sparse_metrics,
        "database_metrics": db_metrics,
        "timings": timings,
    }

    summary_json = VAR_E_DIR / "variant_e_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved Variant E summary to: {summary_json}")

    # Stage 6: Compare with Baseline
    if BASE_SUMMARY_PATH.exists():
        with open(BASE_SUMMARY_PATH, "r", encoding="utf-8") as f:
            base_sum = json.load(f)

        base_sparse = base_sum.get("sparse_model", {})
        base_db = base_sum.get("database_metrics", {})
        base_timings = base_sum.get("timings", {})

        comparison = {
            "dataset": "graham-hall-270-exterior-subset",
            "scope": "270-image exterior sequence (IMG_0348.jpg to IMG_0617.jpg)",
            "registered_images": {
                "baseline": base_sparse.get("registered_images"),
                "variant_e": sparse_metrics.get("registered_images"),
                "delta": sparse_metrics.get("registered_images", 0) - base_sparse.get("registered_images", 0),
            },
            "registration_rate_pct": {
                "baseline": base_sparse.get("registration_rate_pct"),
                "variant_e": sparse_metrics.get("registration_rate_pct"),
                "delta_pct": round(sparse_metrics.get("registration_rate_pct", 0) - base_sparse.get("registration_rate_pct", 0), 2),
            },
            "num_3d_points": {
                "baseline": base_sparse.get("num_3d_points"),
                "variant_e": sparse_metrics.get("num_3d_points"),
                "delta_abs": sparse_metrics.get("num_3d_points", 0) - base_sparse.get("num_3d_points", 0),
                "delta_pct": round(
                    ((sparse_metrics.get("num_3d_points", 0) - base_sparse.get("num_3d_points", 0)) / base_sparse.get("num_3d_points", 1)) * 100, 3
                ),
            },
            "mean_reprojection_error_px": {
                "baseline": base_sparse.get("mean_reprojection_error_px"),
                "variant_e": sparse_metrics.get("mean_reprojection_error_px"),
                "delta_abs": round(sparse_metrics.get("mean_reprojection_error_px", 0) - base_sparse.get("mean_reprojection_error_px", 0), 4),
            },
            "median_reprojection_error_px": {
                "baseline": base_sparse.get("median_reprojection_error_px"),
                "variant_e": sparse_metrics.get("median_reprojection_error_px"),
                "delta_abs": round(sparse_metrics.get("median_reprojection_error_px", 0) - base_sparse.get("median_reprojection_error_px", 0), 4),
            },
            "mean_observations_per_image": {
                "baseline": base_sparse.get("mean_observations_per_image"),
                "variant_e": sparse_metrics.get("mean_observations_per_image"),
                "delta_abs": round(sparse_metrics.get("mean_observations_per_image", 0) - base_sparse.get("mean_observations_per_image", 0), 2),
            },
            "mean_track_length": {
                "baseline": base_sparse.get("mean_track_length"),
                "variant_e": sparse_metrics.get("mean_track_length"),
                "delta_abs": round(sparse_metrics.get("mean_track_length", 0) - base_sparse.get("mean_track_length", 0), 4),
            },
            "two_view_inliers": {
                "baseline": base_db.get("two_view_inliers"),
                "variant_e": db_metrics.get("two_view_inliers"),
                "delta_abs": db_metrics.get("two_view_inliers", 0) - base_db.get("two_view_inliers", 0),
                "delta_pct": round(
                    ((db_metrics.get("two_view_inliers", 0) - base_db.get("two_view_inliers", 0)) / base_db.get("two_view_inliers", 1)) * 100, 3
                ),
            },
            "verified_pairs_count": {
                "baseline": base_db.get("verified_pairs_count"),
                "variant_e": db_metrics.get("verified_pairs_count"),
                "delta_abs": db_metrics.get("verified_pairs_count", 0) - base_db.get("verified_pairs_count", 0),
            },
            "database_size_mb": {
                "baseline": base_db.get("database_size_mb"),
                "variant_e": db_metrics.get("database_size_mb"),
                "delta_mb": round(db_metrics.get("database_size_mb", 0) - base_db.get("database_size_mb", 0), 2),
            },
            "stage_runtimes_s": {
                "feature_extraction": {
                    "baseline": base_timings.get("feature_extraction_s"),
                    "variant_e_incremental": timings.get("feature_extraction_incremental_s"),
                },
                "exhaustive_matching": {
                    "baseline": base_timings.get("exhaustive_matching_s"),
                    "variant_e": timings.get("exhaustive_matching_s"),
                },
                "sparse_mapping": {
                    "baseline": base_timings.get("sparse_mapping_s"),
                    "variant_e": timings.get("sparse_mapping_s"),
                },
                "model_conversion": {
                    "baseline": base_timings.get("model_conversion_s"),
                    "variant_e": timings.get("model_conversion_s"),
                },
            },
        }

        comp_json = VAR_E_DIR / "comparison_with_baseline.json"
        with open(comp_json, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison JSON to: {comp_json}")

    logger.info("=" * 70)
    logger.info("GRAHAM HALL VARIANT E RECONSTRUCTION COMPLETED SUCCESSFULLY")
    logger.info(f"Registered Images: {sparse_metrics.get('registered_images')} / 270")
    logger.info(f"3D Points: {sparse_metrics.get('num_3d_points')}")
    logger.info(f"Mean Reprojection Error: {sparse_metrics.get('mean_reprojection_error_px')} px")
    logger.info(f"Median Reprojection Error: {sparse_metrics.get('median_reprojection_error_px')} px")
    logger.info(f"Mean Observations / Image: {sparse_metrics.get('mean_observations_per_image')}")
    logger.info(f"Mean Track Length: {sparse_metrics.get('mean_track_length')}")
    logger.info(f"Verified Two-View Inliers: {db_metrics.get('two_view_inliers')}")
    logger.info(f"Database Size: {db_metrics.get('database_size_mb')} MB")
    logger.info(f"Total SfM Runtime: {timings['total_sfm_time_s']}s (~{timings['total_sfm_time_s']/3600:.2f}h)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
