"""
resume_person_hall_variant_e.py

Resumes Person Hall Variant E from the completed exhaustive-matching checkpoint.
Executes:
  1. Strict pre-flight checks (preserves all existing outputs, checks DB integrity & disk space)
  2. colmap mapper (incremental sparse reconstruction)
  3. colmap model_converter (binary -> TXT)
  4. Metric extraction from sparse model & SQLite database
  5. Summary & comparison generation (JSON & CSV)
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
BASE_DIR = PROJECT_ROOT / "experiments" / "person_hall" / "baseline"
BASE_DB_PATH = BASE_DIR / "database.db"
BASE_SUMMARY_PATH = BASE_DIR / "baseline_summary.json"

VAR_E_DIR = PROJECT_ROOT / "experiments" / "person_hall" / "variant_e"
IMAGES_DIR = VAR_E_DIR / "images"
DATABASE_PATH = VAR_E_DIR / "database.db"
SPARSE_DIR = VAR_E_DIR / "sparse"
SPARSE_MODEL_DIR = SPARSE_DIR / "0"
LOG_PATH = VAR_E_DIR / "variant_e.log"
RESUME_LOG_PATH = VAR_E_DIR / "resume.log"

COLMAP_EXE = PROJECT_ROOT / "colmap" / "bin" / "colmap.exe"
COLMAP_BIN_DIR = COLMAP_EXE.parent
COLMAP_PLUGINS_DIR = PROJECT_ROOT / "colmap" / "plugins"

HISTORICAL_FEATURE_EXTRACTION_S = 5.7  # 5 enhanced images extraction time; baseline extraction was 173.7s
HISTORICAL_EXHAUSTIVE_MATCHING_S = 13037.5  # Measured matching runtime from completed run

SELECTED_IMAGES = [
    "IMG_1179.JPG",
    "IMG_1282.JPG",
    "IMG_1312.JPG",
    "IMG_1316.JPG",
    "IMG_1323.JPG",
]


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("resume_person_hall_variant_e")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Resume log handler
    fh_resume = logging.FileHandler(RESUME_LOG_PATH, mode="a", encoding="utf-8")
    fh_resume.setLevel(logging.DEBUG)
    fh_resume.setFormatter(fmt)
    logger.addHandler(fh_resume)

    # Append to existing variant_e.log
    fh_main = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh_main.setLevel(logging.DEBUG)
    fh_main.setFormatter(fmt)
    logger.addHandler(fh_main)

    return logger


def check_preflight(logger: logging.Logger) -> None:
    logger.info("=" * 70)
    logger.info("PRE-FLIGHT VALIDATION - RESUMING PERSON HALL VARIANT E")
    logger.info("=" * 70)

    # 1. Images check
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Variant E images dir missing: {IMAGES_DIR}")
    images = list(IMAGES_DIR.glob("*.JPG"))
    if len(images) != 330:
        raise ValueError(f"Expected 330 images in {IMAGES_DIR}, found {len(images)}")
    for name in SELECTED_IMAGES:
        if not (IMAGES_DIR / name).exists():
            raise FileNotFoundError(f"Enhanced image missing: {name}")
    logger.info(f"[PASS] Images verified: {len(images)}/330 present including all 5 enhanced images.")

    # 2. Database integrity & contents
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Variant E database missing: {DATABASE_PATH}")
    conn = sqlite3.connect(str(DATABASE_PATH))
    c = conn.cursor()
    c.execute("PRAGMA integrity_check;")
    chk = c.fetchall()
    if chk != [("ok",)]:
        raise RuntimeError(f"Database integrity check failed: {chk}")
    logger.info("[PASS] SQLite PRAGMA integrity_check: ok.")

    c.execute("SELECT COUNT(*) FROM images;")
    img_cnt = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM cameras;")
    cam_cnt = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM keypoints;")
    kp_cnt = c.fetchone()[0]
    c.execute("SELECT COUNT(*), SUM(rows) FROM matches WHERE rows > 0;")
    m_res = c.fetchone()
    c.execute("SELECT COUNT(*), SUM(rows) FROM two_view_geometries WHERE rows > 0;")
    tv_res = c.fetchone()
    conn.close()

    if img_cnt != 330 or cam_cnt != 1 or kp_cnt != 330:
        raise ValueError(f"Unexpected database entity counts: images={img_cnt}, cameras={cam_cnt}, keypoints={kp_cnt}")
    if tv_res[0] == 0 or tv_res[1] == 0:
        raise ValueError("Database has no two_view_geometries! Matching was not complete.")
    logger.info(f"[PASS] Database verified: {img_cnt} images, {cam_cnt} camera, {kp_cnt} keypoint sets.")
    logger.info(f"[PASS] Matches verified: {m_res[0]} raw pairs ({m_res[1]} rows), {tv_res[0]} verified pairs ({tv_res[1]} inliers).")

    # 3. Preserved artifacts
    manifest_path = VAR_E_DIR / "enhancement_manifest.json"
    enh_log_path = VAR_E_DIR / "enhancement_log.csv"
    if not manifest_path.exists() or not enh_log_path.exists() or not LOG_PATH.exists():
        raise FileNotFoundError("Existing enhancement manifest, log, or variant_e.log missing!")
    logger.info("[PASS] Preserved manifests and logs verified.")

    # 4. Check disk space
    free_bytes = shutil.disk_usage(str(PROJECT_ROOT)).free
    free_gb = free_bytes / (1024 ** 3)
    logger.info(f"[PASS] Available disk space on drive {PROJECT_ROOT.drive}: {free_gb:.2f} GB (Requirement: > 10 GB).")
    if free_gb < 10:
        raise RuntimeError(f"Low disk space: {free_gb:.2f} GB remaining.")

    # 5. Check no colmap process is running
    try:
        proc_check = subprocess.run(["tasklist", "/FI", "IMAGENAME eq colmap.exe"], capture_output=True, text=True)
        if "colmap.exe" in proc_check.stdout:
            raise RuntimeError("Another COLMAP process is currently running! Aborting.")
        logger.info("[PASS] No competing COLMAP process running.")
    except Exception as e:
        logger.warning(f"Process check note: {e}")

    logger.info("All pre-flight checks passed successfully. Ready to run mapper.")


def stream_colmap_stage(subcommand: str, args: list, logger: logging.Logger, stage_name: str) -> float:
    cmd = [str(COLMAP_EXE), subcommand] + args

    env = os.environ.copy()
    env["PATH"] = str(COLMAP_BIN_DIR) + os.pathsep + env.get("PATH", "")
    env["QT_PLUGIN_PATH"] = str(COLMAP_PLUGINS_DIR)

    cmd_display = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    logger.info("=" * 70)
    logger.info(f"RESUMING STAGE: {stage_name}")
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
            # Log every line to file, and filter key milestones to console
            logger.debug(f"  [colmap] {line_str}")
            if any(k in line_str for k in [
                "=>", "Registering image", "Bundle adjustment", "Retriangulation",
                "Number of registered images", "Number of points", "Elapsed time",
                "Initialization", "Extracting", "Writing"
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
        "registration_rate_pct": round((num_reg_images / 330.0) * 100.0, 2),
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
    logger = setup_logger()
    check_preflight(logger)

    SPARSE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Sparse Reconstruction / SfM (mapper)
    t_mapper = stream_colmap_stage(
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
        raise RuntimeError(f"Mapper completed but model dir does not exist: {SPARSE_MODEL_DIR}")

    # Check for binary model files
    bin_files = ["cameras.bin", "images.bin", "points3D.bin"]
    for bf in bin_files:
        if not (SPARSE_MODEL_DIR / bf).exists():
            raise FileNotFoundError(f"Binary model file missing: {bf} in {SPARSE_MODEL_DIR}")
    logger.info(f"Verified binary model files in {SPARSE_MODEL_DIR}")

    # 2. Model Conversion (binary -> TXT)
    t_converter = stream_colmap_stage(
        subcommand="model_converter",
        args=[
            "--input_path", str(SPARSE_MODEL_DIR),
            "--output_path", str(SPARSE_MODEL_DIR),
            "--output_type", "TXT",
        ],
        logger=logger,
        stage_name="Model Conversion (binary -> TXT)",
    )

    # Verify TXT files
    txt_files = ["cameras.txt", "images.txt", "points3D.txt"]
    for tf in txt_files:
        if not (SPARSE_MODEL_DIR / tf).exists():
            raise FileNotFoundError(f"TXT model file missing: {tf} in {SPARSE_MODEL_DIR}")
    logger.info(f"Verified TXT model files in {SPARSE_MODEL_DIR}")

    # 3. Calculate and Save Metrics
    logger.info("Extracting database and model metrics...")
    db_metrics = extract_database_metrics(DATABASE_PATH)
    sparse_metrics = parse_sparse_model(SPARSE_MODEL_DIR)

    # Save SIFT counts CSV
    sift_dict = db_metrics.pop("sift_dict", {})
    sift_csv_path = VAR_E_DIR / "variant_e_sift_counts.csv"
    with open(sift_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "SIFT_Features"])
        for img_name in sorted(sift_dict.keys()):
            writer.writerow([img_name, sift_dict[img_name]])
    logger.info(f"Saved SIFT feature counts for {len(sift_dict)} images to {sift_csv_path}")

    # Build timings dict with clear historical labels
    timings = {
        "historical_feature_extraction_5_enhanced_s": HISTORICAL_FEATURE_EXTRACTION_S,
        "baseline_feature_extraction_reference_s": 173.7,
        "historical_exhaustive_matching_s": HISTORICAL_EXHAUSTIVE_MATCHING_S,
        "measured_sparse_mapping_s": round(t_mapper, 1),
        "measured_model_conversion_s": round(t_converter, 1),
        "total_sfm_pipeline_reconstructed_s": round(
            HISTORICAL_FEATURE_EXTRACTION_S + HISTORICAL_EXHAUSTIVE_MATCHING_S + t_mapper + t_converter, 1
        ),
    }

    summary = {
        "dataset": "person-hall",
        "variant": "variant_e_conservative_selection",
        "total_images": 330,
        "enhanced_images_count": 5,
        "unchanged_images_count": 325,
        "sparse_model": sparse_metrics,
        "database_metrics": db_metrics,
        "timings": timings,
    }

    summary_json_path = VAR_E_DIR / "variant_e_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved Variant E summary to {summary_json_path}")

    # 4. Compare with Baseline
    if BASE_SUMMARY_PATH.exists():
        with open(BASE_SUMMARY_PATH, "r", encoding="utf-8") as f:
            base_sum = json.load(f)

        base_sparse = base_sum.get("sparse_model", {})
        base_db = base_sum.get("database_metrics", {})
        base_timings = base_sum.get("timings", {})

        comparison = {
            "dataset": "person-hall",
            "registered_images": {
                "baseline": base_sparse.get("registered_images"),
                "variant_e": sparse_metrics.get("registered_images"),
                "delta": sparse_metrics.get("registered_images", 0) - base_sparse.get("registered_images", 0),
            },
            "registration_rate_pct": {
                "baseline": round((base_sparse.get("registered_images", 0) / 330.0) * 100.0, 2),
                "variant_e": sparse_metrics.get("registration_rate_pct"),
                "delta_pct": round(sparse_metrics.get("registration_rate_pct", 0) - (base_sparse.get("registered_images", 0) / 330.0) * 100.0, 2),
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
                    "variant_e_incremental": HISTORICAL_FEATURE_EXTRACTION_S,
                },
                "exhaustive_matching": {
                    "baseline": base_timings.get("exhaustive_matching_s"),
                    "variant_e_historical": HISTORICAL_EXHAUSTIVE_MATCHING_S,
                },
                "sparse_mapping": {
                    "baseline": base_timings.get("sparse_mapping_s"),
                    "variant_e_measured": round(t_mapper, 1),
                },
                "model_conversion": {
                    "baseline": base_timings.get("model_conversion_s"),
                    "variant_e_measured": round(t_converter, 1),
                },
            },
        }

        comp_json_path = VAR_E_DIR / "comparison_with_baseline.json"
        with open(comp_json_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison to {comp_json_path}")

    logger.info("=" * 70)
    logger.info("RESUMPTION OF PERSON HALL VARIANT E COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
