"""
run_graham_hall_baseline.py

Executes Phase A: Graham Hall 270-Image Exterior Subset Baseline Reconstruction.

Scope:
  Dataset subset: datasets/graham-hall/Part2/images/exterior/IMG_0348.jpg to IMG_0617.jpg (270 images)
  Camera configuration: Single shared SIMPLE_RADIAL camera (--ImageReader.single_camera 1)
  Matching protocol: exhaustive_matcher (36,315 pairs)

Pipeline Stages:
  1. Image Staging & Strict Verification (MD5 byte-identity, OpenCV decodability, dimensions)
  2. colmap feature_extractor
  3. colmap exhaustive_matcher
  4. colmap mapper
  5. colmap model_converter (binary -> TXT)
  6. Database & Model Metric Extraction
  7. Contrast Analysis & Candidate Identification for Variant E

Outputs:
  Images: experiments/graham_hall/baseline/images/
  Database: experiments/graham_hall/baseline/database.db
  Sparse Model: experiments/graham_hall/baseline/sparse/0/
  Log: experiments/graham_hall/baseline/baseline.log
  Summary: experiments/graham_hall/baseline/baseline_summary.json
  SIFT Counts: experiments/graham_hall/baseline/baseline_sift_counts.csv
  Contrast CSV: experiments/graham_hall/baseline/image_contrast.csv
  Variant E Candidates: experiments/graham_hall/baseline/candidate_images_variant_e.csv
"""

import csv
import hashlib
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
import cv2
import numpy as np

PROJECT_ROOT = Path(r"E:\CV Project").resolve()
SRC_IMAGES_DIR = PROJECT_ROOT / "datasets" / "graham-hall" / "Part2" / "images" / "exterior"

EXP_DIR = PROJECT_ROOT / "experiments" / "graham_hall"
BASE_DIR = EXP_DIR / "baseline"
IMAGES_DIR = BASE_DIR / "images"
DATABASE_PATH = BASE_DIR / "database.db"
SPARSE_DIR = BASE_DIR / "sparse"
SPARSE_MODEL_DIR = SPARSE_DIR / "0"
LOG_PATH = BASE_DIR / "baseline.log"

COLMAP_EXE = PROJECT_ROOT / "colmap" / "bin" / "colmap.exe"
COLMAP_BIN_DIR = COLMAP_EXE.parent
COLMAP_PLUGINS_DIR = PROJECT_ROOT / "colmap" / "plugins"

START_IMG_NUM = 348
END_IMG_NUM = 617
EXPECTED_COUNT = 270


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("graham_hall_baseline")
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


def prepare_and_verify_images(logger: logging.Logger) -> List[Path]:
    logger.info("=" * 70)
    logger.info("STEP 1: PREPARING AND VERIFYING 270-IMAGE EXTERIOR SUBSET")
    logger.info(f"Source: {SRC_IMAGES_DIR}")
    logger.info(f"Destination: {IMAGES_DIR}")
    logger.info("=" * 70)

    if not SRC_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Source exterior image directory not found: {SRC_IMAGES_DIR}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Collect source files
    src_files = []
    for num in range(START_IMG_NUM, END_IMG_NUM + 1):
        candidates = list(SRC_IMAGES_DIR.glob(f"IMG_{num:04d}.*"))
        if not candidates:
            candidates = list(SRC_IMAGES_DIR.glob(f"img_{num:04d}.*"))
        if not candidates:
            raise FileNotFoundError(f"Missing required source image for index {num:04d} in {SRC_IMAGES_DIR}")
        src_files.append(candidates[0])

    if len(src_files) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} images, but collected {len(src_files)}")

    logger.info(f"Located all {EXPECTED_COUNT} continuous source images ({src_files[0].name} to {src_files[-1].name}).")

    # Copy files if not already present or if hash mismatch
    copied_count = 0
    for sf in src_files:
        df = IMAGES_DIR / sf.name
        needs_copy = True
        if df.exists():
            h_src = hashlib.md5(sf.read_bytes()).hexdigest()
            h_dst = hashlib.md5(df.read_bytes()).hexdigest()
            if h_src == h_dst:
                needs_copy = False
        if needs_copy:
            shutil.copy2(sf, df)
            copied_count += 1

    logger.info(f"Staged images in destination: {copied_count} newly copied, {EXPECTED_COUNT - copied_count} verified existing.")

    # Strict Validation
    dest_files = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.JPG")), key=lambda x: x.name)
    seen = set()
    unique_dest = [f for f in dest_files if not (f.name.upper() in seen or seen.add(f.name.upper()))]

    if len(unique_dest) != EXPECTED_COUNT:
        raise ValueError(f"Validation failed: expected {EXPECTED_COUNT} unique images, found {len(unique_dest)}")

    logger.info("Performing byte-identity and decodability validation across all 270 images...")
    for idx, (sf, df) in enumerate(zip(src_files, unique_dest)):
        h_src = hashlib.md5(sf.read_bytes()).hexdigest()
        h_dst = hashlib.md5(df.read_bytes()).hexdigest()
        if h_src != h_dst:
            raise ValueError(f"Byte mismatch on image {df.name}!")

        # Verify decoding and shape on subset samples
        if idx % 30 == 0 or idx == EXPECTED_COUNT - 1:
            im = cv2.imread(str(df))
            if im is None:
                raise ValueError(f"OpenCV failed to decode image: {df.name}")
            if im.shape[0] < 3000 or im.shape[1] < 3000:
                raise ValueError(f"Unexpected image dimensions for {df.name}: {im.shape}")

    logger.info("[PASS] 100% byte-identity and decodability verified for all 270 images.")
    return unique_dest


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
        "registration_rate_pct": round((num_reg_images / EXPECTED_COUNT) * 100.0, 2),
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


def analyze_image_contrast(images_dir: Path, sift_dict: Dict[str, int], logger: logging.Logger) -> Dict:
    logger.info("Computing grayscale contrast standard deviation for all 270 images...")
    contrast_records = []
    candidates = []

    contrast_csv = BASE_DIR / "image_contrast.csv"
    candidate_csv = BASE_DIR / "candidate_images_variant_e.csv"

    images_list = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG")), key=lambda x: x.name)
    seen = set()
    unique_imgs = [f for f in images_list if not (f.name.upper() in seen or seen.add(f.name.upper()))]

    for f in unique_imgs:
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        c_std = round(float(np.std(im)), 2)
        sift_cnt = sift_dict.get(f.name, 0)
        contrast_records.append((f.name, c_std, sift_cnt))

        # Dual-constraint selection rule
        if c_std < 52.0 and sift_cnt < 10000:
            candidates.append({"Image": f.name, "Contrast_StdDev": c_std, "SIFT_Features": sift_cnt})

    # Save contrast CSV
    with open(contrast_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Contrast_StdDev", "SIFT_Features"])
        for r in contrast_records:
            writer.writerow(r)
    logger.info(f"Saved image contrast measurements to: {contrast_csv}")

    # Save candidates CSV
    with open(candidate_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Contrast_StdDev", "SIFT_Features"])
        for c in candidates:
            writer.writerow([c["Image"], c["Contrast_StdDev"], c["SIFT_Features"]])
    logger.info(f"Identified {len(candidates)} Variant E candidates satisfying dual criteria: {candidate_csv}")
    for c in candidates:
        logger.info(f"  Candidate: {c['Image']} (Contrast={c['Contrast_StdDev']}, SIFT={c['SIFT_Features']})")

    return {
        "total_evaluated": len(contrast_records),
        "candidates_count": len(candidates),
        "candidates": candidates,
    }


def main():
    logger = setup_logging(LOG_PATH)
    logger.info("=" * 70)
    logger.info("PHASE 7: GRAHAM HALL 270-IMAGE SUBSET BASELINE RECONSTRUCTION")
    logger.info(f"Images Directory: {IMAGES_DIR}")
    logger.info(f"Database Path: {DATABASE_PATH}")
    logger.info(f"Sparse Output: {SPARSE_DIR}")
    logger.info(f"Log Path: {LOG_PATH}")
    logger.info("=" * 70)

    # 1. Prepare and verify images
    images_list = prepare_and_verify_images(logger)

    # 2. Reset database and sparse directories safely for fresh isolated run
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()
        logger.warning(f"Reset existing database: {DATABASE_PATH}")
    if SPARSE_DIR.exists():
        shutil.rmtree(SPARSE_DIR)
        logger.warning(f"Reset existing sparse directory: {SPARSE_DIR}")
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)

    timings = {}
    total_start = time.time()

    # Stage 1: Feature Extraction (Single Camera SIMPLE_RADIAL)
    timings["feature_extraction_s"] = stream_colmap_stage(
        subcommand="feature_extractor",
        args=[
            "--database_path", str(DATABASE_PATH),
            "--image_path", str(IMAGES_DIR),
            "--ImageReader.single_camera", "1",
        ],
        logger=logger,
        stage_name="Feature Extraction (SIFT, Single Camera)",
    )

    # Stage 2: Exhaustive Feature Matching
    timings["exhaustive_matching_s"] = stream_colmap_stage(
        subcommand="exhaustive_matcher",
        args=[
            "--database_path", str(DATABASE_PATH),
        ],
        logger=logger,
        stage_name="Exhaustive Feature Matching (36,315 pairs)",
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
    sift_csv = BASE_DIR / "baseline_sift_counts.csv"
    with open(sift_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "SIFT_Features"])
        for name in sorted(sift_dict.keys()):
            writer.writerow([name, sift_dict[name]])
    logger.info(f"Saved SIFT feature counts for {len(sift_dict)} images to: {sift_csv}")

    # Stage 6: Contrast Analysis and Variant E Candidates
    candidate_info = analyze_image_contrast(IMAGES_DIR, sift_dict, logger)

    summary = {
        "dataset": "graham-hall-270-exterior-subset",
        "scope": "270-image exterior sequence (IMG_0348.jpg to IMG_0617.jpg)",
        "total_images": len(images_list),
        "sparse_model": sparse_metrics,
        "database_metrics": db_metrics,
        "timings": timings,
        "variant_e_candidates_count": candidate_info["candidates_count"],
        "variant_e_candidates": candidate_info["candidates"],
    }

    summary_json = BASE_DIR / "baseline_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved baseline summary to: {summary_json}")

    logger.info("=" * 70)
    logger.info("GRAHAM HALL BASELINE RECONSTRUCTION COMPLETED SUCCESSFULLY")
    logger.info(f"Registered Images: {sparse_metrics.get('registered_images')} / {len(images_list)}")
    logger.info(f"3D Points: {sparse_metrics.get('num_3d_points')}")
    logger.info(f"Mean Reprojection Error: {sparse_metrics.get('mean_reprojection_error_px')} px")
    logger.info(f"Median Reprojection Error: {sparse_metrics.get('median_reprojection_error_px')} px")
    logger.info(f"Mean Observations / Image: {sparse_metrics.get('mean_observations_per_image')}")
    logger.info(f"Mean Track Length: {sparse_metrics.get('mean_track_length')}")
    logger.info(f"Verified Two-View Inliers: {db_metrics.get('two_view_inliers')}")
    logger.info(f"Database Size: {db_metrics.get('database_size_mb')} MB")
    logger.info(f"Total SfM Runtime: {timings['total_sfm_time_s']}s (~{timings['total_sfm_time_s']/3600:.2f}h)")
    logger.info(f"Variant E Candidates Identified: {candidate_info['candidates_count']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
