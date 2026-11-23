"""
Phase 6 — COLMAP Ablation Runner

Runs COLMAP reconstruction for each ablation variant.
Each run gets its own output directory under experiments/phase6/.

This script should only be run AFTER the ablation plan has been reviewed
and approved. Each run takes ~33 minutes for South Building.

Variants:
  variant_a_clip2.0 — CLAHE clipLimit=2.0 (current setting)
  variant_b_clip1.5 — CLAHE clipLimit=1.5 (mild)
  variant_c_clip1.2 — CLAHE clipLimit=1.2 (conservative)

Usage:
    python experiments/phase6/run_ablation_colmap.py --variant variant_a_clip2.0
    python experiments/phase6/run_ablation_colmap.py --variant variant_b_clip1.5
    python experiments/phase6/run_ablation_colmap.py --variant variant_c_clip1.2
    python experiments/phase6/run_ablation_colmap.py --variant all
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE6_DIR = PROJECT_ROOT / "experiments" / "phase6"
COLMAP_BAT = PROJECT_ROOT / "colmap" / "COLMAP.bat"

ALL_VARIANTS = ["variant_a_clip2.0", "variant_b_clip1.5", "variant_c_clip1.2", "variant_e_conservative_selection"]


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ablation_colmap")
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


def run_colmap_stage(colmap_bat: Path, subcommand: str, args: list,
                     logger: logging.Logger, stage_name: str) -> float:
    """Execute a COLMAP subcommand. Returns elapsed time in seconds."""
    colmap_exe = colmap_bat.parent / "bin" / "colmap.exe"
    colmap_bin_dir = colmap_exe.parent
    colmap_plugins_dir = colmap_bat.parent / "plugins"

    cmd = [str(colmap_exe), subcommand] + args

    env = os.environ.copy()
    env["PATH"] = str(colmap_bin_dir) + os.pathsep + env.get("PATH", "")
    env["QT_PLUGIN_PATH"] = str(colmap_plugins_dir)

    cmd_display = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    logger.info("=" * 70)
    logger.info(f"STAGE: {stage_name}")
    logger.info(f"Command: {cmd_display}")
    logger.info("=" * 70)

    start = time.time()

    if not colmap_exe.exists():
        raise FileNotFoundError(f"colmap.exe not found at: {colmap_exe}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    elapsed = time.time() - start

    if result.stdout.strip():
        for line in result.stdout.splitlines():
            logger.debug(f"  [stdout] {line}")
    if result.stderr.strip():
        for line in result.stderr.splitlines():
            logger.debug(f"  [stderr] {line}")

    if result.returncode != 0:
        logger.error(f"STAGE FAILED: {stage_name} (exit code {result.returncode})")
        if result.stdout.strip():
            for line in result.stdout.splitlines():
                logger.error(f"  {line}")
        if result.stderr.strip():
            for line in result.stderr.splitlines():
                logger.error(f"  {line}")
        raise RuntimeError(f"COLMAP '{subcommand}' failed with exit code {result.returncode}")

    logger.info(f"STAGE COMPLETE: {stage_name} ({elapsed:.1f}s)")
    return elapsed


def run_variant(variant_name: str, logger: logging.Logger) -> dict:
    """Run full COLMAP pipeline for one variant."""
    variant_dir = PHASE6_DIR / variant_name
    images_dir = variant_dir / "images"
    colmap_dir = variant_dir / "colmap_output"
    database_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_model_dir = sparse_dir / "0"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Check if already completed
    if (sparse_model_dir / "points3D.txt").exists():
        logger.info(f"Variant {variant_name} already has reconstruction output. Skipping.")
        logger.info(f"  To re-run, delete: {colmap_dir}")
        return {"variant": variant_name, "status": "skipped", "message": "Already completed"}

    colmap_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale database if exists
    if database_path.exists():
        database_path.unlink()
        logger.warning(f"Removed stale database: {database_path}")

    timings = {}
    total_start = time.time()

    logger.info(f"\n{'#' * 70}")
    logger.info(f"COLMAP RECONSTRUCTION: {variant_name}")
    logger.info(f"Images: {images_dir}")
    logger.info(f"Output: {colmap_dir}")
    logger.info(f"{'#' * 70}\n")

    # Stage 1: Feature Extraction
    timings["Feature Extraction (SIFT)"] = run_colmap_stage(
        COLMAP_BAT, "feature_extractor",
        ["--database_path", str(database_path),
         "--image_path", str(images_dir),
         "--ImageReader.single_camera", "1"],
        logger, "Feature Extraction (SIFT)"
    )

    # Stage 2: Exhaustive Matching
    timings["Exhaustive Feature Matching"] = run_colmap_stage(
        COLMAP_BAT, "exhaustive_matcher",
        ["--database_path", str(database_path)],
        logger, "Exhaustive Feature Matching"
    )

    # Stage 3: Sparse Reconstruction
    sparse_dir.mkdir(parents=True, exist_ok=True)
    timings["Sparse Reconstruction (mapper)"] = run_colmap_stage(
        COLMAP_BAT, "mapper",
        ["--database_path", str(database_path),
         "--image_path", str(images_dir),
         "--output_path", str(sparse_dir)],
        logger, "Sparse Reconstruction / SfM (mapper)"
    )

    if not sparse_model_dir.exists():
        raise RuntimeError(f"Mapper produced no reconstruction at {sparse_model_dir}")

    # Stage 4: Model Conversion
    timings["Model Conversion (binary -> TXT)"] = run_colmap_stage(
        COLMAP_BAT, "model_converter",
        ["--input_path", str(sparse_model_dir),
         "--output_path", str(sparse_model_dir),
         "--output_type", "TXT"],
        logger, "Model Conversion (binary -> TXT)"
    )

    total_elapsed = time.time() - total_start
    timings["total_sfm_time_s"] = round(total_elapsed, 1)

    # Save timing summary
    import json
    timing_path = variant_dir / "colmap_timings.json"
    with open(timing_path, "w") as f:
        json.dump(timings, f, indent=2)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"COLMAP COMPLETE: {variant_name}")
    logger.info(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"  Sparse model: {sparse_model_dir}")
    logger.info(f"  Timings saved: {timing_path}")
    logger.info(f"{'=' * 70}\n")

    return {"variant": variant_name, "status": "completed", "timings": timings}


def main():
    parser = argparse.ArgumentParser(description="Phase 6: COLMAP Ablation Runner")
    parser.add_argument("--variant", required=True,
                        help="Variant to run: variant_a_clip2.0, variant_b_clip1.5, variant_c_clip1.2, or 'all'")
    args = parser.parse_args()

    log_path = PHASE6_DIR / "ablation_colmap.log"
    logger = setup_logging(log_path)

    variants_to_run = ALL_VARIANTS if args.variant == "all" else [args.variant]

    for vname in variants_to_run:
        if vname not in ALL_VARIANTS:
            logger.error(f"Unknown variant: {vname}. Available: {ALL_VARIANTS}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("PHASE 6: COLMAP ABLATION RUNNER — START")
    logger.info(f"Variants: {variants_to_run}")
    logger.info(f"COLMAP: {COLMAP_BAT}")
    logger.info("=" * 70)

    results = []
    for vname in variants_to_run:
        try:
            result = run_variant(vname, logger)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed on {vname}: {e}")
            results.append({"variant": vname, "status": "failed", "error": str(e)})

    logger.info("\n" + "=" * 70)
    logger.info("ABLATION RUNNER — SUMMARY")
    for r in results:
        logger.info(f"  {r['variant']}: {r['status']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
