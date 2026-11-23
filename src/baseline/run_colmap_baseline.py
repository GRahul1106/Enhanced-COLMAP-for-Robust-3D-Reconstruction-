"""
run_colmap_baseline.py

Runs the standard COLMAP reconstruction pipeline (feature extraction,
feature matching, sparse reconstruction, and optionally dense reconstruction)
on a given dataset WITHOUT modifying the original dataset in any way.

All generated files are written to the specified output directory.

Usage:
    python src/baseline/run_colmap_baseline.py \
        --dataset   "e:\\CV Project\\datasets\\gerrard-hall" \
        --output    "e:\\CV Project\\experiments\\baseline\\gerrard-hall" \
        [--colmap_bat "e:\\CV Project\\colmap\\COLMAP.bat"] \
        [--dense] \
        [--overwrite] \
        [--copy_to_results]

Requirements:
    Python 3.8+  —  no third-party packages required (standard library only).
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    """
    Configure a logger that writes to both the console and a log file.
    The log file is created at log_path.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("colmap_baseline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — DEBUG and above (captures full COLMAP stdout/stderr)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# COLMAP subprocess runner
# ---------------------------------------------------------------------------

def run_colmap(
    colmap_bat: Path,
    subcommand: str,
    args: list,
    logger: logging.Logger,
    stage_name: str,
) -> None:
    """
    Execute a single COLMAP subcommand via subprocess.

    Windows path-with-spaces solution
    ----------------------------------
    Calling COLMAP.bat through 'cmd /c' fails when the project path contains
    spaces (e.g. "E:\\CV Project\\") because Windows cmd.exe splits the path
    at the space and cannot find the executable.

    The robust fix is to bypass COLMAP.bat entirely and invoke colmap.exe
    directly as a Python list — subprocess.run() with a list never goes
    through cmd.exe shell parsing, so spaces in the path are handled
    correctly without any quoting.

    To replicate what COLMAP.bat does (set PATH and QT_PLUGIN_PATH), we
    inject those values into the subprocess environment ourselves.

    Parameters
    ----------
    colmap_bat  : full path to COLMAP.bat  (used to locate colmap.exe)
    subcommand  : e.g. "feature_extractor"
    args        : list of --flag value pairs as strings
    logger      : active Logger instance
    stage_name  : human-readable label used in log messages
    """
    # Derive colmap.exe path from COLMAP.bat location:
    #   COLMAP.bat lives in  <install>/
    #   colmap.exe lives in  <install>/bin/
    colmap_exe = colmap_bat.parent / "bin" / "colmap.exe"
    colmap_bin_dir = colmap_exe.parent
    colmap_plugins_dir = colmap_bat.parent / "plugins"

    # Build the subprocess command as a list — no shell parsing, spaces safe.
    cmd = [str(colmap_exe), subcommand] + args

    # Extend the process environment so colmap.exe can find its DLLs and
    # Qt plugins (mirrors what COLMAP.bat sets via set PATH=... and
    # set QT_PLUGIN_PATH=...).
    env = os.environ.copy()
    env["PATH"] = str(colmap_bin_dir) + os.pathsep + env.get("PATH", "")
    env["QT_PLUGIN_PATH"] = str(colmap_plugins_dir)

    # Human-readable display string for logging
    cmd_display = " ".join(f'"{c}"' if " " in c else c for c in cmd)

    logger.info("=" * 70)
    logger.info(f"STAGE: {stage_name}")
    logger.info(f"Command: {cmd_display}")
    logger.info("=" * 70)

    start = datetime.now()

    if not colmap_exe.exists():
        raise FileNotFoundError(
            f"colmap.exe not found at:\n  {colmap_exe}\n"
            "Verify COLMAP is installed and use --colmap_bat to set the correct path."
        )

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",   # never crash on unexpected characters in output
            env=env,            # PATH + QT_PLUGIN_PATH injected above
        )
    except FileNotFoundError:
        logger.error(
            f"Could not launch colmap.exe at:\n  {colmap_exe}\n"
            "Verify COLMAP is installed and use --colmap_bat to set the correct path."
        )
        raise

    elapsed = (datetime.now() - start).total_seconds()

    # Log stdout/stderr at DEBUG level — always visible in the log file
    if result.stdout.strip():
        for line in result.stdout.splitlines():
            logger.debug(f"  [stdout] {line}")
    if result.stderr.strip():
        for line in result.stderr.splitlines():
            logger.debug(f"  [stderr] {line}")

    if result.returncode != 0:
        logger.error(f"STAGE FAILED: {stage_name}  (exit code {result.returncode})")
        logger.error(f"Failed command:\n  {cmd_display}")
        # Re-emit stdout/stderr at ERROR so they appear on the console too
        if result.stdout.strip():
            logger.error("--- stdout ---")
            for line in result.stdout.splitlines():
                logger.error(f"  {line}")
        if result.stderr.strip():
            logger.error("--- stderr ---")
            for line in result.stderr.splitlines():
                logger.error(f"  {line}")
        raise RuntimeError(
            f"COLMAP '{subcommand}' failed with exit code {result.returncode}. "
            "See the log file for full details."
        )

    logger.info(f"STAGE COMPLETE: {stage_name}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Path validation helpers
# ---------------------------------------------------------------------------

def validate_inputs(
    dataset_dir: Path,
    output_dir: Path,
    colmap_bat: Path,
    logger: logging.Logger,
) -> Path:
    """
    Verify that the dataset directory and COLMAP.bat exist, locate the images
    subdirectory, and confirm the output directory won't overlap with the dataset.

    Returns the resolved images directory path.
    Raises FileNotFoundError or ValueError on any problem.
    """
    # --- Dataset ---
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    images_dir = _find_images_dir(dataset_dir, logger)

    # --- Safety: output must NOT be inside the dataset directory ---
    try:
        output_dir.resolve().relative_to(dataset_dir.resolve())
        # If we reach here, output IS under dataset — that's forbidden
        raise ValueError(
            f"Output directory '{output_dir}' is inside the dataset directory "
            f"'{dataset_dir}'. Choose a different --output location to avoid "
            "writing to the original dataset."
        )
    except ValueError as e:
        if "is inside" in str(e):
            raise
        # relative_to() raised ValueError because output is NOT under dataset — good

    # --- COLMAP ---
    if not colmap_bat.exists():
        raise FileNotFoundError(
            f"COLMAP.bat not found at:\n  {colmap_bat}\n"
            "Use --colmap_bat to specify the correct path."
        )

    return images_dir


def _find_images_dir(dataset_dir: Path, logger: logging.Logger) -> Path:
    """
    Locate the directory that directly contains image files within dataset_dir.

    Search strategy:
    1. dataset_dir/images/  contains image files directly   → use it
    2. dataset_dir/images/  contains subdirs with images    → use first subdir
       (handles graham-hall layout: images/exterior/)
    3. dataset_dir itself   contains image files            → use it
    4. Nothing found                                        → raise error
    """
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def has_images(d: Path) -> bool:
        return any(
            f.suffix.lower() in IMAGE_EXTENSIONS
            for f in d.iterdir()
            if f.is_file()
        )

    candidate = dataset_dir / "images"
    if candidate.exists() and candidate.is_dir():
        if has_images(candidate):
            logger.info(f"Images directory located: {candidate}")
            return candidate
        # One level deeper (e.g. graham-hall/images/exterior/)
        for sub in sorted(candidate.iterdir()):
            if sub.is_dir() and has_images(sub):
                logger.info(
                    f"Images directory located (subdirectory): {sub}\n"
                    "         COLMAP will scan this path recursively."
                )
                return sub

    # Last resort: dataset root itself
    if has_images(dataset_dir):
        logger.warning(
            f"No images/ subdirectory found. Using dataset root: {dataset_dir}"
        )
        return dataset_dir

    raise FileNotFoundError(
        f"Could not find a directory containing image files under: {dataset_dir}\n"
        f"Expected: {candidate}"
    )


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_feature_extraction(
    colmap_bat: Path,
    database_path: Path,
    images_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Stage 1 — Feature Extraction

    Detects SIFT keypoints and computes descriptors for every image.
    Results are written into the COLMAP SQLite database.

    Reads:  images_dir  (READ-ONLY — dataset is never modified)
    Writes: database_path  (created inside output directory)
    """
    run_colmap(
        colmap_bat=colmap_bat,
        subcommand="feature_extractor",
        args=[
            "--database_path",              str(database_path),
            "--image_path",                 str(images_dir),
            "--ImageReader.single_camera",  "1",
        ],
        logger=logger,
        stage_name="Feature Extraction (SIFT)",
    )


def stage_feature_matching(
    colmap_bat: Path,
    database_path: Path,
    logger: logging.Logger,
) -> None:
    """
    Stage 2 — Exhaustive Feature Matching

    Matches features between all N*(N-1)/2 image pairs.
    Exhaustive matching is used for baseline reproducibility and is suitable
    for datasets up to a few hundred images.

    Reads:  database_path (features from stage 1)
    Writes: database_path (match results appended)
    """
    run_colmap(
        colmap_bat=colmap_bat,
        subcommand="exhaustive_matcher",
        args=[
            "--database_path", str(database_path),
        ],
        logger=logger,
        stage_name="Exhaustive Feature Matching",
    )


def stage_sparse_reconstruction(
    colmap_bat: Path,
    database_path: Path,
    images_dir: Path,
    sparse_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Stage 3 — Incremental Structure-from-Motion (mapper)

    Registers images incrementally, triangulates 3D points, and runs bundle
    adjustment to produce a sparse 3D reconstruction.

    Reads:  database_path, images_dir (READ-ONLY)
    Writes: sparse_dir/0/
                cameras.bin   — camera intrinsics (binary)
                images.bin    — camera poses      (binary)
                points3D.bin  — 3D points         (binary)
    """
    sparse_dir.mkdir(parents=True, exist_ok=True)

    run_colmap(
        colmap_bat=colmap_bat,
        subcommand="mapper",
        args=[
            "--database_path", str(database_path),
            "--image_path",    str(images_dir),
            "--output_path",   str(sparse_dir),
        ],
        logger=logger,
        stage_name="Sparse Reconstruction / SfM (mapper)",
    )


def stage_convert_model(
    colmap_bat: Path,
    sparse_model_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Stage 4 — Convert binary model to text format

    COLMAP's mapper outputs binary .bin files. This stage adds the equivalent
    human-readable .txt files alongside them so both formats are available for
    inspection and post-processing.

    Reads:  sparse_model_dir/*.bin  (binary model from mapper)
    Writes: sparse_model_dir/*.txt  (cameras.txt, images.txt, points3D.txt)
    """
    run_colmap(
        colmap_bat=colmap_bat,
        subcommand="model_converter",
        args=[
            "--input_path",  str(sparse_model_dir),
            "--output_path", str(sparse_model_dir),
            "--output_type", "TXT",
        ],
        logger=logger,
        stage_name="Model Conversion (binary -> TXT)",
    )


def stage_image_undistortion(
    colmap_bat: Path,
    images_dir: Path,
    sparse_model_dir: Path,
    dense_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Stage 5 (dense only) — Image Undistortion

    Undistorts images using the calibrated camera model from SfM, producing
    pinhole-camera images suitable for Multi-View Stereo.

    Reads:  images_dir (READ-ONLY), sparse_model_dir (camera model)
    Writes: dense_dir/
                images/   — undistorted copies of the input images
                sparse/   — undistorted camera model
    """
    dense_dir.mkdir(parents=True, exist_ok=True)

    run_colmap(
        colmap_bat=colmap_bat,
        subcommand="image_undistorter",
        args=[
            "--image_path",  str(images_dir),
            "--input_path",  str(sparse_model_dir),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
        ],
        logger=logger,
        stage_name="Image Undistortion (MVS prep)",
    )


def stage_patch_match_stereo(
    colmap_bat: Path,
    dense_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Stage 6 (dense only) — Patch Match Stereo (MVS)

    Computes per-image depth maps and normal maps using GPU-accelerated
    Multi-View Stereo.

    REQUIRES: NVIDIA GPU with CUDA support. This stage will fail clearly
    if no compatible GPU is detected.

    Reads:  dense_dir/ (undistorted images + camera model)
    Writes: dense_dir/stereo/
                depth_maps/
                normal_maps/
    """
    run_colmap(
        colmap_bat=colmap_bat,
        subcommand="patch_match_stereo",
        args=[
            "--workspace_path",                    str(dense_dir),
            "--workspace_format",                  "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true",
        ],
        logger=logger,
        stage_name="Patch Match Stereo / MVS (depth maps)",
    )


def stage_stereo_fusion(
    colmap_bat: Path,
    dense_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Stage 7 (dense only) — Stereo Fusion

    Fuses all per-image depth maps into a single dense point cloud (.ply file).

    Reads:  dense_dir/stereo/ (depth maps from patch match stereo)
    Writes: dense_dir/fused.ply
    """
    fused_ply = dense_dir / "fused.ply"

    run_colmap(
        colmap_bat=colmap_bat,
        subcommand="stereo_fusion",
        args=[
            "--workspace_path",   str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type",       "geometric",
            "--output_path",      str(fused_ply),
        ],
        logger=logger,
        stage_name="Stereo Fusion (dense point cloud → fused.ply)",
    )


# ---------------------------------------------------------------------------
# Results copy helper
# ---------------------------------------------------------------------------

def copy_sparse_to_results(
    sparse_dir: Path,
    dataset_name: str,
    results_root: Path,
    logger: logging.Logger,
) -> None:
    """
    Copy the sparse reconstruction files to results/original_colmap/<dataset_name>/
    for later side-by-side comparison with the enhanced pipeline output.

    Only copies .txt and .bin files from sparse/0/.
    """
    dest = results_root / "original_colmap" / dataset_name
    dest.mkdir(parents=True, exist_ok=True)

    sparse_model = sparse_dir / "0"
    if not sparse_model.exists():
        logger.warning(
            f"Sparse model not found at {sparse_model}. "
            "Skipping copy to results/."
        )
        return

    files = list(sparse_model.glob("*.txt")) + list(sparse_model.glob("*.bin"))
    if not files:
        logger.warning(f"No model files found in {sparse_model}. Skipping copy.")
        return

    for src_file in sorted(files):
        dst_file = dest / src_file.name
        shutil.copy2(src_file, dst_file)
        logger.debug(f"  Copied: {src_file.name} → {dst_file}")

    logger.info(
        f"Sparse model copied to results: {dest}  ({len(files)} files)"
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    # Resolve the default COLMAP path relative to this script:
    #   this file  →  src/baseline/run_colmap_baseline.py
    #   up 2 dirs  →  project root
    #   + colmap/COLMAP.bat
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    default_colmap_bat = project_root / "colmap" / "COLMAP.bat"

    parser = argparse.ArgumentParser(
        description=(
            "Run the standard COLMAP baseline reconstruction pipeline.\n"
            "The original dataset directory is never modified.\n\n"
            "Example:\n"
            "  python src/baseline/run_colmap_baseline.py \\\n"
            '      --dataset "datasets\\gerrard-hall" \\\n'
            '      --output  "experiments\\baseline\\gerrard-hall"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Path to the dataset directory. Must contain an images/ "
            "subdirectory (or images/exterior/ for graham-hall)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Directory where all pipeline outputs will be written "
            "(database.db, sparse/, dense/, baseline.log). "
            "Created automatically if it does not exist. "
            "Must NOT be inside --dataset."
        ),
    )
    parser.add_argument(
        "--colmap_bat",
        type=Path,
        default=default_colmap_bat,
        metavar="PATH",
        help=(
            "Path to COLMAP.bat. "
            f"Default: <project_root>/colmap/COLMAP.bat "
            f"(auto-resolved to: {default_colmap_bat})"
        ),
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        default=False,
        help=(
            "Also run dense reconstruction: image_undistorter → "
            "patch_match_stereo → stereo_fusion. "
            "Requires an NVIDIA GPU with CUDA support."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help=(
            "If the output directory already contains a database.db, "
            "delete it and the sparse/ directory and start fresh. "
            "Without this flag the script exits rather than overwrite work."
        ),
    )
    parser.add_argument(
        "--copy_to_results",
        action="store_true",
        default=False,
        help=(
            "After reconstruction, copy the sparse model to "
            "results/original_colmap/<dataset-name>/ for later comparison "
            "with the enhanced pipeline."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    dataset_dir: Path = args.dataset.resolve()
    output_dir: Path  = args.output.resolve()
    colmap_bat: Path  = args.colmap_bat.resolve()

    # ------------------------------------------------------------------
    # Set up output directory and logging
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "baseline.log"
    logger   = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("COLMAP BASELINE PIPELINE — START")
    logger.info("=" * 70)
    logger.info(f"  Dataset:       {dataset_dir}")
    logger.info(f"  Output:        {output_dir}")
    logger.info(f"  COLMAP:        {colmap_bat}")
    logger.info(f"  Dense:         {args.dense}")
    logger.info(f"  Overwrite:     {args.overwrite}")
    logger.info(f"  Copy results:  {args.copy_to_results}")
    logger.info(f"  Log file:      {log_path}")

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    try:
        images_dir = validate_inputs(dataset_dir, output_dir, colmap_bat, logger)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    database_path    = output_dir / "database.db"
    sparse_dir       = output_dir / "sparse"
    sparse_model_dir = sparse_dir / "0"
    dense_dir        = output_dir / "dense"

    # ------------------------------------------------------------------
    # Handle existing output (overwrite guard)
    # ------------------------------------------------------------------
    if database_path.exists():
        if args.overwrite:
            logger.warning(
                f"Existing database found: {database_path}\n"
                "         --overwrite set — deleting database and sparse/ to start fresh."
            )
            database_path.unlink()
            if sparse_dir.exists():
                shutil.rmtree(sparse_dir)
                logger.warning(f"Deleted existing sparse directory: {sparse_dir}")
        else:
            logger.error(
                f"Output directory already contains a database:\n  {database_path}\n"
                "Use --overwrite to delete and rerun, or choose a different --output."
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Run pipeline stages
    # ------------------------------------------------------------------
    try:
        # --- Sparse pipeline (always runs) ---
        stage_feature_extraction(colmap_bat, database_path, images_dir, logger)
        stage_feature_matching(colmap_bat, database_path, logger)
        stage_sparse_reconstruction(
            colmap_bat, database_path, images_dir, sparse_dir, logger
        )

        # Verify mapper produced at least one reconstruction component
        if not sparse_model_dir.exists():
            raise RuntimeError(
                f"Mapper finished but no reconstruction was produced at:\n"
                f"  {sparse_model_dir}\n"
                "This usually means too few image pairs matched successfully. "
                "Check the log for mapper warnings or error messages."
            )

        stage_convert_model(colmap_bat, sparse_model_dir, logger)

        # --- Dense pipeline (optional) ---
        if args.dense:
            logger.info("Dense reconstruction requested — running stages 5–7.")
            stage_image_undistortion(
                colmap_bat, images_dir, sparse_model_dir, dense_dir, logger
            )
            stage_patch_match_stereo(colmap_bat, dense_dir, logger)
            stage_stereo_fusion(colmap_bat, dense_dir, logger)

        # --- Copy to results (optional) ---
        if args.copy_to_results:
            project_root = Path(__file__).resolve().parent.parent.parent
            results_root = project_root / "results"
            copy_sparse_to_results(
                sparse_dir=sparse_dir,
                dataset_name=dataset_dir.name,
                results_root=results_root,
                logger=logger,
            )

    except RuntimeError as e:
        logger.error(str(e))
        logger.error("Pipeline aborted. Resolve the error above and rerun.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C).")
        sys.exit(130)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("COLMAP BASELINE PIPELINE — COMPLETE")
    logger.info(f"  Sparse model:  {sparse_model_dir}")
    logger.info(f"  Log file:      {log_path}")
    if args.dense:
        logger.info(f"  Dense cloud:   {dense_dir / 'fused.ply'}")
    if args.copy_to_results:
        project_root = Path(__file__).resolve().parent.parent.parent
        logger.info(
            f"  Results copy:  "
            f"{project_root / 'results' / 'original_colmap' / dataset_dir.name}"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
