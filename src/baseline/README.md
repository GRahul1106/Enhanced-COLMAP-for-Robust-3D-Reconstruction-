# COLMAP Baseline Pipeline

Runs the standard COLMAP reconstruction pipeline on a dataset without
modifying any original files. All outputs are written to a directory
you specify.

---

## Prerequisites

**Python:** 3.8 or later. No third-party packages are required — the script
uses only the Python standard library.

**COLMAP:** The pre-built Windows binary is already installed at
`colmap/COLMAP.bat` in the project root. The script resolves this path
automatically. Override it with `--colmap_bat` if you move the installation.

**GPU (dense reconstruction only):** `patch_match_stereo` requires an NVIDIA
GPU with CUDA support. Omit `--dense` if you only need a sparse reconstruction.

---

## Quick start

Run from the **project root** (`e:\CV Project`):

```powershell
python src/baseline/run_colmap_baseline.py `
    --dataset "datasets\gerrard-hall" `
    --output  "experiments\baseline\gerrard-hall"
```

Or using full absolute paths (works from any directory):

```powershell
python "e:\CV Project\src\baseline\run_colmap_baseline.py" `
    --dataset  "e:\CV Project\datasets\gerrard-hall" `
    --output   "e:\CV Project\experiments\baseline\gerrard-hall"
```

---

## All command-line options

| Argument | Required | Default | Description |
|---|---|---|---|
| `--dataset` | Yes | — | Path to the dataset directory (must contain `images/`) |
| `--output` | Yes | — | Directory for all outputs. Created automatically if missing. Must NOT be inside `--dataset`. |
| `--colmap_bat` | No | `<project_root>/colmap/COLMAP.bat` | Path to `COLMAP.bat` |
| `--dense` | No | off | Also run MVS dense reconstruction (requires NVIDIA GPU + CUDA) |
| `--overwrite` | No | off | Delete existing output and restart from scratch |
| `--copy_to_results` | No | off | Copy sparse model to `results/original_colmap/<dataset-name>/` after reconstruction |

---

## Examples

**Gerrard Hall — sparse reconstruction only:**
```powershell
python src/baseline/run_colmap_baseline.py `
    --dataset "datasets\gerrard-hall" `
    --output  "experiments\baseline\gerrard-hall"
```

**South Building — sparse + dense reconstruction:**
```powershell
python src/baseline/run_colmap_baseline.py `
    --dataset "datasets\south-building" `
    --output  "experiments\baseline\south-building" `
    --dense
```

**Graham Hall Part 2 — sparse + copy to results for later comparison:**
```powershell
python src/baseline/run_colmap_baseline.py `
    --dataset         "datasets\graham-hall\Part2" `
    --output          "experiments\baseline\graham-hall-part2" `
    --copy_to_results
```

**Person Hall Part 1 — re-run from scratch (deletes previous output):**
```powershell
python src/baseline/run_colmap_baseline.py `
    --dataset   "datasets\person-hall\Part1" `
    --output    "experiments\baseline\person-hall-part1" `
    --overwrite
```

**Custom COLMAP location:**
```powershell
python src/baseline/run_colmap_baseline.py `
    --dataset    "datasets\gerrard-hall" `
    --output     "experiments\baseline\gerrard-hall" `
    --colmap_bat "C:\tools\colmap\COLMAP.bat"
```

---

## Pipeline stages

| # | COLMAP command | Reads | Writes | Purpose |
|---|---|---|---|---|
| 1 | `feature_extractor` | `images/` *(read-only)* | `database.db` | Detects SIFT keypoints + descriptors for every image |
| 2 | `exhaustive_matcher` | `database.db` | `database.db` | Matches features between all image pairs |
| 3 | `mapper` | `database.db`, `images/` *(read-only)* | `sparse/0/*.bin` | Incremental SfM → camera poses + 3D point cloud |
| 4 | `model_converter` | `sparse/0/*.bin` | `sparse/0/*.txt` | Converts binary model to human-readable text format |
| 5* | `image_undistorter` | `images/` *(read-only)*, `sparse/0/` | `dense/` | Undistorts images using calibrated camera model |
| 6* | `patch_match_stereo` | `dense/` | `dense/stereo/` | Computes per-image depth + normal maps (GPU required) |
| 7* | `stereo_fusion` | `dense/stereo/` | `dense/fused.ply` | Fuses depth maps into a single dense point cloud |

\* Only executed when `--dense` is passed.

**The original dataset is never written to.** Stages 1, 3, 5 pass the dataset
images path to COLMAP as a read-only input. Every output file — `database.db`,
`sparse/`, `dense/` — is written under `--output`.

---

## Output structure

```
<output>/
├── baseline.log                  ← timestamped log of every stage
├── database.db                   ← COLMAP SQLite database (features + matches)
├── sparse/
│   └── 0/                        ← first reconstruction component
│       ├── cameras.bin           ← camera intrinsics  (binary — from mapper)
│       ├── images.bin            ← camera poses       (binary — from mapper)
│       ├── points3D.bin          ← 3D points          (binary — from mapper)
│       ├── cameras.txt           ← camera intrinsics  (text   — from model_converter)
│       ├── images.txt            ← camera poses       (text   — from model_converter)
│       └── points3D.txt          ← 3D points          (text   — from model_converter)
└── dense/                        ← only present when --dense is used
    ├── images/                   ← undistorted images
    ├── sparse/                   ← undistorted camera model
    ├── stereo/
    │   ├── depth_maps/           ← per-image depth maps
    │   └── normal_maps/          ← per-image normal maps
    └── fused.ply                 ← final dense point cloud
```

When `--copy_to_results` is used, the `.txt` and `.bin` files from `sparse/0/`
are also copied to:

```
results/original_colmap/<dataset-name>/
```

---

## Exact commands issued (gerrard-hall example)

```
cmd /c "colmap\COLMAP.bat" feature_extractor
    --database_path "experiments\baseline\gerrard-hall\database.db"
    --image_path    "datasets\gerrard-hall\images"
    --ImageReader.single_camera 1

cmd /c "colmap\COLMAP.bat" exhaustive_matcher
    --database_path "experiments\baseline\gerrard-hall\database.db"

cmd /c "colmap\COLMAP.bat" mapper
    --database_path "experiments\baseline\gerrard-hall\database.db"
    --image_path    "datasets\gerrard-hall\images"
    --output_path   "experiments\baseline\gerrard-hall\sparse"

cmd /c "colmap\COLMAP.bat" model_converter
    --input_path  "experiments\baseline\gerrard-hall\sparse\0"
    --output_path "experiments\baseline\gerrard-hall\sparse\0"
    --output_type TXT

# Dense only:
cmd /c "colmap\COLMAP.bat" image_undistorter
    --image_path  "datasets\gerrard-hall\images"
    --input_path  "experiments\baseline\gerrard-hall\sparse\0"
    --output_path "experiments\baseline\gerrard-hall\dense"
    --output_type COLMAP

cmd /c "colmap\COLMAP.bat" patch_match_stereo
    --workspace_path   "experiments\baseline\gerrard-hall\dense"
    --workspace_format COLMAP
    --PatchMatchStereo.geom_consistency true

cmd /c "colmap\COLMAP.bat" stereo_fusion
    --workspace_path   "experiments\baseline\gerrard-hall\dense"
    --workspace_format COLMAP
    --input_type       geometric
    --output_path      "experiments\baseline\gerrard-hall\dense\fused.ply"
```

---

## Log file

Every run writes a full log to `<output>/baseline.log`. The console shows
INFO-level messages. The log file captures everything including full COLMAP
stdout and stderr for every stage — useful for debugging.

If a stage fails, the failed command, exit code, and full COLMAP output are
printed to the console **and** written to the log file before the script exits.

---

## Troubleshooting

**`FileNotFoundError: COLMAP.bat not found`**
The script could not find `colmap\COLMAP.bat` relative to the project root.
Use `--colmap_bat` to point to the correct location explicitly.

**`Mapper finished but no reconstruction was produced at sparse/0`**
Too few image pairs matched successfully for the mapper to register any images.
Open `baseline.log` and search for mapper warnings. Common causes:
- Images are too blurry or dark for SIFT to find enough features.
- The dataset has very little visual overlap between consecutive images.

**`patch_match_stereo` fails immediately**
No compatible NVIDIA GPU was detected. Run without `--dense`.

**`Output directory already contains a database`**
A previous run exists. Either use `--overwrite` to delete it and restart,
or point `--output` to a new directory.

**Graham Hall Part 1** — the existing `cameras.txt` in the original dataset
is empty (0 bytes), meaning the pre-existing sparse reconstruction for Part 1
is incomplete. The baseline script will run a fresh reconstruction from images
regardless of what is in the dataset's `sparse/` folder.

---

## Dataset safety guarantee

The script enforces two protections:

1. **Path check at startup:** if `--output` is detected to be inside `--dataset`,
   the script exits immediately with an error before running any COLMAP stage.

2. **Read-only image access:** COLMAP receives the dataset images path only via
   `--image_path`, which COLMAP treats as read-only. No COLMAP stage writes
   to the images directory.
