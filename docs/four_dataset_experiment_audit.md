# Comprehensive Four-Dataset Experiment Audit Report

**Project:** Quality-Guided Adaptive Image Enhancement for 3D Reconstruction  
**Project Root:** `E:\CV Project`  
**Audit Date:** October 26, 2026  
**Auditor:** Senior Computer Vision Research & COLMAP Engineering Team  
**Scope:** Verification of all four project datasets (`gerrard-hall`, `south-building`, `person-hall`, `graham-hall`) across baseline, enhancement, and comparative evaluation stages.

---

## 1. Executive Summary

This audit rigorously inspects the filesystem, COLMAP databases, sparse reconstruction models, log files, and analytical summaries across all four datasets in the project. Every reported metric has been verified directly against binary and text model files (`cameras.txt`, `images.txt`, `points3D.txt`), and every database has been validated via SQLite `PRAGMA integrity_check`.

### High-Level Status:
- **Gerrard Hall (100 images):** Fully completed and verified (Baseline + Phase 5 Adaptive Enhancement).
- **South Building (128 images):** Fully completed and verified (Baseline + Phase 5 Adaptive + Phase 6 Ablation Variants C & E).
- **Person Hall (330 images):** Fully completed and verified (Baseline + Phase B Conservative Variant E).
- **Graham Hall (1,273 images per part, 6 parts):** **Not started in experiments.** No baseline, enhancement, or evaluation runs exist in `experiments/`. The dataset directory contains only raw images and 2016 reference benchmark files from the original UNC Chapel Hill distribution.

---

## 2. Table 1 — Four-Dataset Status Matrix

| Dataset | Images (Input) | Baseline Status | Enhancement Status | Comparison Status | Key Artifacts & Verification Evidence |
|---|---:|---|---|---|---|
| **`gerrard-hall`** | 100 | **Complete** | **Complete** (Phase 5 Adaptive: 23 enhanced, 77 unchanged) | **Complete & Verified** | • Baseline Model: `experiments/baseline/gerrard-hall/sparse/0/`<br>• Enhanced Model: `experiments/baseline_enhanced/gerrard-hall/sparse/0/`<br>• Log: `experiments/baseline/gerrard-hall/baseline.log`<br>• Report: `experiments/evaluation/report.md` |
| **`south-building`** | 128 | **Complete** | **Complete** (Phase 6 Variant E: 2 enhanced, 126 unchanged; Phase 5: 28 enhanced) | **Complete & Verified** | • Baseline Model: `experiments/baseline/south-building/sparse/0/`<br>• Variant E Model: `experiments/phase6/variant_e_conservative_selection/colmap_output/sparse/0/`<br>• Phase 6 Report: `experiments/phase6/phase6_report.md`<br>• Summary: `experiments/phase6/summary.json` |
| **`person-hall`** | 330 (Part1: 165, Part2: 165) | **Complete** | **Complete** (Phase B Variant E: 5 enhanced, 325 unchanged) | **Complete & Verified** | • Baseline Model: `experiments/person_hall/baseline/sparse/0/`<br>• Variant E Model: `experiments/person_hall/variant_e/sparse/0/`<br>• Baseline Summary: `experiments/person_hall/baseline/baseline_summary.json`<br>• Variant E Summary: `experiments/person_hall/variant_e/variant_e_summary.json`<br>• Comparison: `experiments/person_hall/variant_e/comparison_with_baseline.json`<br>• Multi-Dataset Report: `experiments/person_hall/multi_dataset_validation_report.md` |
| **`graham-hall`** | 1,273 per Part (6 Parts = 7,638 files) | **Not Started** | **Not Started** | **Missing / Incomplete** | • No experiment folders in `experiments/baseline/` or `experiments/`<br>• Original 2016 UNC benchmark ground-truth files in `datasets/graham-hall/Part*/`<br>• Flagged as "Planned / Pending" in `experiments/evaluation/report.md` |

---

## 3. Table 2 — Verified Empirical Metrics Matrix

All numbers in this table are extracted directly from verified sparse model text files (`points3D.txt`, `images.txt`) and SQLite queries on `two_view_geometries`:

| Dataset & Run | Reg. Images | Reg. Rate | 3D Points | Mean Reproj Error (px) | Median Reproj Error (px) | Mean Obs / Img | Mean Track Length | Verified 2-View Inliers | Verified Pairs | Total SfM Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gerrard Hall Baseline** | $100 / 100$ | $100.0\%$ | 58,127 | 1.0769 | 1.0441 | 3,280.11 | 5.6430 | 969,591 | 1,299 | 1,594.5 s (~26.6 m) |
| **Gerrard Hall Enhanced (P5)** | $100 / 100$ | $100.0\%$ | 58,312 | 1.0786 | 1.0429 | 3,278.64 | 5.6226 | 963,777 | 1,288 | 1,241.1 s (~20.7 m) |
| *Gerrard Hall Delta* | *0* | *0.0%* | *+185 (+0.32%)* | *+0.0017 (+0.16%)* | *-0.0012* | *-1.47 (-0.04%)* | *-0.0204 (-0.36%)* | *-5,814 (-0.60%)* | *-11* | *-353.4 s* |
|---|---|---|---|---|---|---|---|---|---|---|
| **South Building Baseline** | $128 / 128$ | $100.0\%$ | 84,557 | 0.6116 | 0.4957 | 3,924.48 | 5.9408 | 1,500,850 | 2,541 | 1,979.1 s (~33.0 m) |
| **South Building P5 (clip=2.0, 28 img)** | $128 / 128$ | $100.0\%$ | 81,018 | 0.6255 | 0.5071 | 3,772.46 | 5.9601 | 1,446,120 | 2,492 | 1,987.7 s (~33.1 m) |
| **South Building P6 Var E (clip=1.2, 2 img)** | $128 / 128$ | $100.0\%$ | 84,593 | 0.6120 | 0.4946 | 3,925.08 | 5.9391 | 1,500,098 | 2,525 | 1,863.1 s (~31.1 m) |
| *South Building Delta (Var E vs Base)* | *0* | *0.0%* | *+36 (+0.043%)* | *+0.0004 (+0.07%)* | *-0.0011* | *+0.60 (+0.015%)* | *-0.0017 (-0.03%)* | *-752 (-0.05%)* | *-16* | *-116.0 s* |
|---|---|---|---|---|---|---|---|---|---|---|
| **Person Hall Baseline** | $330 / 330$ | $100.0\%$ | 206,994 | 1.2550 | 1.2608 | 4,308.24 | 6.8684 | 7,468,981 | 7,347 | 16,983.8 s (~4.72 h) |
| **Person Hall Var E (clip=1.2, 5 img)** | $330 / 330$ | $100.0\%$ | 207,851 | 1.2569 | 1.2627 | 4,313.19 | 6.8479 | 7,464,515 | 7,343 | 17,577.4 s (~4.88 h) |
| *Person Hall Delta (Var E vs Base)* | *0* | *0.0%* | *+857 (+0.414%)* | *+0.0019 (+0.151%)* | *+0.0019 (+0.151%)* | *+4.95 (+0.115%)* | *-0.0205 (-0.298%)* | *-4,466 (-0.060%)* | *-4* | *+593.6 s* |

---

## 4. Table 3 — Missing Work, Limitations & Recommended Next Steps

| Dataset / Area | Missing Work / Inconsistencies | Severity | Recommended Next Steps |
|---|---|---|---|
| **`graham-hall`** | • No baseline reconstruction exists.<br>• No enhancement configuration exists.<br>• No comparative evaluation exists.<br>• 1,273 images per part across 6 parts. | **Major Unfinished Component** | • **Do NOT run exhaustive matching naively on all 1,273 images** (810,276 pairs = ~50–60 CPU hours).<br>• If evaluated, select a manageable partition (e.g., `Part2` exterior only: 562 images = 157,641 pairs, or sequential/vocab-tree matching). |
| **Consolidated Multi-Dataset Synthesis** | • `experiments/evaluation/report.md` was created during Phase 5 (2026-10-10) and does not reflect South Building Phase 6 or Person Hall completions. | **Minor Documentation Gap** | • The project has `experiments/person_hall/multi_dataset_validation_report.md` covering South Building + Person Hall.<br>• Update the central `experiments/evaluation/report.md` to incorporate Gerrard Hall, South Building Phase 6 Variant E, and Person Hall Variant E into a unified 3-dataset table. |
| **Results Clean Copy Directory** | • `results/original_colmap/` and `results/enhanced_colmap/` currently only host `gerrard-hall` and `south-building`. | **Minor Organizational Gap** | • Copy the final Person Hall baseline and Variant E models into `results/original_colmap/person-hall/` and `results/enhanced_colmap/person-hall/` for uniform archival. |

---

## 5. Detailed Dataset Audits

### 5.1 Gerrard Hall
- **Dataset Path:** `datasets/gerrard-hall/images/` (100 images).
- **Baseline:** Complete in `experiments/baseline/gerrard-hall/`. 
  - Sparse model: `sparse/0/` with all 11 binary and TXT files.
  - Database: `database.db` (179.97 MB, `PRAGMA integrity_check` = `ok`).
  - Metrics: 100/100 images registered, 58,127 points, 1.0769 px mean error.
- **Enhanced Configuration:** Complete in `experiments/enhanced/gerrard-hall/` and reconstructed in `experiments/baseline_enhanced/gerrard-hall/`.
  - Enhancement: Phase 3 adaptive rule (Laplacian variance & contrast std dev). 23 images enhanced (5 CLAHE, 17 sharpen, 1 both), 77 unchanged.
  - Sparse model: `sparse/0/` with all 11 binary and TXT files.
  - Database: `database.db` (181.72 MB, `PRAGMA integrity_check` = `ok`).
  - Metrics: 100/100 images registered, 58,312 points (+185), 1.0786 px mean error (+0.0017 px).
- **Evaluation Status:** Fully documented in `experiments/evaluation/report.md`, `summary.json`, and `summary.csv`.

### 5.2 South Building
- **Dataset Path:** `datasets/south-building/images/` (128 images).
- **Baseline:** Complete in `experiments/baseline/south-building/`.
  - Sparse model: `sparse/0/` with all 11 binary and TXT files.
  - Database: `database.db` (245.82 MB, `PRAGMA integrity_check` = `ok`).
  - Metrics: 128/128 images registered, 84,557 points, 0.6116 px mean error.
- **Enhanced Configurations:**
  - *Phase 5 Adaptive (clip 2.0, 28 images enhanced):* Demonstrated catastrophic point collapse (81,018 points, -3,539 points, +2.27% error).
  - *Phase 6 Variant E (clip 1.2, 2 images enhanced: `P1180166.JPG`, `P1180179.JPG`, 126 unchanged):* Surpassed baseline with 84,593 points (+36 points), 0.6120 px error (+0.0004 px).
  - Sparse model: `experiments/phase6/variant_e_conservative_selection/colmap_output/sparse/0/` (all 11 binary and TXT files).
  - Database: `experiments/phase6/variant_e_conservative_selection/colmap_output/database.db` (246.08 MB, `PRAGMA integrity_check` = `ok`).
- **Evaluation Status:** Thoroughly documented in `experiments/phase6/phase6_report.md` and `experiments/phase6/summary.json`.

### 5.3 Person Hall
- **Dataset Path:** `datasets/person-hall/` (Part1: 165 images, Part2: 165 images; 330 total).
- **Baseline:** Complete in `experiments/person_hall/baseline/`.
  - Reconstructed from consolidated 330-image pool with single unified `SIMPLE_RADIAL` camera model.
  - Sparse model: `sparse/0/` with all 11 binary and TXT files.
  - Database: `database.db` (718.85 MB, `PRAGMA integrity_check` = `ok`).
  - Metrics: 330/330 registered, 206,994 points, 1.2550 px mean error, 7,468,981 two-view inliers.
- **Enhanced Configuration (Variant E):** Complete in `experiments/person_hall/variant_e/`.
  - Dual-criteria gating (Contrast $< 52.0$ & SIFT $< 10,000$) selected exactly 5 images (`IMG_1179.JPG`, `IMG_1282.JPG`, `IMG_1312.JPG`, `IMG_1316.JPG`, `IMG_1323.JPG`).
  - 325 images are verified 100% byte-identical.
  - Sparse model: `sparse/0/` with all 11 binary and TXT files.
  - Database: `database.db` (719.33 MB, `PRAGMA integrity_check` = `ok`).
  - Metrics: 330/330 registered, 207,851 points (+857 points, +0.414%), 1.2569 px mean error (+0.0019 px), 7,464,515 two-view inliers.
- **Evaluation Status:** Fully documented in `experiments/person_hall/variant_e/comparison_with_baseline.json` and `experiments/person_hall/multi_dataset_validation_report.md`.

### 5.4 Graham Hall
- **Dataset Path:** `datasets/graham-hall/` (Part1 through Part6).
  - Each Part contains: `images/exterior/` (562 images) and `images/interior/` (711 images) = 1,273 images per part.
  - Total image files across 6 parts: 7,638.
- **Audit Findings:**
  - Files inside `datasets/graham-hall/Part*/database.db` and `datasets/graham-hall/Part*/sparse/` carry timestamps from October 5, 2016. These are the original benchmark publication assets from the Schönberger et al. dataset release.
  - In `experiments/`, **there are zero directories, zero databases, zero logs, and zero models for Graham Hall**.
  - In `docs/project_recovery_and_phase_history.md` and `experiments/evaluation/report.md`, Graham Hall is explicitly designated as "Pending / Not Run" due to compute duration and memory constraints.
- **Scalability & Runtime Considerations:**
  - Exhaustive matching on 1,273 images requires $810,276$ pair verifications ($14.9\times$ more than Person Hall).
  - Given Person Hall's exhaustive matching runtime of $14,485\text{ s}$ (~$4.02\text{ hours}$ on CPU for $54,285$ pairs), a full exhaustive match of Graham Hall would require approximately **55 to 65 CPU hours** (~2.5 days of non-stop computation), plus another **8 to 12 hours** for mapper.

---

## 6. Answers to Explicit Audit Questions

1. **Have all four datasets been evaluated?**  
   **No.** Only three of the four datasets (`gerrard-hall`, `south-building`, and `person-hall`) have been evaluated. Graham Hall has not been run.

2. **On how many datasets was a baseline completed?**  
   **Three (3) datasets:** `gerrard-hall` (100 images), `south-building` (128 images), and `person-hall` (330 images).

3. **On how many datasets was enhancement tested?**  
   **Three (3) datasets:**  
   - `gerrard-hall` (Phase 5 Adaptive: 23 enhanced)
   - `south-building` (Phase 5 Adaptive: 28 enhanced; Phase 6 Variant C: 28 enhanced; Phase 6 Variant E: 2 enhanced)
   - `person-hall` (Phase B Variant E: 5 enhanced)

4. **On how many datasets is the comparison supported by verified artifacts?**  
   **Three (3) datasets.** All three datasets have complete, verified binary and TXT sparse models (`sparse/0/points3D.txt`, `images.txt`, `cameras.txt`), clean SQLite databases passing integrity checks, detailed JSON/CSV summaries, and cross-dataset comparative reports.

5. **Is Graham Hall unfinished, partially completed, or already completed?**  
   **Graham Hall is completely unstarted / unfinished in the experimental pipeline.** No experiment directories exist for it under `experiments/`. The only files present are the raw images and the 2016 reference assets in `datasets/graham-hall/`.

6. **What is the smallest, safest next step to improve the project?**  
   The smallest, safest next step is **purely analytical and organizational**:
   - Synthesize the verified results of the three completed datasets (`gerrard-hall`, `south-building`, and `person-hall`) into a single consolidated Phase 7 master report in `experiments/evaluation/`.
   - Copy the Person Hall baseline and Variant E sparse models to `results/original_colmap/person-hall/` and `results/enhanced_colmap/person-hall/` to complete the publication archive.
   - If Graham Hall is ever tackled in the future, evaluate a single constrained subset (e.g., `Part2` exterior only: 562 images) or use vocab-tree matching rather than launching an unsustainable 60-hour full exhaustive SfM run.
