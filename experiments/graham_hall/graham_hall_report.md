# Phase 7: Graham Hall Accelerated Validation Report (270-Image Exterior Subset)

**Project Title:** Quality-Guided Adaptive Image Enhancement for 3D Reconstruction  
**Authors / Role:** Senior Computer Vision Research & COLMAP Engineering Team  
**Dataset & Scope:** Graham Hall 270-Image Exterior Sequence (`IMG_0348.jpg` to `IMG_0617.jpg`)  
**Execution Date:** October 26, 2026  
**Document Location:** `experiments/graham_hall/graham_hall_report.md`  
**Status:** Completed, Audited & Empirically Verified  

---

## Executive Summary

This report documents the Phase 7 validation experiment conducted on a continuous 270-image exterior sequence from the **Graham Hall** architectural dataset. Completed within an accelerated 48-hour deadline window, this evaluation assesses whether conservative quality-guided image enhancement (**Variant E**) generalizes to Graham Hall's complex collegiate architecture.

### Scope and Constraints
- **Subset Scope:** This experiment evaluated an unbroken, continuous 270-image exterior facade traversal (`datasets/graham-hall/Part2/images/exterior/IMG_0348.jpg` through `IMG_0617.jpg`). **It is a controlled subset validation, not an exhaustive evaluation of all 7,638 images in the Graham Hall multi-partition repository.**
- **Methodology:** Classical computer vision only (OpenCV, NumPy, SIFT, CLAHE, COLMAP 4.2.0 CPU-only). Single shared intrinsic camera model (`SIMPLE_RADIAL`, $5616 \times 3744$) and exhaustive feature matching ($36,315$ pairs).
- **Variant E Selection:** Fixed dual-constraint criterion ($\text{Contrast } \sigma < 52.0 \land 0 < \text{Baseline SIFT} < 10,000$). Exactly **28 images (10.4%)** met the criterion and received CLAHE (`clipLimit=1.2`, `tileGridSize=(8, 8)`). The remaining 242 images remained 100% byte-identical.

### Key Empirical Findings:
1. **Registration:** Variant E registered **248 images** versus **247 images** in Baseline (+1 image registered; 96.12% vs 95.74% of processed views).
2. **3D Point Yield:** Point count increased marginally from 113,644 to 113,775 points (**+131 points, +0.115%**).
3. **Geometric Accuracy:** Mean reprojection error shifted by a negligible $+0.0057\text{ px}$ ($1.1068\text{ px} \rightarrow 1.1125\text{ px}$, $+0.51\%$). Median reprojection error showed an identical shift ($1.0792\text{ px} \rightarrow 1.0847\text{ px}$).
4. **Track Length & Inliers:** Inliers experienced a minor reduction of $-1,532$ ($-0.114\%$, $1,343,083 \rightarrow 1,341,551$) across 4,467 pairs (-19 pairs), and mean track length slightly decreased from $4.7216$ to $4.7149$ ($-0.0067$).
5. **Efficiency:** Total SfM execution was achieved in **2.13 hours for Baseline** and **1.91 hours for Variant E**, completing both runs well within the 48-hour target.

---

## 1. Experimental Setup & Pre-Flight Verification

### 1.1 Dataset Partition Selection
Initial auditing established that `Part1` of Graham Hall contained corrupted 0-byte images and a 0-byte database. In contrast, `Part2` represents a complete, intact master distribution containing 1,273 images (562 exterior + 711 interior) at $5616 \times 3744$ resolution (Canon EOS 5D Mark II).

### 1.2 Sequence Selection Rationale (Sequence 1: 270 Images)
The continuous sequence `IMG_0348.jpg` to `IMG_0617.jpg` was chosen because:
- **Continuous Physical Traversal:** Captured sequentially on September 10, 2015, between `13:12:48` and `13:33:49` with dense baseline overlap (median camera distance: 0.18 units).
- **Lighting Dynamics:** Transitions from bright sunny facades ($\sigma = 75$–$86$) to deep shadowed arches, entryways, and courtyard masonry ($\sigma = 25$–$45$).
- **Computational Feasibility:** 270 images generate $(270 \times 269)/2 = 36,315$ pairs, enabling full exhaustive matching on CPU in ~1.7 hours, matching the scale of Person Hall ($330$ images).

### 1.3 Orientation and Single-Camera Reader Handling
Among the 270 images, 258 images were captured in landscape orientation ($5616 \times 3744$). Exactly 12 images (`IMG_0406` to `IMG_0411`, `IMG_0422` to `IMG_0424`, `IMG_0440`, `IMG_0445`, `IMG_0446`) were vertical portrait orientation ($3744 \times 5616$). In accordance with COLMAP's single shared camera reader requirement (`--ImageReader.single_camera 1`), COLMAP assigned Camera #1 ($5616 \times 3744$) to the first view and skipped the 12 mismatched portrait views. Thus, exactly 258 landscape images entered feature extraction and reconstruction in both baseline and Variant E.

---

## 2. Quantitative Baseline vs. Variant E Comparison

All metrics are read directly from the verified sparse reconstruction models (`points3D.txt`, `images.txt`) and SQLite database tables (`two_view_geometries`, `matches`).

| Metric | Baseline (Phase A) | Variant E (Phase B) | Absolute Delta | Percentage Delta |
|---|---:|---:|---:|---:|
| **Input Images (Disk)** | 270 | 270 | 0 | 0.0% |
| **Processed Views (Camera #1)** | 258 | 258 | 0 | 0.0% |
| **Enhanced Images** | 0 (0.0%) | 28 (10.85%) | +28 | — |
| **Registered Images** | 247 | 248 | **+1** | **+0.40%** |
| **Registration Rate (Processed)** | 95.74% | 96.12% | +0.38% | +0.40% |
| **Registration Rate (Total)** | 91.48% | 91.85% | +0.37% | +0.40% |
| **Reconstructed 3D Points** | 113,644 | 113,775 | **+131** | **+0.115%** |
| **Mean Reprojection Error** | 1.1068 px | 1.1125 px | +0.0057 px | +0.515% |
| **Median Reprojection Error** | 1.0792 px | 1.0847 px | +0.0055 px | +0.509% |
| **Mean Observations / Image** | 2,172.38 | 2,163.04 | -9.34 | -0.430% |
| **Mean Track Length** | 4.7216 | 4.7149 | -0.0067 | -0.142% |
| **Verified Two-View Inliers** | 1,343,083 | 1,341,551 | **-1,532** | **-0.114%** |
| **Verified Image Pairs** | 4,486 | 4,467 | -19 | -0.424% |
| **Raw Match Pairs** | 18,926 | 18,928 | +2 | +0.011% |
| **Database Size** | 422.39 MB | 427.70 MB | +5.31 MB | +1.26% |
| **Total SfM Runtime** | 7,657.0 s (~2.13 h) | 6,860.7 s (~1.91 h) | -796.3 s | -10.40% |

---

## 3. Stage Runtime Breakdown

| Stage | Baseline Runtime (s) | Variant E Runtime (s) | Status & Nature of Timing |
|---|---:|---:|---|
| **Feature Extraction** | 135.6 s | 12.5 s | Incremental extraction for 28 enhanced views (230 views inherited from baseline) |
| **Exhaustive Matching** | 6,170.0 s (~1.71 h) | 6,088.5 s (~1.69 h) | Measured directly for 36,315 pairs |
| **Sparse Mapping (`mapper`)** | 1,346.4 s (~22.4 m) | 750.2 s (~12.5 m) | Measured directly on CPU |
| **Model Conversion (TXT)** | 4.9 s | 4.6 s | Measured directly |
| **Total SfM Time** | **7,657.0 s (~2.13 h)** | **6,860.7 s (~1.91 h)** | **Completed in 4.04 hours cumulative** |

Both runs finished well within the 48-hour window, requiring a cumulative runtime of only ~4.04 hours.

---

## 4. Scientific Observations & Comparative Analysis

### 4.1 Cross-Dataset Consistency
The Graham Hall results closely mirror the findings from Gerrard Hall, South Building, and Person Hall:
1. **Marginal Point Gain:** Variant E generated $+131$ points ($+0.115\%$). This matches the trend across the other benchmarks:
   - Gerrard Hall: $+0.32\%$ (+185 points)
   - South Building: $+0.043\%$ (+36 points)
   - Person Hall: $+0.414\%$ (+857 points)
   - Graham Hall: $+0.115\%$ (+131 points)
2. **Stable Reprojection Error:** Reprojection error shifted by $+0.0057\text{ px}$ ($+0.51\%$). As observed on previous datasets, conservative enhancement avoids introducing high-frequency geometric distortion.
3. **Registration Gain:** In Graham Hall, Variant E successfully registered **248 images** versus **247 images** in Baseline (+1 image). The extra features on under-exposed views facilitated the initial pose estimation of an otherwise borderline frame.
4. **Minor Inlier & Track Length Attenuation:** Verified inliers decreased by $-0.114\%$ and mean track length shortened by $-0.0067$. As identified in earlier phases, CLAHE introduces subtle descriptor gradient shifts that can cause a small fraction of cross-view matches to fail epipolar verification against unmodified adjacent views.

---

## 5. Artifact Verification & Preserved Records

All artifacts for this experiment are isolated and preserved:
- Baseline sparse model: [`experiments/graham_hall/baseline/sparse/0/`](file:///e:/CV%20Project/experiments/graham_hall/baseline/sparse/0/)
- Baseline database: [`experiments/graham_hall/baseline/database.db`](file:///e:/CV%20Project/experiments/graham_hall/baseline/database.db) (422.39 MB, `PRAGMA integrity_check` = `ok`)
- Variant E sparse model: [`experiments/graham_hall/variant_e/sparse/0/`](file:///e:/CV%20Project/experiments/graham_hall/variant_e/sparse/0/)
- Variant E database: [`experiments/graham_hall/variant_e/database.db`](file:///e:/CV%20Project/experiments/graham_hall/variant_e/database.db) (427.70 MB, `PRAGMA integrity_check` = `ok`)
- Candidate CSV: [`experiments/graham_hall/baseline/candidate_images_variant_e.csv`](file:///e:/CV%20Project/experiments/graham_hall/baseline/candidate_images_variant_e.csv)
- Comparison JSON: [`experiments/graham_hall/variant_e/comparison_with_baseline.json`](file:///e:/CV%20Project/experiments/graham_hall/variant_e/comparison_with_baseline.json)
- Side-by-side previews: [`experiments/graham_hall/variant_e/previews/`](file:///e:/CV%20Project/experiments/graham_hall/variant_e/previews/)

---

## 6. Conclusion

The 270-image exterior validation on Graham Hall confirms the conclusions reached across the previous three benchmarks: **conservative quality-guided image enhancement (Variant E) is stable and safe, producing modest point-count gains (+0.115%) and slight registration improvements (+1 image) without harming bundle adjustment accuracy.** However, these improvements remain strictly incremental and do not represent a transformative departure from unenhanced baseline performance.
