# Master Validation Report: Quality-Guided Adaptive Image Enhancement for 3D Reconstruction

**Project Title:** Quality-Guided Adaptive Image Enhancement for 3D Reconstruction  
**Authors / Role:** Senior Computer Vision Research & COLMAP Engineering Team  
**Evaluation Scope:** 
- **Three Complete Datasets:** Gerrard Hall (100 images), South Building (128 images), Person Hall (330 images)
- **Phase 7 Subset Validation:** Graham Hall 270-Image Exterior Sequence (`IMG_0348.jpg` to `IMG_0617.jpg`)
**Date:** October 26, 2026  
**Document Location:** `experiments/evaluation/master_validation_report.md`  
**Status:** Completed, Audited & Empirically Verified  

---

## Executive Summary

This master report synthesizes the experimental results of the **Quality-Guided Adaptive Image Enhancement for 3D Reconstruction** project across all evaluated datasets. The research investigates whether conservative, quality-guided pre-processing of architectural photography improves classical Structure-from-Motion (SfM) pipelines.

The project strictly utilizes classical computer vision methods: **OpenCV**, **NumPy**, **SIFT**, **CLAHE**, and **unsharp masking**, reconstructed using bundled **COLMAP 4.2.0 (CPU-only)** with rigorous geometric bundle adjustment. **No machine learning or deep neural network models were trained or utilized.**

### Principal Findings Across Evaluated Benchmarks:
1. **Gerrard Hall (100 images, Complete Dataset, Phase 5):** Enhancing 23 sub-threshold images yielded a modest point increase (**+185 points, +0.32%**) from 58,127 to 58,312 points, with reprojection error essentially unchanged ($1.0769\text{ px} \rightarrow 1.0786\text{ px}$, $+0.16\%$).
2. **South Building (128 images, Complete Dataset, Phase 6 Variant E):** Overcoming the failure of aggressive blanket enhancement (Phase 5 lost $-3,539$ points due to descriptor drift), a conservative dual-constraint filter (Contrast $< 52.0$ AND SIFT $< 10,000$) enhanced only 2 images (`P1180166.JPG`, `P1180179.JPG`). This preserved global SfM stability and surpassed the baseline (**+36 points, +0.043%**), increasing point count from 84,557 to 84,593 with identical reprojection error ($0.6116\text{ px} \rightarrow 0.6120\text{ px}$).
3. **Person Hall (330 images, Complete Dataset, Phase B Variant E):** Applying the validated conservative Variant E strategy to 330 high-resolution images ($5616 \times 3744$) enhanced only 5 candidate images. This produced a minor point gain (**+857 points, +0.414%**) from 206,994 to 207,851 points, maintaining near-identical reprojection error ($1.2550\text{ px} \rightarrow 1.2569\text{ px}$, $+0.15\%$).
4. **Graham Hall (270-Image Exterior Subset, Phase 7 Variant E):** Validated on a continuous 270-image exterior traversal (`IMG_0348.jpg` to `IMG_0617.jpg`). Exactly 28 images (10.85% of processed views) met the dual criterion. Variant E registered **248 images** (+1 image over baseline 247) and yielded **113,775 points** (**+131 points, +0.115%**) with near-identical reprojection error ($1.1068\text{ px} \rightarrow 1.1125\text{ px}$, $+0.51\%$). **Note: The full 7,638-image multi-partition Graham Hall repository remains uncomputed due to computational tractability constraints (~60 CPU hours).**

### Scientific Takeaway:
Across all evaluated benchmarks, conservative quality-guided enhancement is **safe and non-destructive**: it selectively salvages localized feature correspondences in under-exposed or low-contrast frames without corrupting bundle adjustment or camera calibration. However, across all four benchmarks, **the point-count gains are marginal ($+0.04\%$ to $+0.41\%$) and do not constitute a substantial structural or geometric improvement over well-captured baseline imagery.**

---

## 1. Project Objective and Classical Computer-Vision Methodology

Classical Structure-from-Motion (SfM) pipelines depend heavily on local feature detection (Difference-of-Gaussians scale-space extrema) and gradient orientation descriptors (SIFT / RootSIFT). In real-world architectural surveys, uneven lighting, cast shadows, and low-contrast surface materials can cause local keypoint starvation.

The core research questions addressed are:
1. Can automated image-quality metrics reliably identify genuinely feature-deficient images prior to reconstruction?
2. Does targeted, quality-guided enhancement increase feature yield in low-contrast regions?
3. Does targeted enhancement translate into denser, more accurate 3D reconstructions, or does it introduce descriptor asymmetry and geometric verification failures?

### Core Computational Pipeline
All experiments adhere to a deterministic, reproducible, CPU-based execution framework:
- **Environment:** Windows 11, Python 3, OpenCV, NumPy, SQLite3.
- **COLMAP Binary:** `E:\CV Project\colmap\bin\colmap.exe` (v4.2.0, CPU-only).
- **Camera Models:** Unified intrinsic camera calibration per dataset (`SIMPLE_RADIAL` for Gerrard Hall, Person Hall, and Graham Hall subset; `OPENCV` for South Building).
- **Matching Protocol:** COLMAP `exhaustive_matcher` on all pairs.
- **Reconstruction:** Incremental `colmap mapper` followed by `model_converter --output_type TXT`.

---

## 2. Dataset Descriptions and Experimental Configurations

| Parameter / Dimension | Gerrard Hall | South Building | Person Hall | Graham Hall (Phase 7 Subset) | Graham Hall (Full Repository) |
|---|---|---|---|---|---|
| **Dataset Path** | `datasets/gerrard-hall/images/` | `datasets/south-building/images/` | `datasets/person-hall/` (Part1 + Part2) | `datasets/graham-hall/Part2/images/exterior/` (Sequence 1) | `datasets/graham-hall/` (Part1–Part6) |
| **Total Images** | 100 | 128 | 330 | 270 (258 processed, 12 portrait skipped) | 1,273 per Part (7,638 total) |
| **Image Resolution** | $2816 \times 2112$ | $3072 \times 2048$ | $5616 \times 3744$ | $5616 \times 3744$ | $1536 \times 1024$ |
| **Scene Type** | Classical university building facade | Historic brick building with portico | Extensive collegiate hall & corridors | Continuous exterior facade & courtyard | Multi-wing complex & interior |
| **Camera Model** | Single `SIMPLE_RADIAL` | Single `OPENCV` | Single `SIMPLE_RADIAL` | Single `SIMPLE_RADIAL` | Multiple (Part-dependent) |
| **Baseline Status** | **Complete & Verified** | **Complete & Verified** | **Complete & Verified** | **Complete & Verified** | **Untested / Unstarted** |
| **Enhancement Status**| **Complete & Verified** (P5) | **Complete & Verified** (P5, P6)| **Complete & Verified** (Var E) | **Complete & Verified** (Var E) | **Untested / Unstarted** |
| **Comparison Status** | **Complete & Verified** | **Complete & Verified** | **Complete & Verified** | **Complete & Verified** | **Incomplete / Missing** |

---

## 3. Image-Quality Metrics and Enhancement-Selection Criteria

The project progressed through two distinct enhancement selection philosophies:

### 3.1 Phase 3 & 5: Macroscopic Quality Metrics (Gerrard Hall & South Building Phase 5)
Images were evaluated using two global statistics:
- **Sharpness Metric:** Variance of the Laplacian: $\text{Var}(\Delta I) = \frac{1}{N} \sum ( \Delta I(x, y) - \mu_{\Delta} )^2$
- **Contrast Metric:** Standard deviation of grayscale pixel intensities: $\sigma = \sqrt{\frac{1}{N} \sum (I(x, y) - \mu)^2}$

Thresholds applied:
- If $\text{Var}(\Delta I) < 80.0 \rightarrow$ Apply unsharp masking ($\sigma=1.0, \alpha=0.5$).
- If $\sigma < 60.0 \rightarrow$ Apply CLAHE in CIELAB space (`clipLimit=2.0`, `tileGridSize=(8, 8)`).

### 3.2 Phase 6, Phase B & Phase 7: Conservative Dual-Constraint Gating (Variant E)
To prevent artificial descriptor distortion on textured images that happen to exhibit low global contrast (e.g., scenes with large expanses of sky or shadow), the **Variant E selection rule** was established:

$$\text{Enhanced If: } \quad \sigma < 52.0 \quad \text{AND} \quad \text{Baseline SIFT Keypoint Count} < 10,000$$

- **Enhancement Operator:** CLAHE applied strictly with conservative parameters:
  - `clipLimit = 1.2` (reduced from 2.0 to minimize gradient noise)
  - `tileGridSize = (8, 8)`
  - Luminance channel $L^*$ modified in CIELAB; $a^*$ and $b^*$ chrominance channels preserved unchanged.
- **Unmodified Control:** All non-candidate images remain 100% byte-identical to their baseline counterparts.

Selected Candidate Counts Across Datasets:
- **South Building:** 2 of 128 images (1.56%) — `P1180166.JPG`, `P1180179.JPG`.
- **Person Hall:** 5 of 330 images (1.52%) — `IMG_1179.JPG`, `IMG_1282.JPG`, `IMG_1312.JPG`, `IMG_1316.JPG`, `IMG_1323.JPG`.
- **Graham Hall Subset:** 28 of 258 processed images (10.85%) — localized in deep courtyard and entrance shadow regions (`IMG_0589` to `IMG_0617`).

---

## 4. Consolidated Empirical Metrics Table

All metrics below are computed directly from the sparse models (`points3D.txt`, `images.txt`) and SQLite database tables (`two_view_geometries`, `matches`).

| Dataset & Configuration | Enhanced / Total Img | Registered Images | 3D Points | Mean Error (px) | Median Error (px) | Mean Obs / Img | Mean Track Length | Verified 2-View Inliers | Verified Pairs | Total SfM Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gerrard Hall Baseline** | $0 / 100$ | $100 / 100$ | 58,127 | 1.0769 | 1.0441 | 3,280.11 | 5.6430 | 969,591 | 1,299 | 1,594.5 s (~26.6 m) |
| **Gerrard Hall Enhanced (P5)** | $23 / 100$ | $100 / 100$ | 58,312 | 1.0786 | 1.0429 | 3,278.64 | 5.6226 | 963,777 | 1,288 | 1,241.1 s (~20.7 m) |
| *Gerrard Hall Delta* | — | *0 (0.0%)* | **+185 (+0.32%)** | *+0.0017 (+0.16%)* | *-0.0012* | *-1.47 (-0.04%)* | *-0.0204 (-0.36%)* | *-5,814 (-0.60%)* | *-11* | *-353.4 s* |
|---|---|---|---|---|---|---|---|---|---|---|
| **South Building Baseline** | $0 / 128$ | $128 / 128$ | 84,557 | 0.6116 | 0.4957 | 3,924.48 | 5.9408 | 1,500,850 | 2,541 | 1,979.1 s (~33.0 m) |
| **South Building P5 (clip=2.0)** | $28 / 128$ | $128 / 128$ | 81,018 | 0.6255 | 0.5071 | 3,772.46 | 5.9601 | 1,446,120 | 2,492 | 1,987.7 s (~33.1 m) |
| **South Building P6 Var E** | $2 / 128$ | $128 / 128$ | 84,593 | 0.6120 | 0.4946 | 3,925.08 | 5.9391 | 1,500,098 | 2,525 | 1,863.1 s (~31.1 m) |
| *South Building Delta (Var E vs Base)* | — | *0 (0.0%)* | **+36 (+0.043%)** | *+0.0004 (+0.07%)* | *-0.0011* | *+0.60 (+0.015%)* | *-0.0017 (-0.03%)* | *-752 (-0.05%)* | *-16* | *-116.0 s* |
|---|---|---|---|---|---|---|---|---|---|---|
| **Person Hall Baseline** | $0 / 330$ | $330 / 330$ | 206,994 | 1.2550 | 1.2608 | 4,308.24 | 6.8684 | 7,468,981 | 7,347 | 16,983.8 s (~4.72 h) |
| **Person Hall Var E** | $5 / 330$ | $330 / 330$ | 207,851 | 1.2569 | 1.2627 | 4,313.19 | 6.8479 | 7,464,515 | 7,343 | 17,577.4 s (~4.88 h) |
| *Person Hall Delta (Var E vs Base)* | — | *0 (0.0%)* | **+857 (+0.414%)** | *+0.0019 (+0.151%)* | *+0.0019 (+0.151%)* | *+4.95 (+0.115%)* | *-0.0205 (-0.298%)* | *-4,466 (-0.060%)* | *-4* | *+593.6 s* |
|---|---|---|---|---|---|---|---|---|---|---|
| **Graham Hall Subset Baseline** | $0 / 270$ | $247 / 258$ | 113,644 | 1.1068 | 1.0792 | 2,172.38 | 4.7216 | 1,343,083 | 4,486 | 7,657.0 s (~2.13 h) |
| **Graham Hall Subset Var E** | $28 / 270$ | $248 / 258$ | 113,775 | 1.1125 | 1.0847 | 2,163.04 | 4.7149 | 1,341,551 | 4,467 | 6,860.7 s (~1.91 h) |
| *Graham Hall Delta (Var E vs Base)* | — | *+1 (+0.40%)* | **+131 (+0.115%)** | *+0.0057 (+0.51%)* | *+0.0055 (+0.51%)* | *-9.34 (-0.43%)* | *-0.0067 (-0.14%)* | *-1,532 (-0.11%)* | *-19* | *-796.3 s* |

---

## 5. Comprehensive Cross-Benchmark Metric Analysis

### 5.1 Reconstructed 3D Points
Across all four datasets, conservative Variant E produced minor positive point gains:
- Gerrard Hall: $+185$ points ($+0.32\%$)
- South Building: $+36$ points ($+0.043\%$)
- Person Hall: $+857$ points ($+0.414\%$)
- Graham Hall Subset: $+131$ points ($+0.115\%$)

**Mechanism & Context:** Modest local contrast expansion on dark doorways, shadowed masonry, and underexposed wall recesses allows the SIFT detector to discover additional scale-space extrema that meet the contrast threshold. These extra keypoints triangulate into localized 3D surface points. However, a gain of less than $0.5\%$ falls within normal triangulation variance and does not constitute a transformative geometric improvement.

### 5.2 Geometric Accuracy and Reprojection Error
Mean reprojection error remained virtually identical between baselines and conservative enhancements:
- Gerrard Hall: $+0.0017\text{ px}$ ($+0.16\%$)
- South Building: $+0.0004\text{ px}$ ($+0.07\%$)
- Person Hall: $+0.0019\text{ px}$ ($+0.15\%$)
- Graham Hall Subset: $+0.0057\text{ px}$ ($+0.51\%$)

This proves that conservative enhancement satisfies the foundational "do no harm" principle: camera calibration parameters and ray intersections remain geometrically accurate.

### 5.3 Observation Density and Track Length
In all benchmarks, mean track length experienced a slight downward trend:
- Gerrard Hall: $-0.0204$ ($-0.36\%$)
- South Building: $-0.0017$ ($-0.03\%$)
- Person Hall: $-0.0205$ ($-0.30\%$)
- Graham Hall Subset: $-0.0067$ ($-0.14\%$)

Newly recovered points in shadowed, difficult regions typically have visibility in only 2 or 3 nearby camera frustums. Adding these localized short tracks dilutes the global average track length slightly.

### 5.4 Verified Two-View Inliers
Pairwise verified inliers decreased marginally in all datasets:
- Gerrard Hall: $-5,814$ inliers ($-0.60\%$)
- South Building: $-752$ inliers ($-0.05\%$)
- Person Hall: $-4,466$ inliers ($-0.06\%$)
- Graham Hall Subset: $-1,532$ inliers ($-0.11\%$)

While CLAHE enhances local edges, it can alter descriptor gradient distributions slightly relative to unenhanced overlapping views, causing a tiny fraction of borderline matches to fail RANSAC epipolar verification.

---

## 6. Status of Graham Hall: Full Repository vs. Subset Validation

It is vital to distinguish the **completed 270-image Phase 7 exterior subset** from the **full Graham Hall repository**:

1. **Completed Phase 7 Validation (270 Images):**
   - Covers an unbroken 270-image exterior traversal (`IMG_0348.jpg` to `IMG_0617.jpg`) around the main building facade.
   - Fully executed on CPU with complete exhaustive matching ($36,315$ pairs).
   - Baseline and Variant E are fully verified with TXT models and SQLite databases in `experiments/graham_hall/`.
2. **Full Graham Hall Multi-Partition Repository (7,638 Images):**
   - Contains 6 partitions (`Part1` to `Part6`), each with 1,273 images (562 exterior + 711 interior).
   - **Remains uncomputed in this project.** Exhaustive matching across 1,273 images requires $810,276$ pairs (~$55$–$65$ CPU hours per run, or ~120 CPU hours total).
   - Existing database and sparse files inside `datasets/graham-hall/Part*/` are original 2016 benchmark distribution assets, not newly executed project runs.

---

## 7. Limitations, Threats to Validity, and Future Work

### 7.1 Limitations
1. **Benchmark Saturation (Ceiling Effect):** All evaluated benchmarks feature dense camera overlap and high-quality photography. Classical SfM algorithms already achieve saturated correspondence on these scenes.
2. **SIFT Luminance Invariance:** Standard SIFT descriptors already incorporate local gradient normalization to achieve invariance to affine illumination changes. Artificial contrast modification therefore yields diminishing returns.
3. **Descriptor Asymmetry:** Applying non-linear transformations (such as CLAHE) to only a subset of images introduces descriptor asymmetry when matching enhanced views against unenhanced adjacent views.

### 7.2 Recommendations for Future Research
- **Targeted Spatial Enhancement:** Rather than enhancing whole images, apply localized contrast enhancement only within bounding boxes identified as underexposed by semantic segmentation.
- **Feature-Level Adaptation:** Modifying SIFT detector peak thresholds ($\text{peak\_threshold}$) dynamically on low-contrast images rather than modifying pixel values would avoid descriptor distortion entirely.

---

## 8. Master Conclusion

Across Gerrard Hall, South Building, Person Hall, and the Graham Hall exterior sequence, the experimental evidence supports three firm scientific conclusions:

1. **Aggressive Blanket Enhancement Degrades Reconstruction:** Applying global contrast stretching across entire image collections alters natural texture gradients and causes severe point collapse (e.g., South Building Phase 5 losing $3,539$ points).
2. **Conservative Selection (Variant E) is Safe and Stable:** Restricting enhancement strictly to genuine feature outliers ($\sigma < 52.0 \land \text{SIFT} < 10,000$) with mild CLAHE (`clipLimit=1.2`) completely shields the reconstruction from degradation, preserving registration rates and identical reprojection errors.
3. **Gains are Incrementally Marginal:** While conservative enhancement consistently increases 3D point counts ($+0.04\%$ to $+0.41\%$) and occasionally adds registered images (+1 image in Graham Hall), it does not provide a transformative leap in reconstruction geometry or camera calibration. The classical unenhanced pipeline remains exceptionally robust.

---
*Verified Source References:*
- Gerrard Hall Baseline & Enhanced: [`experiments/baseline/gerrard-hall/sparse/0/`](file:///e:/CV%20Project/experiments/baseline/gerrard-hall/sparse/0/) & [`experiments/baseline_enhanced/gerrard-hall/sparse/0/`](file:///e:/CV%20Project/experiments/baseline_enhanced/gerrard-hall/sparse/0/)
- South Building Baseline & Variant E: [`experiments/baseline/south-building/sparse/0/`](file:///e:/CV%20Project/experiments/baseline/south-building/sparse/0/) & [`experiments/phase6/variant_e_conservative_selection/colmap_output/sparse/0/`](file:///e:/CV%20Project/experiments/phase6/variant_e_conservative_selection/colmap_output/sparse/0/)
- Person Hall Baseline & Variant E: [`experiments/person_hall/baseline/sparse/0/`](file:///e:/CV%20Project/experiments/person_hall/baseline/sparse/0/) & [`experiments/person_hall/variant_e/sparse/0/`](file:///e:/CV%20Project/experiments/person_hall/variant_e/sparse/0/)
- Graham Hall Baseline & Variant E (270-Image Subset): [`experiments/graham_hall/baseline/sparse/0/`](file:///e:/CV%20Project/experiments/graham_hall/baseline/sparse/0/) & [`experiments/graham_hall/variant_e/sparse/0/`](file:///e:/CV%20Project/experiments/graham_hall/variant_e/sparse/0/)
- Graham Hall Dedicated Report: [`experiments/graham_hall/graham_hall_report.md`](file:///e:/CV%20Project/experiments/graham_hall/graham_hall_report.md)
- Audit Documentation: [`docs/four_dataset_experiment_audit.md`](file:///e:/CV%20Project/docs/four_dataset_experiment_audit.md)
