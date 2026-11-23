# Phase 6: Enhancement Optimization and Ablation Study
## Controlled Experimental Design and Execution Plan

**Document Version:** 1.0  
**Date:** October 11, 2026  
**Target Dataset:** South Building (`datasets/south-building`, 128 images)  
**Output Base:** `experiments/phase6/`

---

## 1. Executive Context & Objective

In Phase 5, multi-dataset evaluation revealed a critical divergence in reconstruction behavior:
- **Gerrard Hall (100 images):** Point count increased from 58,127 to 58,312 (+185 points, +0.32%).
- **South Building (128 images):** Point count decreased from 84,557 to 81,018 (-3,539 points, -4.19%), while mean reprojection error increased slightly (+2.27%).

In South Building, natural sharpness was exceptionally high (min Laplacian variance 122.79 vs threshold 80.0), so 0 images received sharpening. However, 28 images (21.88%) failed the contrast threshold ($\sigma < 60.0$) and received CLAHE (`clipLimit=2.0`, `tileGridSize=(8, 8)` in CIELAB $L$-space).

The objective of Phase 6 is to systematically isolate whether **excessive CLAHE clipping intensity** caused the point-count reduction, determine if milder clipping restores or improves reconstruction fidelity, and formulate an improved selection/enhancement strategy.

---

## 2. Experimental Variants

| Variant ID | Name | Enhancement Method | Target Images | Parameters | Output Location |
|---|---|---|---|---|---|
| **A** | `baseline` | Unmodified Original | None (0 / 128) | N/A | `experiments/baseline/south-building/` *(Completed in Phase 5)* |
| **B** | `variant_a_clip2.0` | Current Adaptive (Re-validation) | Same 28 images | CLAHE `clipLimit=2.0`, `tileGrid=(8, 8)` | `experiments/phase6/variant_a_clip2.0/` |
| **C** | `variant_b_clip1.5` | Mild CLAHE | Same 28 images | CLAHE `clipLimit=1.5`, `tileGrid=(8, 8)` | `experiments/phase6/variant_b_clip1.5/` |
| **D** | `variant_c_clip1.2` | Conservative CLAHE | Same 28 images | CLAHE `clipLimit=1.2`, `tileGrid=(8, 8)` | `experiments/phase6/variant_c_clip1.2/` |
| **E** | `variant_e_conservative_sel` | Conservative Selection Strategy | Subset of 28 (e.g., contrast < 52.0) | CLAHE `clipLimit=1.2`, `tileGrid=(8, 8)` | `experiments/phase6/variant_e_conservative_sel/` |

### Key Controls
1. **Isolated Variable:** In Variants B, C, and D, the **exact same 28 images** are enhanced, and the 100 normal images remain identical bit-for-bit copies. Only `clipLimit` varies ($2.0 \rightarrow 1.5 \rightarrow 1.2$).
2. **Identical SfM Parameters:** All COLMAP runs use identical flags:
   - SIFT extraction: single camera model (`SIMPLE_RADIAL`), shared intrinsics prior.
   - Matcher: `exhaustive_matcher`.
   - Mapper: standard bundle adjustment and triangulation.
3. **Data Protection:** Original dataset images and existing Phase 5 results remain strictly read-only and preserved.

---

## 3. Pre-Reconstruction Quality Analysis Findings

Statistical evaluation across the 28 CLAHE-selected images (`analyze_ablation_quality.py`) reveals the following behavior:

| Metric | Original (Baseline) | Variant A (`clip=2.0`) | Variant B (`clip=1.5`) | Variant C (`clip=1.2`) | Trend / Interpretation |
|---|---|---|---|---|---|
| **Mean Contrast ($\sigma$)** | 53.37 | 64.73 | 62.06 | 60.21 | Milder clipping brings contrast just to the 60 threshold |
| **Laplacian Var (Blur/Noise)** | 273.08 | 826.28 (+202%) | 668.87 (+145%) | 572.31 (+109%) | High clipping severely magnifies high-frequency noise |
| **Mean Intensity** | 127.45 | 128.56 | 128.85 | 128.96 | Mean exposure remains well-balanced across all variants |
| **Histogram Correlation vs Original** | 1.0000 | 0.8045 | 0.8625 | 0.8915 | Variant C preserves 89.2% of natural histogram shape |
| **Raw OpenCV SIFT Keypoints** | 95,947 | 142,907 (+48.9%) | 137,026 (+42.8%) | 131,643 (+37.2%) | Raw SIFT triggers heavily on noise/tile edges at clip=2.0 |

### Scientific Hypothesis
- At `clipLimit=2.0`, CLAHE over-enhances subtle continuous architectural surfaces and induces high Laplacian variance (826 vs 273), which disrupts cross-view descriptor matching against adjacent unenhanced views.
- At `clipLimit=1.2` (Variant C), the image undergoes gentle contrast enhancement ($\sigma = 60.21$), maintaining high histogram fidelity ($r = 0.8915$) and controlled noise amplification (572 vs 826).

---

## 4. Computational Budget & Resource Risk Assessment

### 4.1 Environment Profile
- **COLMAP Binary:** Version 4.2.0 (Commit be5e291, Windows x86_64, CPU / OpenCL SIFT).
- **GPU Acceleration:** CUDA dense reconstruction is disabled. SIFT feature extraction and matching execute across CPU worker threads.

### 4.2 Per-Reconstruction Timings (South Building, 128 images)
Based on Phase 5 verified execution logs:
- **Feature Extraction (SIFT):** ~40–75 seconds
- **Exhaustive Feature Matching:** ~1,630–1,690 seconds (~27–28 minutes)
- **Incremental SfM (Mapper):** ~260–270 seconds (~4.5 minutes)
- **Model Conversion:** ~3 seconds
- **Total Duration per Variant:** **~33.0 minutes** (~1,980 seconds)

### 4.3 Sequential Execution Strategy
Running all 3 ablation variants back-to-back would require $\approx 100$ minutes.
**Recommended Protocol:**
1. Execute **Variant C (`variant_c_clip1.2`) first**.
   - If `clip=1.2` successfully reverses the point-count loss, it confirms the over-clipping hypothesis with immediate positive utility.
2. If time permits, run Variant B (`clip=1.5`) to map the full parameter response curve.
3. Keep Phase 5 results as the historical benchmark for `clip=2.0` (Variant B), avoiding redundant computation unless re-verification in the new directory is specifically requested.
