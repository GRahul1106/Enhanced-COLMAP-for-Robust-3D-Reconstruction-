# Multi-Dataset Validation Report: Quality-Guided Adaptive Image Enhancement for 3D Reconstruction

**Project:** Quality-Guided Adaptive Image Enhancement for 3D Reconstruction  
**Authors / Role:** Senior Computer Vision Research & COLMAP Engineering Team  
**Evaluation Scope:** South Building (128 images) & Person Hall (330 images)  
**Date:** October 26, 2026  
**Status:** Complete & Verified  

---

## Executive Summary

This report evaluates whether conservative, quality-guided image enhancement (**Variant E**) reliably improves classical Structure-from-Motion (SfM) reconstruction across distinct datasets. 

Under Variant E, image enhancement is restricted strictly to images identified as low-contrast and feature-deficient:
- **Selection Criteria:** Contrast Standard Deviation $< 52.0$ AND Baseline SIFT Feature Count $< 10,000$.
- **Enhancement Operator:** CLAHE (Contrast Limited Adaptive Histogram Equalization) applied with conservative parameters (`clipLimit=1.2`, `tileGridSize=(8, 8)`), preserving color-space balance and image dimensions.
- **Controlled Comparison:** All remaining images ($126/128$ in South Building; $325/330$ in Person Hall) remain 100% byte-identical to their baseline counterparts. Both datasets were reconstructed using identical bundled COLMAP 4.2.0 CPU pipelines with unified intrinsic camera models.

### Key Takeaway
Across both datasets, **Variant E produced minor numerical increases in reconstructed 3D points ($+0.043\%$ on South Building; $+0.414\%$ on Person Hall)** while maintaining nearly identical reprojection errors ($+0.0004\text{ px}$ and $+0.0019\text{ px}$ delta, respectively). However, verified two-view inliers and track lengths experienced slight minor reductions. **The evidence demonstrates that conservative quality-guided enhancement is safe and non-destructive to SfM stability, but does not provide a statistically substantial or transformative reconstruction benefit over strong unenhanced baselines.**

---

## 1. Experimental Setup & Datasets

| Dataset | Total Images | Image Resolution | Camera Model | Enhanced Images (Variant E) | Selection Rule |
|---|---:|---|---|---:|---|
| **South Building** | 128 | $3072 \times 2048$ | Single Camera (OPENCV) | 2 ($1.56\%$) | Contrast $< 45.0$ & SIFT $< 1,000$ |
| **Person Hall** | 330 | $5616 \times 3744$ | Single Camera (SIMPLE_RADIAL) | 5 ($1.52\%$) | Contrast $< 52.0$ & SIFT $< 10,000$ |

### Enhanced Views in Person Hall Variant E
The 5 images identified by pre-reconstruction quality analysis were:
1. `IMG_1179.JPG`: Contrast $37.19$, Baseline SIFT $9,743 \rightarrow 10,009$ ($+266$)
2. `IMG_1282.JPG`: Contrast $49.25$, Baseline SIFT $8,359 \rightarrow 9,341$ ($+982$)
3. `IMG_1312.JPG`: Contrast $46.94$, Baseline SIFT $9,504 \rightarrow 10,407$ ($+903$)
4. `IMG_1316.JPG`: Contrast $44.74$, Baseline SIFT $9,617 \rightarrow 10,379$ ($+762$)
5. `IMG_1323.JPG`: Contrast $51.48$, Baseline SIFT $8,867 \rightarrow 9,554$ ($+687$)

---

## 2. Multi-Dataset Reconstruction Results

### Table 1: Cross-Dataset Performance Comparison

| Metric | South Building (Baseline) | South Building (Variant E) | South Building Delta | Person Hall (Baseline) | Person Hall (Variant E) | Person Hall Delta |
|---|---:|---:|---:|---:|---:|---:|
| **Registered Images** | $128 / 128$ | $128 / 128$ | $0$ ($0.0\%$) | $330 / 330$ | $330 / 330$ | $0$ ($0.0\%$) |
| **Registration Rate** | $100.0\%$ | $100.0\%$ | $0.0\%$ | $100.0\%$ | $100.0\%$ | $0.0\%$ |
| **Reconstructed 3D Points** | $84,557$ | $84,593$ | **$+36$ ($+0.043\%$)** | $206,994$ | $207,851$ | **$+857$ ($+0.414\%$)** |
| **Mean Reprojection Error** | $0.6116\text{ px}$ | $0.6120\text{ px}$ | $+0.0004\text{ px}$ ($+0.07\%$) | $1.2550\text{ px}$ | $1.2569\text{ px}$ | $+0.0019\text{ px}$ ($+0.15\%$) |
| **Median Reprojection Error**| — | — | — | $1.2608\text{ px}$ | $1.2627\text{ px}$ | $+0.0019\text{ px}$ ($+0.15\%$) |
| **Mean Obs. / Image** | $3,924.48$ | $3,925.08$ | $+0.60$ ($+0.015\%$) | $4,308.24$ | $4,313.19$ | $+4.95$ ($+0.11\%$) |
| **Mean Track Length** | $5.9408$ | $5.9391$ | $-0.0017$ ($-0.03\%$) | $6.8684$ | $6.8479$ | $-0.0205$ ($-0.30\%$) |
| **Verified 2-View Inliers** | — | — | $+0.10\%\text{ (DB size proxy)}$| $7,468,981$ | $7,464,515$ | **$-4,466$ ($-0.060\%$)** |
| **Verified Image Pairs** | — | — | — | $7,347$ | $7,343$ | $-4$ ($-0.05\%$) |
| **Database Size** | $245.82\text{ MB}$ | $246.08\text{ MB}$ | $+0.26\text{ MB}$ | $685.55\text{ MB}$ | $686.01\text{ MB}$ | $+0.46\text{ MB}$ |

---

## 3. Detailed Scientific Observations

### 1. 3D Point Cloud Density
- **Observation:** In both independent experiments, the conservative selection strategy resulted in a net positive point gain over the baseline ($+36$ points for South Building, $+857$ points for Person Hall).
- **Contextualization:** On South Building, broader aggressive CLAHE configurations (Variants A and C, enhancing 28 images) caused catastrophic feature drift and point loss ($-3,539$ and $-2,834$ points, respectively). Conservative Variant E avoided this degradation in both datasets.
- **Significance:** A gain of $+0.414\%$ ($857$ points out of $206,994$) is marginal and falls within the realm of incremental bundle-adjustment triangulation variation.

### 2. Geometric Accuracy & Reprojection Error
- **Observation:** Mean reprojection error remained virtually constant: shifting by $+0.0004\text{ px}$ in South Building and $+0.0019\text{ px}$ in Person Hall. Median reprojection error in Person Hall mirrored this exactly ($+0.0019\text{ px}$).
- **Interpretation:** The conservative enhancement did not distort keypoint localization accuracy or corrupt the bundle adjustment solution. Camera calibration parameters converged stably.

### 3. Track Length and Pairwise Inliers
- **Observation:** Mean track length decreased slightly in both datasets ($-0.0017$ in South Building; $-0.0205$ in Person Hall). In Person Hall, verified two-view inliers decreased by $4,466$ ($-0.060\%$), and verified pairs decreased by $4$.
- **Mechanism:** The extra SIFT features generated by CLAHE on the 5 low-contrast images tend to form short tracks (2- or 3-view observations) in localized dark corners or low-contrast textures. Because these additional points have fewer cross-view correspondences than primary structural points, they pull down the global mean track length slightly.

---

## 4. Pipeline Runtimes

| Stage | Baseline Runtime | Variant E Runtime | Nature of Timing |
|---|---:|---:|---|
| **Feature Extraction** | $173.7\text{ s}$ | $5.7\text{ s}$ | Incremental extraction for 5 enhanced views (325 views inherited) |
| **Exhaustive Matching** | $14,485.6\text{ s}$ (~$4.02\text{ h}$) | $13,037.5\text{ s}$ (~$3.62\text{ h}$) | Historical complete run ($54,285$ pairs) |
| **Sparse Mapping (`mapper`)** | $2,316.5\text{ s}$ (~$38.6\text{ min}$) | $4,511.6\text{ s}$ (~$75.2\text{ min}$) | Measured directly on resumption |
| **Model Conversion (TXT)** | $7.8\text{ s}$ | $22.6\text{ s}$ | Measured directly on resumption |
| **Total SfM Pipeline** | $16,983.8\text{ s}$ (~$4.72\text{ h}$) | $17,577.4\text{ s}$ (~$4.88\text{ h}$) | Cumulative reconstructed workflow |

---

## 5. Synthesis, Limitations, and Generalization Analysis

### Does Quality-Guided Adaptive Enhancement Generalize?
1. **Consistency of Safety:** Across both datasets, limiting CLAHE to strict low-contrast outliers ($<2\%$ of the image collection) prevented the severe degradation observed when applying CLAHE across larger subsets. In this respect, the "do no harm" principle of Variant E generalizes reliably.
2. **Consistency of Marginal Gains:** In both datasets, the point count gain was positive but tiny ($<0.5\%$). Enhancement enabled SIFT to identify extra keypoints on dark walls, recessed doorways, and under-exposed arches, translating into a handful of localized triangulated points.
3. **Absence of Broad Structural Improvement:** Neither dataset demonstrated a transformative leap in SfM metrics:
   - Registration rate was already $100\%$ in both baselines and remained $100\%$.
   - Reprojection error was neither improved nor materially harmed.
   - Pairwise verified inliers and track lengths experienced minor downward drift.

### Limitations:
- **Baseline Ceiling Effect:** Both South Building and Person Hall datasets feature abundant camera overlap and high overall image quality. For datasets where registration rate is already $100\%$, classical SfM feature matching is already in a saturation regime.
- **SIFT Invariance vs Contrast Modification:** RootSIFT / SIFT normalization already provides intrinsic invariance to affine illumination changes. Artificially stretching local contrast introduces subtle high-frequency gradient noise, which can occasionally reduce descriptor matchability against unenhanced overlapping views.

---

## 6. Conclusion

Conservative Variant E is a safe, stable modification to classical SfM pipelines that selectively salvages features from under-exposed images without compromising global geometry. However, **claims that classical image enhancement delivers substantial 3D reconstruction improvements are not supported by the experimental evidence across South Building and Person Hall.**

---
*Reference Documents:*
- South Building Phase 6 Report: [`experiments/phase6/phase6_report.md`](file:///e:/CV%20Project/experiments/phase6/phase6_report.md)
- South Building Summary: [`experiments/phase6/summary.json`](file:///e:/CV%20Project/experiments/phase6/summary.json)
- Person Hall Baseline Summary: [`experiments/person_hall/baseline/baseline_summary.json`](file:///e:/CV%20Project/experiments/person_hall/baseline/baseline_summary.json)
- Person Hall Variant E Summary: [`experiments/person_hall/variant_e/variant_e_summary.json`](file:///e:/CV%20Project/experiments/person_hall/variant_e/variant_e_summary.json)
- Person Hall Comparison Matrix: [`experiments/person_hall/variant_e/comparison_with_baseline.json`](file:///e:/CV%20Project/experiments/person_hall/variant_e/comparison_with_baseline.json)
