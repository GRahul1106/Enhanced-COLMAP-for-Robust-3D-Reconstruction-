# Consolidated Multi-Dataset Evaluation Report
## Adaptive Image Enhancement for 3D Reconstruction (COLMAP)
**Generated:** 2026-10-10 23:27:01  
**Phase:** 5 — Multi-Dataset Evaluation & Robustness Testing  

---

## 1. Experiment Completion & Comparability Matrix

| Dataset | Input Images | Baseline SfM | Enhanced Images | Enhanced SfM | Valid Comparison? | Status / Notes |
|---|---|---|---|---|---|---|
| **gerrard-hall** | 100 | ✅ Complete | ✅ Done (23 enh) | ✅ Complete | ✅ Yes | Completed & Evaluated |
| **south-building** | 128 | ✅ Complete | ✅ Done (28 enh) | ✅ Complete | ✅ Yes | Completed & Evaluated |
| **person-hall** | 330 | ❌ Missing | ❌ Missing | ⏳ Pending | ❌ Incomplete | Planned |
| **graham-hall** | 1273 | ❌ Missing | ❌ Missing | ⏳ Pending | ❌ Incomplete | Planned |

---

## 2. Quantitative Baseline vs. Enhanced Comparison

| Dataset | Metric | Original Baseline | Adaptive Enhanced | Absolute Diff | Relative Diff (%) | Interpretation |
|---|---|---|---|---|---|---|
| **gerrard-hall** | Registered Images | 100 (100.0%) | 100 (100.0%) | 0 | 0.0% | Perfect registration maintained |
| | Reconstructed 3D Points | 58,127 | 58,312 | +185 | +0.32% | +185 points added |
| | Mean Reprojection Error | 1.0769 px | 1.0786 px | +0.0017 px | +0.16% | Slightly higher error |
| | Mean Observations / Image | 3280.11 | 3278.64 | -1.47 | -0.04% | Comparable coverage |
| | Mean Track Length | 5.6430 | 5.6226 | -0.0204 | -0.36% | Stable track length |
| **south-building** | Registered Images | 128 (100.0%) | 128 (100.0%) | 0 | 0.0% | Perfect registration maintained |
| | Reconstructed 3D Points | 84,557 | 81,018 | -3,539 | -4.19% | -3539 points |
| | Mean Reprojection Error | 0.6116 px | 0.6255 px | +0.0139 px | +2.27% | Slightly higher error |
| | Mean Observations / Image | 3924.48 | 3772.46 | -152.02 | -3.87% | Observation density shifted |
| | Mean Track Length | 5.9408 | 5.9601 | +0.0193 | +0.32% | Stable track length |

---

## 3. Scientific Analysis & Observations

1. **3D Points vs. Quality:** Adding 3D points (+185 in Gerrard Hall) demonstrates that enhancement exposed additional features in low-contrast regions. However, more 3D points do not automatically guarantee superior reconstruction geometry.
2. **Reprojection Error:** The slight increase (+0.0017 px / +0.16%) in Gerrard Hall reprojection error is typical when triangulating features in previously weak or marginal contrast regions.
3. **Dataset Diversity:** South Building exhibits a markedly higher natural sharpness (minimum Laplacian variance 122.79 vs Gerrard Hall's 37.14). Hence, sharpening was selectively skipped, while 28 low-contrast images (21.9%) were enhanced via CLAHE.

---

## 4. Processing Runtime & Bottleneck Summary

| Dataset | Quality Analysis (s) | Enhancement (s) | Baseline SfM Total (s) | Enhanced SfM Total (s) | SfM Time Diff (s) |
|---|---|---|---|---|---|
| **gerrard-hall** | 49.0 | 9.0 | 1594.5 | 1241.1 | -353.4s |
| **south-building** | 27.0 | 4.0 | 1979.1 | 1987.7 | +8.6s |
| **person-hall** | N/A | N/A | Pending | Pending | N/A |
| **graham-hall** | N/A | N/A | Pending | Pending | N/A |
