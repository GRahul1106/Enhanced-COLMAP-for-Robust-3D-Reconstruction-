# Quality-Guided Adaptive Image Enhancement for 3D Reconstruction
## Project Context Recovery and Phase 1–5 History

**Document Version:** 1.0  
**Recovery Date:** October 11, 2026  
**Project Root:** `E:\CV Project`  
**Corpus / Repository:** `GRahul1106/Enhanced-COLMAP-for-Robust-3D-Reconstruction-`  
**Execution Environment:** Windows (x86_64), Python 3.8+, COLMAP 4.2.0 (Commit be5e291, CPU/OpenCL SIFT)

---

## 1. Executive Summary & Project Identity

### 1.1 Core Thesis & Objective
Structure-from-Motion (SfM) pipelines like COLMAP rely critically on consistent, repeatable local visual features (SIFT keypoints and descriptors) across multiple camera views. In real-world multi-view architectural datasets, images frequently suffer from localized visual degradation, notably:
- Low illumination or shadow contrast
- High-frequency blur (motion or defocus)
- Unfavorable dynamic range

The core objective of this project is to develop and evaluate a **quality-guided adaptive image enhancement framework** that selectively enhances degraded images using classical computer vision techniques (CLAHE and Unsharp Masking) prior to feature extraction and matching, comparing the resulting 3D reconstructions against the unenhanced baseline in COLMAP.

### 1.2 Classical Computer Vision Constraint
This project strictly employs classical image processing (NumPy, OpenCV, Matplotlib, SciPy) and COLMAP. Deep-learning enhancement (e.g., neural denoisers, super-resolution, or learned feature extractors) and alternative SfM frameworks are excluded unless explicitly requested.

---

## 2. Recovered Phase History (Phases 1–5)

```mermaid
flowchart TD
    subgraph Phase1["Phase 1: Architecture & Baseline Setup"]
        A1[Multi-View Dataset] --> A2[COLMAP Baseline Runner]
        A2 --> A3[Baseline Sparse Reconstruction]
    end

    subgraph Phase2["Phase 2: Image Quality Assessment (IQA)"]
        B1[Raw Images] --> B2[Laplacian Variance (Blur)]
        B1 --> B3[Intensity StdDev (Contrast)]
        B1 --> B4[Mean Intensity (Brightness)]
        B1 --> B5[Immerkaer Filter (Noise)]
        B2 & B3 & B4 & B5 --> B6[Per-Image Quality CSV & JSON Summary]
    end

    subgraph Phase3["Phase 3: Quality-Guided Adaptive Enhancement"]
        B6 --> C1{Quality Decision Engine}
        C1 -- "Contrast < 60" --> C2[CLAHE in LAB L-channel]
        C1 -- "Blur < 80" --> C3[Unsharp Masking]
        C1 -- "Both Triggers" --> C4[CLAHE + Unsharp Masking]
        C1 -- "Neither Trigger" --> C5[Direct Bit-for-Bit Copy]
        C2 & C3 & C4 & C5 --> C6[Enhanced Image Dataset]
    end

    subgraph Phase4_5["Phases 4 & 5: SfM Evaluation & Comparative Analysis"]
        C6 --> D1[COLMAP Enhanced SfM Runner]
        D1 --> D2[Enhanced Sparse Reconstruction]
        A3 & D2 --> D3[Consolidated Evaluator]
        D3 --> D4[Quantitative Metrics: Points, Error, Observations, Timings]
    end
```

---

### Phase 1 — Problem Definition, Architecture, and Baseline Reconstruction

- **Objective:** Establish the non-destructive COLMAP Structure-from-Motion baseline workflow. Ensure original dataset assets are permanently protected as read-only inputs while isolating all output artifacts.
- **Implementation:**
  - Script: [`src/baseline/run_colmap_baseline.py`](file:///e:/CV%20Project/src/baseline/run_colmap_baseline.py)
  - Pre-built bundled COLMAP installation: `colmap/COLMAP.bat` and `colmap/bin/colmap.exe`.
  - Windows path handling: Direct invocation of `colmap.exe` with process environment variables (`PATH`, `QT_PLUGIN_PATH`) to prevent `cmd.exe` path-with-spaces truncation on `"E:\CV Project"`.
  - Stages implemented:
    1. `feature_extractor` (`--ImageReader.single_camera 1`, SIFT)
    2. `exhaustive_matcher` (all $N(N-1)/2$ image pairs)
    3. `mapper` (incremental bundle adjustment and triangulation)
    4. `model_converter` (exports binary `.bin` camera poses and point clouds to human-readable `.txt`)
    5. Optional dense reconstruction stages (`image_undistorter`, `patch_match_stereo`, `stereo_fusion` — omitted when GPU CUDA is unavailable).
- **Dataset Safety Enforcements:** Hardcoded assertion prohibiting `--output` inside `--dataset`; COLMAP accesses `--image_path` strictly in read-only mode.

---

### Phase 2 — Image-Quality Assessment (IQA)

- **Objective:** Measure objective per-image quality metrics to detect blur, poor contrast, extreme exposure, and high-frequency sensor noise across all images in a dataset without altering original files.
- **Implementation:**
  - Script: [`src/enhanced/analyze_image_quality.py`](file:///e:/CV%20Project/src/enhanced/analyze_image_quality.py)
- **Mathematical Formulations & Code Verification:**
  1. **Blur / Sharpness:** Variance of the discrete Laplacian operator:
     $$\text{Blur} = \text{Var}(\nabla^2 I_{gray})$$
     Implemented via `cv2.Laplacian(gray, cv2.CV_64F).var()`. High value indicates sharp edges; low value indicates blur.
  2. **Contrast:** Sample standard deviation of pixel intensities:
     $$\text{Contrast} = \sigma(I_{gray}) = \sqrt{\frac{1}{HW}\sum_{x,y} (I(x,y) - \mu)^2}$$
     Implemented via `float(np.std(gray))`.
  3. **Brightness / Exposure:** Sample mean pixel intensity:
     $$\text{Brightness} = \mu(I_{gray}) = \frac{1}{HW}\sum_{x,y} I(x,y)$$
     Implemented via `float(np.mean(gray))`. Values $< 60$ indicate underexposure; $> 190$ indicate overexposure.
  4. **Noise:** Immerkaer's fast noise variance estimator using the $3 \times 3$ Laplacian residual operator $N$:
     $$N = \begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix}, \quad \sigma_n = \frac{\sqrt{\pi/2}}{6(W-2)(H-2)} \sum_{x,y} |(I * N)(x,y)|$$
     Implemented via `cv2.filter2D` with normalization factor `np.sqrt(0.5 * np.pi) / (6.0 * (W - 2) * (H - 2))`.
- **Output Artifacts:**
  - `experiments/quality_analysis/<dataset>/image_quality.csv`
  - `experiments/quality_analysis/<dataset>/quality_summary.json`
  - `experiments/quality_analysis/<dataset>/plots/*.png`

---

### Phase 3 — Adaptive Quality-Guided Enhancement

- **Objective:** Selectively apply localized classical enhancement algorithms only to images failing defined empirical quality thresholds, preserving pristine images without alteration.
- **Implementation:**
  - Script: [`src/enhanced/adaptive_enhancement.py`](file:///e:/CV%20Project/src/enhanced/adaptive_enhancement.py)
- **Decision Rules & Verified Thresholds:**
  - **Blur Threshold:** $\tau_{\text{blur}} = 80.0$
  - **Contrast Threshold:** $\tau_{\text{contrast}} = 60.0$
  - **Decision Matrix:**
    1. If $\text{Contrast} < 60.0$ and $\text{Blur} < 80.0 \implies$ **Both CLAHE and Unsharp Masking**
    2. If $\text{Contrast} < 60.0$ and $\text{Blur} \ge 80.0 \implies$ **CLAHE Only**
    3. If $\text{Contrast} \ge 60.0$ and $\text{Blur} < 80.0 \implies$ **Sharpen Only**
    4. If $\text{Contrast} \ge 60.0$ and $\text{Blur} \ge 80.0 \implies$ **Unchanged (direct bit-for-bit file copy)**
- **Algorithm Implementations & Parameters:**
  - **CLAHE:** Applied strictly in the CIELAB color space on the Luminance ($L$) channel to prevent chromatic distortion:
    - `clipLimit = 2.0`
    - `tileGridSize = (8, 8)`
    - Channels $A$ and $B$ recombined unchanged before converting back to BGR.
  - **Unsharp Masking:** High-frequency edge emphasis via Gaussian spatial blurring:
    - $\text{blurred} = \text{GaussianBlur}(I, (0, 0), \sigma=1.0)$
    - $I_{\text{sharp}} = \text{clip}((1 + \alpha) I - \alpha \cdot \text{blurred}, 0, 255)$ with $\alpha = 0.5$.
  - **Denoising:** Denoising was explicitly evaluated and omitted in Phase 3A to avoid edge suppression.
- **Verification Routine:** Built-in verification asserts that input image count equals output image count, CSV rows equal input count, and categorical counts sum to $100\%$. Side-by-side verification previews (`[Original | Enhanced]`) are automatically rendered.

---

### Phase 4 — COLMAP Reconstruction Pipeline Integration

- **Objective:** Execute identical COLMAP reconstruction pipelines on original image collections versus adaptively enhanced collections to establish fair, repeatable comparisons.
- **Implementation & Structure:**
  - Baseline runs stored at: `experiments/baseline/<dataset>/`
  - Enhanced reconstruction runs stored at: `experiments/baseline_enhanced/<dataset>/`
  - Clean publication models copied to: `results/original_colmap/<dataset>/` and `results/enhanced_colmap/<dataset>/`
  - Identical COLMAP commands executed:
    - `feature_extractor --database_path <db> --image_path <images> --ImageReader.single_camera 1`
    - `exhaustive_matcher --database_path <db>`
    - `mapper --database_path <db> --image_path <images> --output_path <sparse>`
    - `model_converter --input_path <sparse/0> --output_path <sparse/0> --output_type TXT`

---

### Phase 5 — Multi-Dataset Evaluation & Robustness Testing

- **Objective:** Aggregate multi-metric quantitative evaluations across heterogeneous architectural datasets.
- **Implementation:**
  - Evaluation Script: [`src/evaluation/evaluate_experiments.py`](file:///e:/CV%20Project/src/evaluation/evaluate_experiments.py)
  - Outputs generated:
    - [`experiments/evaluation/report.md`](file:///e:/CV%20Project/experiments/evaluation/report.md)
    - [`experiments/evaluation/summary.csv`](file:///e:/CV%20Project/experiments/evaluation/summary.csv)
    - [`experiments/evaluation/summary.json`](file:///e:/CV%20Project/experiments/evaluation/summary.json)
    - [`experiments/evaluation/plots/`](file:///e:/CV%20Project/experiments/evaluation/plots/)
- **Datasets Analyzed & Verification Status:**

| Dataset | Total Images | Image Distribution | Verified Baseline Status | Verified Enhanced Status | Completed Comparison? |
|---|---|---|---|---|---|
| **Gerrard Hall** | 100 | Single folder (`images/`) | Complete (`sparse/0/*.txt`) | Complete (`sparse/0/*.txt`) | **Yes** (Valid) |
| **South Building** | 128 | Single folder (`images/`) | Complete (`sparse/0/*.txt`) | Complete (`sparse/0/*.txt`) | **Yes** (Valid) |
| **Person Hall** | 330 | Part1: 165, Part2: 165 | Not Run | Not Run | **No** (Pending) |
| **Graham Hall** | 1,273 per Part | 6 Parts (each 562 ext + 711 int) | Not Run | Not Run | **No** (Pending) |

- **Verified Empirical Results:**

#### 1. Gerrard Hall (100 Images)
- **Enhancement Selection:** 23 enhanced (23.0%), 77 unchanged (77.0%).
  - CLAHE only: 5 images
  - Sharpen only: 17 images
  - Both CLAHE & Sharpen: 1 image
- **Baseline Reconstruction:**
  - Registered Images: 100 / 100 (100.0%)
  - Reconstructed 3D Points: **58,127**
  - Mean Reprojection Error: **1.0769 px**
  - Mean Observations / Image: 3,280.11
  - Mean Track Length: 5.6430
  - Total SfM Runtime: 1,594.5 s (~26.6 min)
- **Enhanced Reconstruction:**
  - Registered Images: 100 / 100 (100.0%)
  - Reconstructed 3D Points: **58,312** (+185 points, **+0.32%**)
  - Mean Reprojection Error: **1.0786 px** (+0.0017 px, **+0.16%**)
  - Mean Observations / Image: 3,278.64 (-1.47, -0.04%)
  - Mean Track Length: 5.6226 (-0.0204, -0.36%)
  - Total SfM Runtime: 1,241.1 s (~20.7 min)
- **Interpretation:** Marginal improvement in point count (+185 points) with virtually unchanged reprojection error.

#### 2. South Building (128 Images)
- **Enhancement Selection:** 28 enhanced (21.88%), 100 unchanged (78.12%).
  - Contrast Range: Min 46.82, Mean 68.51, Max 84.26 $\implies$ exactly 28 images $< 60.0$.
  - Blur Range: Min 122.79, Mean 270.20, Max 514.42 $\implies$ **0 images $< 80.0$**.
  - All 28 selected images received **CLAHE only** (clipLimit=2.0, tileGrid=(8,8)).
- **Baseline Reconstruction:**
  - Registered Images: 128 / 128 (100.0%)
  - Reconstructed 3D Points: **84,557**
  - Mean Reprojection Error: **0.6116 px**
  - Mean Observations / Image: 3,924.48
  - Mean Track Length: 5.9408
  - Total SfM Runtime: 1,979.1 s (~33.0 min)
- **Enhanced Reconstruction:**
  - Registered Images: 128 / 128 (100.0%)
  - Reconstructed 3D Points: **81,018** (-3,539 points, **-4.19%**)
  - Mean Reprojection Error: **0.6255 px** (+0.0139 px, **+2.27%**)
  - Mean Observations / Image: 3,772.46 (-152.02, -3.87%)
  - Mean Track Length: 5.9601 (+0.0193, **+0.32%**)
  - Total SfM Runtime: 1,987.7 s (~33.1 min)
- **Interpretation:** **Point-count degradation (-3,539 points)** and slight reprojection error inflation (+2.27%).

---

## 3. Discrepancy Analysis & Key Scientific Insights

### 3.1 Why Did South Building Degrade While Gerrard Hall Improved?
Deep inspection of the COLMAP execution logs (`experiments/baseline/south-building/baseline.log` vs `experiments/baseline_enhanced/south-building/baseline.log`) revealed the direct proximate mechanism:
1. **Severe Per-Image SIFT Feature Loss under CLAHE (clipLimit=2.0):**
   - On the unchanged images (100 images), SIFT feature counts are bit-for-bit identical (e.g., `P1180172.JPG`: 9,170 SIFT in baseline, 9,170 SIFT in enhanced).
   - On CLAHE-enhanced images, SIFT detections dropped dramatically:
     - `P1180167.JPG`: **14,784** features (baseline) $\rightarrow$ **9,269** features (enhanced) $\implies$ **-37.3% loss**
     - `P1180168.JPG`: **13,942** features (baseline) $\rightarrow$ **8,623** features (enhanced) $\implies$ **-38.2% loss**
2. **Mechanism of SIFT Suppression:**
   - CLAHE with `clipLimit=2.0` redistributes local histograms across $8 \times 8$ tiles. In architectural images containing subtle, continuous masonry gradients and shadows, local histogram equalization introduces subtle tile-boundary discontinuties and compresses subtle natural gradient structures.
   - SIFT's Difference-of-Gaussians (DoG) detector suppresses scale-space extrema whose principal curvature ratio exceeds the threshold ($\text{edge\_threshold}=10$), or whose contrast falls below the peak threshold. CLAHE alters these local DoG responses, leading to heavy keypoint rejection.
3. **Database Match Density Reduction:**
   - Baseline South Building database: `database.db` size = **245.8 MB**.
   - Enhanced South Building database: `database.db` size = **239.9 MB** (-5.9 MB).
   - Fewer matched features directly translated to fewer triangulated 3D points during bundle adjustment.

---

## 4. Work Incomplete at End of Phase 5
1. **Root-Cause Ablation on South Building:** Systematic isolation of CLAHE intensity (`clipLimit` = 2.0 vs 1.5 vs 1.2 vs 1.0) and tile geometry.
2. **Evaluation of Person Hall & Graham Hall:** These larger datasets remain queued and unexecuted due to compute duration and memory constraints.
3. **Optimized Selection Strategy:** Refining whether global contrast thresholding ($\tau = 60$) inappropriately flags architectural shadows that already possess abundant SIFT keypoints.

This transitions directly into **Phase 6: Enhancement Optimization and Ablation Study**.
