# Chest X-ray Pneumonia Detection with Custom Image Enhancement

Investigating whether a custom image enhancement pipeline improves deep learning-based pneumonia classification on chest X-rays.

## Overview

X-ray images often suffer from low contrast and poor definition, making it difficult to identify pathological features such as lung opacities associated with pneumonia. This project implements and evaluates a 30-step image enhancement pipeline built on top of **Adaptive Gradient Domain Guided Image Filtering (AGDGIF)**, then measures its impact on classification performance using EfficientNetB0 transfer learning.

## Project Structure

```
chest-xray-enhancement/
├── notebooks/
│   ├── filter_pipeline.ipynb        # AGDGIF + 30-step enhancement pipeline
│   └── classification.ipynb         # EfficientNetB0 original vs enhanced comparison
├── figures/
│   ├── agdgif_base_detail_decomposition.png
│   ├── pipeline_intermediate_stages.png
│   ├── original_vs_enhanced_samples.png
│   ├── quality_metrics_comparison.png
│   └── intensity_distribution_analysis.png
├── .gitignore
└── README.md
```

## Pipeline

The enhancement pipeline consists of 30 sequential steps applied to each X-ray image:

1. Intensity rescale
2. Gaussian denoise
3. AGDGIF base layer extraction
4. Adaptive β detail boost
5. CLAHE
6. Bilateral filter
7. Gamma correction
8. Laplacian edge extraction
9. Edge-weighted fusion
10. Median filter
11. Local std deviation map
12. Texture-adaptive gain
13. Unsharp mask
14. Top-hat transform
15. Black-hat transform
16. Morphological feature fusion
17. Sobel gradient magnitude
18. Gradient-weighted blend
19. Percentile clipping
20. Log transform
21. Inverse log correction
22. Partial local mean correction
23. Sigmoid contrast stretch
24. Histogram matching
25. Gentle blur
26. Detail residual recombination
27. Min-max normalization
28. Power-law fine tune
29. Weighted average with original
30. Final quantization

## Quality Metrics

Following Li et al. (2022), five objective metrics are used to evaluate enhancement quality:

| Metric | Original | Enhanced |
|--------|----------|----------|
| AG (Average Gradient) | 6.64 | 6.18 |
| H (Information Entropy) | 7.36 | 7.79 |
| ALC (Average Local Contrast) | 0.43 | 0.55 |
| SF (Spatial Frequency) | 9.84 | 10.11 |
| MG (Mean Gradient) | 33.79 | 41.58 |

## Dataset

[Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Download and place under `data/`:
```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Requirements

```
numpy
opencv-python
scikit-image
scipy
matplotlib
pandas
tqdm
tensorflow
```

## Reference

Li, L.; Lv, M.; Ma, H.; Jia, Z.; Yang, X.; Yang, W. *X-ray Image Enhancement Based on Adaptive Gradient Domain Guided Image Filtering.* Appl. Sci. 2022, 12, 10453. https://doi.org/10.3390/app122010453
