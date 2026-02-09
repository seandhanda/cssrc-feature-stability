# Accuracy Is Not Enough: Evaluating Feature Stability Across Resampled Training Sets

This repository contains code to reproduce the experiments and figures from the paper:

**Accuracy Is Not Enough: Evaluating Feature Stability Across Resampled Training Sets**  
Sean Dhanda, University of British Columbia

## Abstract
Machine learning models are most often evaluated using aggregate predictive performance metrics such as accuracy or loss. While these metrics quantify predictive correctness, many real-world applications also rely on model interpretability, where stable and consistent feature importance estimates are critical for trust, reproducibility, and downstream decision-making. Despite this reliance, the stability of learned feature attributions under repeated data resampling is rarely evaluated in standard machine learning workflows.

In this work, we conduct an empirical study of feature stability in supervised learning models under repeated train/test resampling. Using the Breast Cancer Wisconsin diagnostic dataset, we train logistic regression and random forest classifiers across 30 independent resampled splits. We jointly evaluate predictive accuracy and feature stability using mean test accuracy, pairwise Spearman rank correlation of feature importance rankings, and variance of importance magnitudes across resamples.

Our results show that although logistic regression achieves slightly higher mean accuracy, random forests exhibit substantially greater feature stability across resampled training sets. These findings demonstrate that models with comparable predictive performance can differ markedly in the robustness of their explanations, and that accuracy alone is insufficient when model explanations are required.

This study highlights the importance of complementing traditional performance metrics with stability-based diagnostics and provides a simple, fully reproducible framework for assessing feature robustness in applied machine learning settings.

## Requirements
Python 3.9+

Install dependencies:
```bash
pip install -r requirements.txt
