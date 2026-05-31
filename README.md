# Smartphone Usage & Addiction Analysis

**Data Science Term Project 2026 — Group 7**

End-to-end ML pipeline for predicting smartphone addiction and clustering user behavior patterns.

---

## Overview

| Item | Detail |
|---|---|
| Dataset | [Smartphone Usage and Addiction Analysis](https://www.kaggle.com/datasets/algozee/smartphone-usage-and-addiction-analysis-dataset) (Kaggle) |
| Records | 7,500 users |
| Features after preprocessing | 14 features |
| Target | `addicted_label` (0 = Not Addicted, 1 = Addicted) |
| Class distribution | 0: 29.2% / 1: 70.8% (imbalanced) |
| Primary metric | Weighted F1-score |

---

## Pipeline Structure

```
run_pipeline(csv_path)
│
├── [1] Preprocessing
│     ├── Drop: transaction_id, user_id, addiction_level (leakage)
│     ├── Ordinal encoding: stress_level (Low=0, Medium=1, High=2)
│     ├── Binary encoding:  academic_work_impact (No=0, Yes=1)
│     └── One-Hot encoding: gender (Female / Male / Other)
│
├── [2] EDA
│     ├── run_eda() — shape, missing values, statistical description
│     ├── Class distribution (0 vs 1)
│     └── |Pearson r| correlation with addicted_label
│
├── [3] Classification
│     ├── Decision Tree  — GridSearchCV (~1,240 combinations)
│     ├── KNN            — 3 scalers × 5 K values + Manhattan/Chebyshev (extra)
│     └── Logistic Regression — 4 scalers × 5 C values + threshold tuning (extra)
│
├── [4] Clustering
│     ├── Partition A: usage time / app type (5 features, 44 experiments)
│     └── Partition B: sleep / notifications / stress (5 features, 44 experiments)
│
└── [5] Output
      ├── eda_class_dist.png
      ├── eda_correlation.png
      ├── eda_boxplots.png
      ├── classification_top5.csv
      ├── confusion_matrix_best.png
      ├── f1_comparison.png
      └── cluster_scatter_top1~5.png
```

---

## Results

### Model-wise Best Classification Results

Each model is represented by its best result. The integrated pipeline outputs one best result per model type.

| Rank | Model | Scaler | Key Params | Accuracy | F1 (weighted) |
|---|---|---|---|---|---|
| 1 | Decision Tree | MinMaxScaler | gini, depth=2, balanced | 0.9380 | **0.9395** |
| 2 | KNN | RobustScaler | k=11, Manhattan† | 0.9017 | **0.9007** |
| 3 | Logistic Regression | None | C=10, L2, thr=0.488 | 0.8936 | **0.8924** |

† Manhattan distance was added as an extra distance-metric experiment.

### Clustering — Best Results per Partition

| Partition | Algorithm | Scaler | k | Cluster Sizes | Key Pattern |
|---|---|---|---|---|---|
| Usage / App type | K-Means (Euclidean) | MinMaxScaler | 4 | 1835–1908 (balance 0.962) | High/Low usage × Gaming/Work |
| Sleep / Notifications / Stress | K-Means (Euclidean) | RobustScaler | 4 | 1831–1925 (balance 0.951) | Sleep shortage/sufficient × High notifications/App launches |

### Clustering — Usage / App type Top 5 Experiments

Selected based on cluster-size balance, profile interpretability, and method diversity.

| Rank | Experiment | Model | k | Balance |
|---|---|---|---|---|
| 1 | `minmax_kmeans_euclidean_k4` | K-Means | 4 | 0.962 |
| 2 | `standard_kmeans_euclidean_k3` | K-Means | 3 | 0.634 |
| 3 | `robust_kmeans_euclidean_k4` | K-Means | 4 | 0.929 |
| 4 | `minmax_agglomerative_complete_manhattan_k4` | Agglomerative | 4 | 0.770 |
| 5 | `standard_agglomerative_average_manhattan_k5` | Agglomerative | 5 | 0.671 |

Balance = min cluster size / max cluster size. Closer to 1.0 means more evenly distributed clusters.

---

## Key Insights

**1. Two features dominate the Decision Tree classification**

In the best Decision Tree model with `max_depth=2`, `daily_screen_time_hours` (importance 0.571) and `social_media_hours` (0.429) were used as the main split features.

**2. EDA, model, and clustering all point to usage-related features**

EDA, Decision Tree feature importance, and Partition A clustering all showed that usage-time related variables were important in explaining smartphone addiction patterns.

**3. Sleep and notifications were less important in the best Decision Tree, but useful for clustering**

In the best Decision Tree model, `sleep_hours` and `notifications_per_day` had 0.000 feature importance. However, Clustering Partition B still revealed different lifestyle patterns based on sleep, notifications, app opens, and stress level.

---

## Installation

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Dataset

The dataset is not included in this repository. Download it from Kaggle and place it as `data/smartphone_addiction_analysis.csv`.

> [Smartphone Usage and Addiction Analysis Dataset](https://www.kaggle.com/datasets/algozee/smartphone-usage-and-addiction-analysis-dataset)

---

## Usage

```python
from src.smartphone_addiction_pipeline import run_pipeline

# Full pipeline (preprocessing → EDA → classification → clustering)
results = run_pipeline("data/smartphone_addiction_analysis.csv")

# Skip EDA
results = run_pipeline(
    "data/smartphone_addiction_analysis.csv",
    do_eda=False,
)

# Classification only, no plots
results = run_pipeline(
    "data/smartphone_addiction_analysis.csv",
    run_clustering=False,
    save_plots=False,
)

# Access results
print(results["classification"]["top5"])
print(results["clustering"]["top5"][0]["experiment"])
print(results["eda"]["top_corr"])
```

### CLI

```bash
python src/smartphone_addiction_pipeline.py data/smartphone_addiction_analysis.csv
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `csv_path` | str | — | Path to dataset CSV |
| `random_state` | int | 42 | Reproducibility seed |
| `cv_splits` | int | 5 | StratifiedKFold folds |
| `do_eda` | bool | True | Print stats & save EDA plots |
| `run_classification` | bool | True | Run DT / KNN / LR |
| `run_clustering` | bool | True | Run K-Means / Agglomerative |
| `save_plots` | bool | True | Save PNG outputs |
| `output_dir` | str | `"outputs"` | Output directory |

---

## Functions

| Function | Description |
|---|---|
| `run_pipeline()` | Top-level entry point. Runs all steps in order. |
| `load_and_preprocess()` | Loads CSV, drops leakage columns, encodes categoricals. |
| `build_preprocessor()` | Builds a ColumnTransformer that scales numerical columns. |
| `run_eda()` | Prints dataset summary and saves EDA plots. |
| `run_decision_tree()` | GridSearchCV over criterion, depth, leaf size, class weight. |
| `run_knn()` | Evaluates KNN across 3 scalers × 5 K values + extra distance experiments. |
| `run_logistic_regression()` | Evaluates LR across scalers/C values with threshold tuning. |
| `run_clustering_part()` | Runs 44 clustering experiments for one feature partition. |
| `plot_confusion_matrix()` | Saves a confusion matrix heatmap as PNG. |
| `plot_cluster_scatter()` | Saves a PCA-reduced 2D cluster scatter plot as PNG. |
| `plot_classification_comparison()` | Saves a F1-score bar chart for top classification results. |

---

## Algorithm Scope

| Component | In scope | Extra experiments |
|---|---|---|
| Scaling | StandardScaler, MinMaxScaler, RobustScaler | — |
| Classification | Decision Tree, KNN, Logistic Regression | Manhattan distance, threshold tuning |
| Clustering | K-Means, Agglomerative | — |
| Evaluation | k-fold CV, weighted F1, Accuracy, Precision, Recall, Confusion Matrix | — |

---

## File Structure

```
repo/
├── src/
│   └── smartphone_addiction_pipeline.py   # main pipeline
├── data/
│   └── smartphone_addiction_analysis.csv  # download from Kaggle (not included)
├── outputs/                               # generated plots and CSVs (auto-created)
├── requirements.txt
└── README.md
```

---

## Team

Group 7 — Data Science 2026
