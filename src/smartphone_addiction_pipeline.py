"""
smartphone_addiction_pipeline.py
Group 7 | Data Science Term Project 2026

This file integrates all the main steps of the project into a single pipeline.
Call run_pipeline() to run everything end-to-end:

    1. Preprocessing  — drop leakage columns, encode categoricals, scale features
    2. EDA            — dataset description, class distribution, correlations
    3. Classification — Decision Tree, KNN, Logistic Regression (k-fold CV)
    4. Clustering     — K-Means, Agglomerative (44 experiments per partition)
    5. Evaluation     — weighted F1 (classification), balance score (clustering)
    6. Output         — Top-5 results + saved plots and CSV

Dataset
-------
Kaggle: Smartphone Usage and Addiction Analysis Dataset (7,500 rows)
https://www.kaggle.com/datasets/algozee/smartphone-usage-and-addiction-analysis-dataset

Required columns
    Numerical   : age, daily_screen_time_hours, social_media_hours,
                  gaming_hours, work_study_hours, sleep_hours,
                  notifications_per_day, app_opens_per_day,
                  weekend_screen_time
    Categorical : gender, stress_level, academic_work_impact
    Target      : addicted_label  (0 = Not Addicted, 1 = Addicted)
    Drop        : transaction_id, user_id, addiction_level (leakage)

Dependencies
------------
    pip install pandas numpy scikit-learn matplotlib seaborn

Usage
-----
    from smartphone_addiction_pipeline import run_pipeline

    results = run_pipeline("smartphone_addiction_analysis.csv")

    # Access results
    results["classification"]["top5"]      # DataFrame
    results["clustering"]["top5"]          # list of dicts
    results["classification"]["best"]      # dict with model + metrics

Parameters
----------
run_pipeline(csv_path, random_state=42, cv_splits=5,
             do_eda=True, run_classification=True, run_clustering=True,
             save_plots=True, output_dir="outputs")

Returns
-------
dict with keys "classification" and "clustering"

Notes
-----
- Algorithms restricted to course scope (KNN, Decision Tree,
  Logistic Regression, K-Means, Agglomerative Clustering).
- Manhattan distance for KNN and threshold tuning for Logistic Regression
  were added as extra experiments.
- addicted_label is not used as a clustering feature.
  It is only used later to compare cluster profiles.
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    StratifiedKFold,    # class-stratified k-fold (course scope)
    cross_val_predict,  # OOF prediction aggregation
    GridSearchCV,       # exhaustive hyperparameter search
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
# ColumnTransformer: applies different transformers to different column subsets
from sklearn.compose import ColumnTransformer

# Suppress non-critical library warnings to keep output readable.
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(csv_path: str) -> tuple:
    """
    Load dataset, drop leakage/ID columns, and encode categoricals.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (7500 × 14 after encoding).
    y : pd.Series
        Binary target vector (0 = Not Addicted, 1 = Addicted).
    df_raw : pd.DataFrame
        Original DataFrame (used for cluster profiling).

    Notes
    -----
    Encoding decisions:
      - gender           → One-Hot (nominal: no order)
      - stress_level     → Ordinal  (Low=0, Medium=1, High=2)
      - academic_work_impact → Binary (No=0, Yes=1)

    Drop decisions:
      - transaction_id, user_id : identifier columns, removed from features
      - addiction_level         : removed because of possible data leakage
    """
    df = pd.read_csv(csv_path)
    df_raw = df.copy()

    # ── Drop leakage & ID columns ─────────────────────────────────────────────
    # addiction_level is strongly related to addicted_label.
    # Including it risks data leakage because it may reveal target-related information.
    drop_cols = [c for c in ["transaction_id", "user_id", "addiction_level"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)

    # ── Ordinal encoding: stress_level ────────────────────────────────────────
    stress_map = {"Low": 0, "Medium": 1, "High": 2}
    df["stress_level"] = df["stress_level"].map(stress_map)

    # ── Binary encoding: academic_work_impact ─────────────────────────────────
    df["academic_work_impact"] = df["academic_work_impact"].map({"No": 0, "Yes": 1})

    # ── One-Hot encoding: gender ──────────────────────────────────────────────
    # drop_first=False: 'Other' category has ~2486 samples → keep all dummies
    df = pd.get_dummies(df, columns=["gender"], drop_first=False)

    X = df.drop(columns=["addicted_label"])
    y = df["addicted_label"]

    return X, y, df_raw


def build_preprocessor(scaler) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies ``scaler`` to numerical columns
    and passes encoded categorical columns through unchanged.

    Parameters
    ----------
    scaler : sklearn scaler instance or ``'passthrough'``
        Scaler to apply to the 9 numerical features.

    Returns
    -------
    ColumnTransformer

    Notes
    -----
    ColumnTransformer(transformers=[...], remainder='passthrough')
      - transformers : list of (name, transformer, columns)
      - remainder    : what to do with columns not listed
                       'passthrough' → keep as-is
    """
    num_cols = [
        "age", "daily_screen_time_hours", "social_media_hours",
        "gaming_hours", "work_study_hours", "sleep_hours",
        "notifications_per_day", "app_opens_per_day", "weekend_screen_time",
    ]
    return ColumnTransformer(
        transformers=[("num", scaler, num_cols)],
        remainder="passthrough",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. EDA
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(df_raw: pd.DataFrame,
            save_plots: bool = True,
            output_dir: str = "outputs") -> dict:
    """
    Print statistical summary and save EDA visualisations.

    Covers the Data Exploration step of the End-to-End process:
    dataset description (shape, dtypes, missing values), statistical
    summary (mean / std / min / max per numerical feature), class
    distribution check, and feature-target correlation analysis.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw DataFrame (original columns, before encoding or dropping).
    save_plots : bool, default True
        Whether to save PNG plots to ``output_dir``.
    output_dir : str, default "outputs"
        Directory for saved plots.

    Returns
    -------
    dict with keys:
        "shape"      : tuple (rows, cols)
        "dtypes"     : pd.Series
        "missing"    : pd.Series  — missing value count per column
        "describe"   : pd.DataFrame — numerical statistics (describe)
        "class_dist" : dict  — {0: count, 1: count}
        "top_corr"   : pd.Series — |Pearson r| with addicted_label, sorted desc

    Saved plots (when save_plots=True)
    -----------------------------------
    eda_class_dist.png  — bar chart of class imbalance (0 vs 1)
    eda_correlation.png — bar chart of |Pearson r| per numerical feature
    eda_boxplots.png    — box plots of top-4 correlated features by label
    """
    num_cols = [
        "age", "daily_screen_time_hours", "social_media_hours",
        "gaming_hours", "work_study_hours", "sleep_hours",
        "notifications_per_day", "app_opens_per_day", "weekend_screen_time",
    ]

    # ── Basic info ────────────────────────────────────────────────────────
    print(f"    Shape   : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    missing = df_raw.isnull().sum()
    print(f"    Missing : {missing[missing > 0].to_dict() or 'None'}")

    print("\n    ── Statistical Description (numerical features) ──")
    desc = df_raw[num_cols].describe().round(3)
    print(desc.to_string())

    # ── Class distribution ────────────────────────────────────────────────
    class_dist = df_raw["addicted_label"].value_counts().sort_index().to_dict()
    total = sum(class_dist.values())
    pct0 = class_dist.get(0, 0) / total * 100
    pct1 = class_dist.get(1, 0) / total * 100
    print(f"\n    Class distribution  →  "
          f"Not Addicted (0): {class_dist.get(0, 0):,} ({pct0:.1f}%)  |  "
          f"Addicted (1): {class_dist.get(1, 0):,} ({pct1:.1f}%)")

    # ── Correlation with target ───────────────────────────────────────────
    corr = (df_raw[num_cols + ["addicted_label"]]
            .corr()["addicted_label"]
            .drop("addicted_label")
            .abs()
            .sort_values(ascending=False))
    print("\n    ── |Pearson r| with addicted_label ──")
    print(corr.round(3).to_string())

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

        # ── Class distribution bar chart ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = [class_dist.get(0, 0), class_dist.get(1, 0)]
        bars = ax.bar(
            ["Not Addicted (0)", "Addicted (1)"], counts,
            color=["#6C63FF", "#FF6584"], edgecolor="white",
        )
        for bar, cnt, pct in zip(bars, counts, [pct0, pct1]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 40,
                    f"{cnt:,}\n({pct:.1f}%)", ha="center", fontsize=9)
        ax.set_title("Class Distribution (addicted_label)", fontsize=11)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "eda_class_dist.png"), dpi=150)
        plt.close(fig)

        # ── Correlation bar chart ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(corr)), corr.values, color="#43D9A3", edgecolor="white")
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels(corr.index, rotation=30, ha="right", fontsize=8)
        ax.set_title("|Pearson r| with addicted_label", fontsize=11)
        ax.set_ylabel("|r|")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "eda_correlation.png"), dpi=150)
        plt.close(fig)

        # ── Box plots: top-4 correlated features by class ─────────────────
        top4 = corr.index[:4].tolist()
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for ax, col in zip(axes, top4):
            df_raw.boxplot(column=col, by="addicted_label", ax=ax,
                           boxprops=dict(color="#6C63FF"),
                           medianprops=dict(color="#FF6584"),
                           whiskerprops=dict(color="#6C63FF"),
                           capprops=dict(color="#6C63FF"))
            ax.set_title(col, fontsize=8)
            ax.set_xlabel("addicted_label")
            ax.set_ylabel("hours" if "hours" in col else "count")
        fig.suptitle("Top-4 Correlated Features by Addiction Label", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "eda_boxplots.png"), dpi=150)
        plt.close(fig)

    return {
        "shape"      : df_raw.shape,
        "dtypes"     : df_raw.dtypes,
        "missing"    : missing,
        "describe"   : desc,
        "class_dist" : class_dist,
        "top_corr"   : corr,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_clf(pipeline, X, y, cv) -> dict:
    """
    Run cross_val_predict and return a metrics dict.

    cross_val_predict returns predictions for each validation fold.
    Each row is predicted once when it is not used for training.

    Parameters
    ----------
    pipeline : sklearn Pipeline
    X : pd.DataFrame
    y : pd.Series
    cv : StratifiedKFold

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix, y_pred
    """
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    return {
        "accuracy" : round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, average="weighted",
                                           zero_division=0), 4),
        "recall"   : round(recall_score(y, y_pred, average="weighted",
                                        zero_division=0), 4),
        "f1"       : round(f1_score(y, y_pred, average="weighted",
                                    zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "y_pred"   : y_pred,
    }


def run_decision_tree(X, y, cv, random_state=42) -> list:
    """
    Grid-search Decision Tree over criterion, max_depth,
    min_samples_split, min_samples_leaf, class_weight.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    cv : StratifiedKFold
    random_state : int

    Returns
    -------
    list of dicts, sorted by F1 descending.

    Notes
    -----
    GridSearchCV(estimator, param_grid, scoring, refit, cv)
      - scoring : metric used to rank candidates
      - refit   : refit the best estimator on full data after search
    DecisionTreeClassifier is relatively insensitive to feature scaling,
    but MinMaxScaler is applied per course requirements.
    """
    scaler = MinMaxScaler()
    preprocessor = build_preprocessor(scaler)

    param_grid = {
        "clf__criterion"        : ["gini", "entropy"],
        "clf__max_depth"        : [None] + list(range(1, 31)),
        "clf__min_samples_split": list(range(2, 11)),
        "clf__min_samples_leaf" : list(range(1, 11)),
        "clf__class_weight"     : [None, "balanced"],
    }

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", DecisionTreeClassifier(random_state=random_state)),
    ])

    gs = GridSearchCV(
        pipeline, param_grid,
        scoring="f1_weighted", refit=True,
        cv=cv, n_jobs=-1, verbose=0,
    )
    gs.fit(X, y)

    best_pipe = gs.best_estimator_
    metrics = _evaluate_clf(best_pipe, X, y, cv)

    return [{
        "model"       : "Decision Tree",
        "scaler"      : "MinMaxScaler",
        "params"      : gs.best_params_,
        **{k: v for k, v in metrics.items() if k != "y_pred"},
        "y_pred"      : metrics["y_pred"],
    }]


def run_knn(X, y, cv) -> list:
    """
    Evaluate KNN across 3 scalers × 5 K values (15 combinations).
    Also tests Manhattan and Chebyshev distance metrics
    as extra distance experiments.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    cv : StratifiedKFold

    Returns
    -------
    list of dicts, sorted by F1 descending.

    Notes
    -----
    Pipeline ensures the scaler is fit only on training folds,
    preventing data leakage during cross-validation.

    KNeighborsClassifier(n_neighbors, metric, p)
      n_neighbors : number of nearest neighbors to consider
      metric      : distance function
      p           : power for Minkowski distance (p=2 → Euclidean)
    """
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler"  : MinMaxScaler(),
        "RobustScaler"  : RobustScaler(),
    }
    k_values = [3, 5, 7, 9, 11]

    results = []
    for scaler_name, scaler in scalers.items():
        preprocessor = build_preprocessor(scaler)
        for k in k_values:
            pipe = Pipeline([
                ("pre", preprocessor),
                ("clf", KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2)),
            ])
            m = _evaluate_clf(pipe, X, y, cv)
            results.append({
                "model"  : "KNN",
                "scaler" : scaler_name,
                "params" : {"n_neighbors": k, "metric": "euclidean"},
                **{k2: v for k2, v in m.items() if k2 != "y_pred"},
                "y_pred" : m["y_pred"],
            })

    # ── Extra experiment: Manhattan and Chebyshev distance ───────────────────
    # Added to compare other distance metrics with Euclidean distance.
    preprocessor_robust = build_preprocessor(RobustScaler())
    for metric in ["manhattan", "chebyshev"]:
        pipe = Pipeline([
            ("pre", preprocessor_robust),
            ("clf", KNeighborsClassifier(n_neighbors=11, metric=metric)),
        ])
        m = _evaluate_clf(pipe, X, y, cv)
        results.append({
            "model"  : "KNN",
            "scaler" : "RobustScaler",
            "params" : {"n_neighbors": 11, "metric": metric,
                        "note": "extra experiment"},
            **{k2: v for k2, v in m.items() if k2 != "y_pred"},
            "y_pred" : m["y_pred"],
        })

    results.sort(key=lambda x: (x["f1"], x["accuracy"]), reverse=True)
    return results


def run_logistic_regression(X, y, cv) -> list:
    """
    Evaluate Logistic Regression across 4 scalers × 5 C values
    × 2 class_weight settings, then apply threshold tuning on Top-10.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    cv : StratifiedKFold

    Returns
    -------
    list of dicts, sorted by F1 descending (after threshold tuning).

    Notes
    -----
    LogisticRegression(C, penalty, class_weight, max_iter)
      C            : inverse regularisation strength (larger = weaker)
      penalty      : regularisation type ('l2' = Ridge)
      class_weight : 'balanced' up-weights minority class during training
      max_iter     : solver iteration limit

    Threshold tuning (extra experiment):
      Default threshold is 0.5. Since class 1 is more common in this dataset,
      we also tested thresholds around 0.5 and selected the one with the best weighted F1.
    """
    scalers = {
        "None"         : "passthrough",
        "StandardScaler": StandardScaler(),
        "MinMaxScaler" : MinMaxScaler(),
        "RobustScaler" : RobustScaler(),
    }
    C_values      = [0.01, 0.1, 1, 10, 100]
    class_weights = [None, "balanced"]

    results = []
    for scaler_name, scaler in scalers.items():
        preprocessor = build_preprocessor(scaler)
        for C in C_values:
            for cw in class_weights:
                pipe = Pipeline([
                    ("pre", preprocessor),
                    ("clf", LogisticRegression(
                        C=C, penalty="l2",
                        class_weight=cw,
                        max_iter=1000,
                        random_state=42,
                    )),
                ])
                m = _evaluate_clf(pipe, X, y, cv)
                results.append({
                    "model"    : "Logistic Regression",
                    "scaler"   : scaler_name,
                    "params"   : {"C": C, "penalty": "l2",
                                  "class_weight": cw, "threshold": 0.5},
                    **{k2: v for k2, v in m.items() if k2 != "y_pred"},
                    "y_pred"   : m["y_pred"],
                    "_pipe"    : pipe,
                })

    results.sort(key=lambda x: (x["f1"], x["accuracy"]), reverse=True)
    top10 = results[:10]

    # ── Extra experiment: threshold tuning on Top-10 ─────────────────────────
    # predict_proba returns P(class=1); default cutoff = 0.5.
    # We sweep thresholds in [0.45, 0.50] at 0.001 intervals
    # and pick the one that maximises weighted F1.
    tuned = []
    for entry in top10:
        pipe = entry["_pipe"]
        y_proba = cross_val_predict(
            pipe, X, y, cv=cv, method="predict_proba"
        )[:, 1]
        best_f1, best_thr, best_pred = 0, 0.5, None
        for thr in np.arange(0.45, 0.501, 0.001):
            y_t = (y_proba >= thr).astype(int)
            f = f1_score(y, y_t, average="weighted", zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, round(thr, 3)
                best_pred = y_t
        new_entry = dict(entry)
        new_entry["params"] = {**entry["params"], "threshold": best_thr}
        new_entry["f1"]        = round(best_f1, 4)
        new_entry["accuracy"]  = round(accuracy_score(y, best_pred), 4)
        new_entry["precision"] = round(precision_score(y, best_pred,
                                        average="weighted", zero_division=0), 4)
        new_entry["recall"]    = round(recall_score(y, best_pred,
                                        average="weighted", zero_division=0), 4)
        new_entry["confusion_matrix"] = confusion_matrix(y, best_pred)
        new_entry["y_pred"]    = best_pred
        tuned.append(new_entry)

    tuned.sort(key=lambda x: (x["f1"], x["accuracy"]), reverse=True)
    # remove internal _pipe key before returning
    for e in tuned:
        e.pop("_pipe", None)
    return tuned


def _run_classification(X, y, random_state=42, cv_splits=5) -> dict:
    """
    Run all three classification algorithms and return Top-5.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    random_state : int
    cv_splits : int
        Number of folds for StratifiedKFold.

    Returns
    -------
    dict with keys:
        "all"  : list of all result dicts, sorted by F1
        "top5" : pd.DataFrame of top 5 results
        "best" : dict of the single best result
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                         random_state=random_state)

    print("  [Classification] Running Decision Tree (GridSearch)...")
    dt_results = run_decision_tree(X, y, cv, random_state)
    # Decision Tree is represented here by its single best GridSearch result.
    # The separate score-based report lists the top parameter combinations individually.

    print("  [Classification] Running KNN (15 combinations + extra distance experiments)...")
    knn_results = run_knn(X, y, cv)

    print("  [Classification] Running Logistic Regression (+ threshold tuning)...")
    lr_results = run_logistic_regression(X, y, cv)

    all_results = dt_results + knn_results + lr_results
    all_results.sort(key=lambda x: (x["f1"], x["accuracy"]), reverse=True)

    top5_rows = []
    for i, r in enumerate(all_results[:5], 1):
        top5_rows.append({
            "Rank"               : i,
            "Model"              : r["model"],
            "Scaler"             : r["scaler"],
            "Key Params"         : str(r["params"]),
            "Accuracy"           : r["accuracy"],
            "Precision (weighted)": r["precision"],
            "Recall (weighted)"  : r["recall"],
            "F1 (weighted)"      : r["f1"],
        })

    return {
        "all"  : all_results,
        "top5" : pd.DataFrame(top5_rows),
        "best" : all_results[0],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_profile(feature_df, target, labels) -> pd.DataFrame:
    """
    Compute per-cluster mean profile and addicted_label ratio.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Encoded feature DataFrame used for clustering (all columns numeric).
        Must NOT contain addicted_label.
    target : pd.Series
        addicted_label column from the original DataFrame (used post-hoc only).
    labels : array-like
        Cluster assignment for each row.

    Returns
    -------
    pd.DataFrame with cluster means + size + addicted ratio.

    Notes
    -----
    feature_df must be fully numeric (encoding already applied) so that
    groupby().mean() works correctly for all columns including stress_level
    and academic_work_impact in Partition B.
    """
    tmp = feature_df.copy()
    tmp["addicted_label"] = target.values
    tmp["cluster"] = labels
    feat_cols = feature_df.columns.tolist()
    profile = tmp.groupby("cluster")[feat_cols].mean().round(3)
    profile["size"]           = tmp.groupby("cluster").size()
    profile["addicted_ratio"] = tmp.groupby("cluster")["addicted_label"].mean().round(3)
    return profile


def run_clustering_part(df_raw, feature_cols, encode_map=None,
                        random_state=42) -> list:
    """
    Run 44 clustering experiments for one feature set.

    Experiment space
    ----------------
    K-Means       : 3 scalers × 4 k values            = 12
    Agglomerative : 2 scalers × 2 linkages × 2 metrics × 4 k = 32
    Total         : 44

    Parameters
    ----------
    df_raw : pd.DataFrame
        Original DataFrame.
    feature_cols : list of str
        Feature columns to cluster on (must NOT include addicted_label).
    encode_map : dict, optional
        ``{column_name: {value: int}}`` for ordinal/binary encoding.
        Applied before scaling.
    random_state : int

    Returns
    -------
    list of dicts sorted by cluster-size balance (descending).

    Notes
    -----
    addicted_label is not used as a clustering feature.
    It is only used after clustering to check the addicted ratio per cluster.

    Cluster size balance = min_cluster_size / max_cluster_size.
    A value close to 1.0 indicates evenly distributed clusters.
    """
    data = df_raw[feature_cols].copy()

    if encode_map:
        for col, mapping in encode_map.items():
            data[col] = data[col].map(mapping)

    scalers = {
        "standard": StandardScaler(),
        "minmax"  : MinMaxScaler(),
        "robust"  : RobustScaler(),
    }
    k_values = [2, 3, 4, 5]

    results = []

    # ── K-Means (3 scalers × 4 k) ─────────────────────────────────────────
    for sname, scaler in scalers.items():
        X_scaled = scaler.fit_transform(data)
        for k in k_values:
            model = KMeans(n_clusters=k, random_state=random_state,
                           n_init=10, max_iter=300)
            labels = model.fit_predict(X_scaled)
            sizes = pd.Series(labels).value_counts()
            balance = round(sizes.min() / sizes.max(), 3)
            profile = _cluster_profile(data, df_raw["addicted_label"], labels)
            results.append({
                "experiment"  : f"{sname}_kmeans_euclidean_k{k}",
                "model"       : "K-Means",
                "scaler"      : sname,
                "linkage"     : None,
                "metric"      : "euclidean",
                "k"           : k,
                "balance"     : balance,
                "cluster_sizes": sizes.sort_index().to_dict(),
                "profile"     : profile,
                "labels"      : labels,
            })

    # ── Agglomerative (2 scalers × 2 linkages × 2 metrics × 4 k) ────────
    for sname in ["standard", "minmax"]:
        X_scaled = scalers[sname].fit_transform(data)
        for linkage in ["complete", "average"]:
            for metric in ["euclidean", "manhattan"]:
                for k in k_values:
                    model = AgglomerativeClustering(
                        n_clusters=k, linkage=linkage, metric=metric
                    )
                    labels = model.fit_predict(X_scaled)
                    sizes = pd.Series(labels).value_counts()
                    balance = round(sizes.min() / sizes.max(), 3)
                    profile = _cluster_profile(data, df_raw["addicted_label"], labels)
                    results.append({
                        "experiment"  : f"{sname}_agglomerative_{linkage}_{metric}_k{k}",
                        "model"       : "Agglomerative",
                        "scaler"      : sname,
                        "linkage"     : linkage,
                        "metric"      : metric,
                        "k"           : k,
                        "balance"     : balance,
                        "cluster_sizes": sizes.sort_index().to_dict(),
                        "profile"     : profile,
                        "labels"      : labels,
                    })

    results.sort(key=lambda x: x["balance"], reverse=True)
    return results


def _run_clustering(df_raw, random_state=42) -> dict:
    """
    Run clustering for both feature partitions and return Top-5.

    Partition A — Usage time / App type
        daily_screen_time_hours, social_media_hours, gaming_hours,
        work_study_hours, weekend_screen_time

    Partition B — Sleep / Notifications / Stress
        sleep_hours, notifications_per_day, app_opens_per_day,
        stress_level (ordinal), academic_work_impact (binary)

    Parameters
    ----------
    df_raw : pd.DataFrame
    random_state : int

    Returns
    -------
    dict with keys:
        "usage"       : list of 44 results (partition A)
        "sleep_stress": list of 44 results (partition B)
        "top5"        : list of 5 selected clustering results (usage partition)
    """
    # ── Partition A ───────────────────────────────────────────────────────
    print("  [Clustering] Running usage-time / app-type partition (44 exp)...")
    usage_features = [
        "daily_screen_time_hours", "social_media_hours",
        "gaming_hours", "work_study_hours", "weekend_screen_time",
    ]
    usage_results = run_clustering_part(
        df_raw, usage_features, random_state=random_state
    )

    # ── Partition B ───────────────────────────────────────────────────────
    print("  [Clustering] Running sleep / notifications / stress partition (44 exp)...")
    sleep_features = [
        "sleep_hours", "notifications_per_day", "app_opens_per_day",
        "stress_level", "academic_work_impact",
    ]
    encode_map = {
        "stress_level"        : {"Low": 0, "Medium": 1, "High": 2},
        "academic_work_impact": {"No": 0, "Yes": 1},
    }
    sleep_results = run_clustering_part(
        df_raw, sleep_features,
        encode_map=encode_map,
        random_state=random_state,
    )

    # ── Top-5 from usage partition ────────────────────────────────────────
    # Manually selected to match the report's top-5 for the usage/app-type partition.
    # Criteria: cluster-size balance, interpretability of usage patterns,
    # and method diversity (K-Means + Agglomerative both included).
    # (partition, experiment_name) tuples to avoid key collision:
    # usage_results and sleep_results share the same experiment names
    # (e.g. "minmax_kmeans_euclidean_k4" exists in both partitions).
    paper_top5_names = [
        ("usage", "minmax_kmeans_euclidean_k4"),                    # representative: balanced usage-type split
        ("usage", "standard_kmeans_euclidean_k3"),                  # simpler 3-cluster comparison
        ("usage", "robust_kmeans_euclidean_k4"),                    # RobustScaler comparison
        ("usage", "minmax_agglomerative_complete_manhattan_k4"),    # Agglomerative method comparison
        ("usage", "standard_agglomerative_average_manhattan_k5"),   # finer-grained view
    ]
    result_map = {("usage", r["experiment"]): r for r in usage_results}
    result_map.update({("sleep_stress", r["experiment"]): r for r in sleep_results})

    top5 = []
    for part, name in paper_top5_names:
        if (part, name) in result_map:
            r = result_map[(part, name)]
            entry = {**r, "partition": part}
            entry.pop("labels", None)
            entry.pop("profile", None)
            top5.append(entry)

    return {
        "usage"        : usage_results,
        "sleep_stress" : sleep_results,
        "top5"         : top5,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm: np.ndarray, title: str, path: str) -> None:
    """
    Save a confusion matrix heatmap to disk.

    Parameters
    ----------
    cm : np.ndarray
        2×2 confusion matrix.
    title : str
        Plot title.
    path : str
        Output file path (PNG).
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Addicted (0)", "Addicted (1)"],
                yticklabels=["Not Addicted (0)", "Addicted (1)"])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cluster_scatter(X_features, labels, title, path) -> None:
    """
    PCA-reduced 2D scatter plot of cluster assignments.

    Parameters
    ----------
    X_features : array-like
        Original feature matrix (before PCA).
    labels : array-like
        Cluster labels.
    title : str
    path : str
        Output file path (PNG).

    Notes
    -----
    PCA(n_components=2) reduces high-dimensional data to 2D for
    visualisation.  The scatter plot shows approximate cluster
    separation; final interpretation relies on the profile table.
    """
    pca = PCA(n_components=2, random_state=42)
    pts = pca.fit_transform(StandardScaler().fit_transform(X_features))
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(pts[:, 0], pts[:, 1], c=labels,
                         cmap="tab10", s=10, alpha=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_classification_comparison(clf_results, path) -> None:
    """
    Bar chart comparing F1-scores of all classification experiments.

    Parameters
    ----------
    clf_results : list of dicts
    path : str
    """
    labels = [f"{r['model'][:3]}-{r['scaler'][:3]}" for r in clf_results[:15]]
    f1s    = [r["f1"] for r in clf_results[:15]]
    colors = ["#6C63FF" if r["model"] == "Decision Tree"
              else "#43D9A3" if r["model"] == "KNN"
              else "#FF6584"
              for r in clf_results[:15]]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(labels)), f1s, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1-score (weighted)")
    ax.set_title("Classification F1-score Comparison (Top 15)", fontsize=12)
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6. TOP-LEVEL FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    csv_path: str,
    random_state: int = 42,
    cv_splits: int = 5,
    do_eda: bool = True,
    run_classification: bool = True,
    run_clustering: bool = True,
    save_plots: bool = True,
    output_dir: str = "outputs",
) -> dict:
    """
    End-to-end pipeline for smartphone addiction analysis.

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV file.
    random_state : int, default 42
        Seed for reproducibility (applied to all stochastic components).
    cv_splits : int, default 5
        Number of folds for StratifiedKFold cross-validation.
    do_eda : bool, default True
        Whether to print statistical summary and save EDA plots.
    run_classification : bool, default True
        Whether to run Decision Tree, KNN, and Logistic Regression.
    run_clustering : bool, default True
        Whether to run K-Means and Agglomerative Clustering.
    save_plots : bool, default True
        Whether to save PNG visualisations to ``output_dir``.
    output_dir : str, default "outputs"
        Directory for saved plots and CSV summaries.

    Returns
    -------
    dict
        {
            "eda": {
                "shape", "dtypes", "missing", "describe",
                "class_dist", "top_corr"
            },
            "classification": {
                "all"  : list of all result dicts,
                "top5" : pd.DataFrame,
                "best" : dict
            },
            "clustering": {
                "usage"        : list (44 results, partition A),
                "sleep_stress" : list (44 results, partition B),
                "top5"         : list of 5 dicts
            }
        }

    Examples
    --------
    >>> results = run_pipeline("smartphone_addiction_analysis.csv")
    >>> print(results["classification"]["top5"])
    >>> print(results["clustering"]["top5"][0]["experiment"])

    Only run clustering:
    >>> results = run_pipeline(
    ...     "smartphone_addiction_analysis.csv",
    ...     run_classification=False,
    ...     run_clustering=True,
    ... )
    """
    os.makedirs(output_dir, exist_ok=True)
    output = {}

    print("=" * 60)
    print("Smartphone Addiction Analysis Pipeline")
    print("=" * 60)

    # ── Step 1. Load & Preprocess ─────────────────────────────────────────
    print("\n[1] Loading and preprocessing data...")
    X, y, df_raw = load_and_preprocess(csv_path)
    print(f"    X shape : {X.shape}  (samples × features)")
    print(f"    y dist  : {y.value_counts().to_dict()}")

    # ── Step 2. EDA ───────────────────────────────────────────────────────
    if do_eda:
        print("\n[2] Running Exploratory Data Analysis...")
        output["eda"] = run_eda(df_raw, save_plots=save_plots,
                                output_dir=output_dir)

    # ── Step 3. Classification ────────────────────────────────────────────
    if run_classification:
        print("\n[3] Running Classification experiments...")
        clf_out = _run_classification(X, y, random_state, cv_splits)
        output["classification"] = clf_out

        print("\n    ── Top 5 Classification Results ──")
        print(clf_out["top5"].to_string(index=False))

        best = clf_out["best"]
        print(f"\n    ★ Best: {best['model']} | {best['scaler']}")
        print(f"      F1={best['f1']}  Accuracy={best['accuracy']}")

        if save_plots:
            # confusion matrix for best model
            plot_confusion_matrix(
                best["confusion_matrix"],
                title=f"Best Model: {best['model']} ({best['scaler']})",
                path=os.path.join(output_dir, "confusion_matrix_best.png"),
            )
            # F1 comparison bar chart
            plot_classification_comparison(
                clf_out["all"],
                path=os.path.join(output_dir, "f1_comparison.png"),
            )
            # top-5 confusion matrices
            for i, r in enumerate(clf_out["all"][:5], 1):
                plot_confusion_matrix(
                    r["confusion_matrix"],
                    title=f"Top{i}: {r['model']} ({r['scaler']})",
                    path=os.path.join(output_dir, f"cm_top{i}.png"),
                )

        # Save top-5 CSV
        clf_out["top5"].to_csv(
            os.path.join(output_dir, "classification_top5.csv"), index=False
        )

    # ── Step 4. Clustering ────────────────────────────────────────────────
    if run_clustering:
        print("\n[4] Running Clustering experiments...")
        clust_out = _run_clustering(df_raw, random_state)
        output["clustering"] = clust_out

        print("\n    ── Top 5 Clustering Results ──")
        for i, r in enumerate(clust_out["top5"], 1):
            print(f"    {i}. {r['experiment']:50s}  balance={r['balance']}")

        if save_plots:
            for i, r in enumerate(clust_out["top5"], 1):
                part = r["partition"]
                feat = (["daily_screen_time_hours", "social_media_hours",
                          "gaming_hours", "work_study_hours", "weekend_screen_time"]
                        if part == "usage" else
                        ["sleep_hours", "notifications_per_day", "app_opens_per_day",
                         "stress_level", "academic_work_impact"])
                data = df_raw[feat].copy()
                if part == "sleep_stress":
                    data["stress_level"] = data["stress_level"].map(
                        {"Low": 0, "Medium": 1, "High": 2})
                    data["academic_work_impact"] = data["academic_work_impact"].map(
                        {"No": 0, "Yes": 1})
                # retrieve labels from full results list
                full = (clust_out["usage"] if part == "usage"
                        else clust_out["sleep_stress"])
                labels = next(
                    (x["labels"] for x in full
                     if x["experiment"] == r["experiment"]), None
                )
                if labels is not None:
                    plot_cluster_scatter(
                        data.values, labels,
                        title=f"Top{i}: {r['experiment']}",
                        path=os.path.join(output_dir, f"cluster_scatter_top{i}.png"),
                    )

    print("\n" + "=" * 60)
    print(f"Pipeline complete. Outputs saved to: {output_dir}/")
    print("=" * 60)
    return output


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "smartphone_addiction_analysis.csv"
    run_pipeline(csv)
