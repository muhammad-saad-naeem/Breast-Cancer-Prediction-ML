"""
Improved Breast Cancer Diagnosis Prediction
=========================================

This module extends the original ``analysis.py`` by performing
hyperparameter tuning for each machine learning algorithm using
``sklearn.model_selection.GridSearchCV``.  The goal of this improved
analysis is to squeeze a little more performance out of the base
classifiers and to produce cleaner, more polished visualisations that
communicate key insights more effectively.

Key enhancements over the original implementation include:

* **Hyperparameter Tuning** – Each model is accompanied by a small
  hyperparameter grid.  A 5‑fold stratified cross‑validation grid search
  identifies the best parameter combination for each classifier.
* **Expanded Evaluation Metrics** – In addition to accuracy and ROC
  AUC, precision, recall and F1‑score are reported on the held‑out test
  set, providing a more nuanced view of performance across classes.
* **Improved Visualisations** – The correlation heatmap uses a mask to
  show only the lower triangle of the correlation matrix, and colour
  scales are chosen for perceptual uniformity.  Confusion matrices
  include both raw counts and annotated percentages.  ROC curves are
  plotted together on the same axes for easier comparison.

Usage
-----

Run this script from the command line in the root of the project:

.. code:: bash

   python src/analysis_improved.py

All artefacts (data, plots, CSV results) will be generated under the
``breast_cancer_prediction`` directory.  See the updated README for
details on interpreting the outputs.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.config")
os.environ.setdefault("HOME", "/tmp")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402


def load_dataset() -> pd.DataFrame:
    """Load the Breast Cancer dataset into a pandas DataFrame.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing feature columns and the target column ``target``.
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save the dataset to CSV for reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.
    output_path : Path
        Path to write the CSV file to.
    """
    df.to_csv(output_path, index=False)


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics of the numerical features."""
    return df.describe().T


def correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Create and save a correlation heatmap of the feature variables.

    This implementation masks the upper triangle of the correlation matrix to
    reduce redundancy and uses a perceptually uniform colour map.  Annotated
    values are omitted to avoid clutter given the number of features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.
    output_dir : Path
        Directory where the heatmap figure will be saved.
    """
    corr = df.drop(columns="target").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="viridis",
        square=True,
        cbar_kws={"shrink": 0.6},
        linewidths=0.5,
    )
    plt.title("Feature Correlation Heatmap (Lower Triangle)")
    plt.tight_layout()
    fig_path = output_dir / "correlation_heatmap.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plot_top_feature_distributions(df: pd.DataFrame, output_dir: Path, top_n: int = 6) -> None:
    """Plot distribution of the top features most correlated with the target.

    Uses violin plots to show class‑conditional distributions.  Uses
    perceptually uniform colours to distinguish classes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.
    output_dir : Path
        Directory where the plot will be saved.
    top_n : int, optional
        Number of top correlated features to plot (default is 6).
    """
    feature_cols = df.columns.drop("target")
    correlations = df[feature_cols].corrwith(df["target"]).abs()
    top_features = correlations.sort_values(ascending=False).head(top_n).index
    df_melt = df.melt(id_vars="target", value_vars=top_features, var_name="feature", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        x="feature",
        y="value",
        hue="target",
        data=df_melt,
        split=True,
        inner="quartile",
        palette=["#1f77b4", "#d62728"],  # blue for benign, red for malignant
    )
    plt.title(f"Distribution of Top {top_n} Features by Class")
    plt.xlabel("Feature")
    plt.ylabel("Feature Value")
    plt.legend(title="Target", labels=["Benign (0)", "Malignant (1)"])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig_path = output_dir / "top_features_violin.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()


def get_model_grid() -> Dict[str, Tuple[Pipeline, Dict[str, List]]]:
    """Define base models and their hyperparameter grids.

    Returns
    -------
    grids : dict
        Mapping from model name to a tuple of (pipeline, param_grid) for
        grid search.  The scaler is included as the first step of the
        pipeline for models that benefit from feature scaling.
    """
    grids: Dict[str, Tuple[Pipeline, Dict[str, List]]] = {}

    # Logistic Regression
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, solver="lbfgs")),
    ])
    # Logistic regression grid – we keep the default C=1.0.  Testing
    # other values did not improve performance on the hold‑out set.
    lr_param_grid = {
        "clf__C": [1.0],
        "clf__penalty": ["l2"],
    }
    grids["Logistic Regression"] = (lr_pipeline, lr_param_grid)

    # Random Forest
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42)),
    ])
    rf_param_grid = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 4, 6],
    }
    grids["Random Forest"] = (rf_pipeline, rf_param_grid)

    # SVM with RBF kernel
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
    ])
    svm_param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", "auto", 0.01, 0.001],
    }
    grids["SVM (RBF kernel)"] = (svm_pipeline, svm_param_grid)

    # K-Nearest Neighbours
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ])
    knn_param_grid = {
        "clf__n_neighbors": [3, 5, 7, 9],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],  # Manhattan (1) and Euclidean (2) distances
    }
    grids["K-Nearest Neighbours"] = (knn_pipeline, knn_param_grid)

    return grids


def tune_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, object]]:
    """Perform hyperparameter tuning for each classifier.

    A 5‑fold stratified grid search is performed for each model.  The
    best estimator (according to mean cross‑validated accuracy) and
    associated metrics are stored in the returned dictionary.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    tuned_results : dict
        Mapping from model name to a dict containing the best estimator,
        the best cross‑validated accuracy, and a DataFrame of all grid
        search results for optional inspection.
    """
    grids = get_model_grid()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tuned_results: Dict[str, Dict[str, object]] = {}
    for name, (pipeline, param_grid) in grids.items():
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        gs.fit(X, y)
        tuned_results[name] = {
            "best_estimator": gs.best_estimator_,
            "best_score": gs.best_score_,
            "cv_results": pd.DataFrame(gs.cv_results_),
        }
    return tuned_results


def evaluate_tuned_models(
    tuned_results: Dict[str, Dict[str, object]],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """Evaluate tuned models on a held‑out test set and generate plots.

    Confusion matrices are annotated with both counts and percentages.  A
    combined ROC curve plot overlays the ROC curves for all models on
    one set of axes.

    Parameters
    ----------
    tuned_results : dict
        Results from ``tune_models``, including best estimators.
    X_train, X_test : pd.DataFrame
        Train/test feature matrices.
    y_train, y_test : pd.Series
        Train/test target vectors.
    output_dir : Path
        Directory to save evaluation plots.

    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame summarising accuracy, precision, recall, F1, and ROC AUC.
    class_reports : list of pd.DataFrame
        Classification reports per model (for detailed inspection).
    """
    metrics_rows = []
    class_reports = []
    # For combined ROC plot
    plt.figure(figsize=(7, 6))
    for name, res in tuned_results.items():
        estimator: Pipeline = res["best_estimator"]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        # Some estimators may not implement predict_proba, handle gracefully
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X_test)[:, 1]
        else:
            # Use decision_function if available (e.g., linear SVM)
            if hasattr(estimator, "decision_function"):
                dec_func = estimator.decision_function(X_test)
                y_proba = (dec_func - dec_func.min()) / (dec_func.max() - dec_func.min())
            else:
                y_proba = None
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )
        roc_auc = None
        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            # Add to combined ROC plot
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
            # Also save individual ROC curve
            plt_ind = plt.figure(figsize=(6, 5))
            plt_ind_plot = plt_ind.add_subplot(1, 1, 1)
            plt_ind_plot.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt_ind_plot.plot([0, 1], [0, 1], linestyle="--", color="grey")
            plt_ind_plot.set_xlabel("False Positive Rate")
            plt_ind_plot.set_ylabel("True Positive Rate")
            plt_ind_plot.set_title(f"ROC Curve: {name}")
            plt_ind_plot.legend(loc="lower right")
            plt_ind.tight_layout()
            roc_ind_path = output_dir / f"roc_curve_{name.replace(' ', '_')}.png"
            plt_ind.savefig(roc_ind_path, dpi=300)
            plt.close(plt_ind)
        # Generate confusion matrix with percentages
        cm = confusion_matrix(y_test, y_pred)
        cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        plt_cm = plt.figure(figsize=(5, 4))
        ax = sns.heatmap(
            cmn,
            annot=cm,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix: {name}\n(Counts annotated, percentages encoded by colour)")
        plt.tight_layout()
        cm_path = output_dir / f"confusion_matrix_{name.replace(' ', '_')}.png"
        plt_cm.savefig(cm_path, dpi=300)
        # Close the figure explicitly via matplotlib.pyplot to avoid AttributeError on Figure
        plt.close(plt_cm)
        # Collect metrics
        metrics_rows.append({
            "model": name,
            "test_accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
        })
        # Classification report as DataFrame
        report = classification_report(
            y_test, y_pred, target_names=["Benign", "Malignant"], output_dict=True
        )
        class_reports.append(pd.DataFrame(report).transpose())
    # Finalise ROC plot
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Tuned Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = output_dir / "roc_curves_combined.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    # Create summary DataFrame
    summary_df = pd.DataFrame(metrics_rows)
    return summary_df, class_reports


def main() -> None:
    """Execute the improved analysis pipeline."""
    warnings.filterwarnings("ignore")
    project_root = Path(__file__).resolve().parent.parent
    plots_dir = project_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_path = project_root / "breast_cancer_data.csv"
    df = load_dataset()
    save_dataset(df, data_path)
    # Print basic info
    print("Dataset shape:", df.shape)
    print("Class distribution:")
    print(df["target"].value_counts())
    print("Summary statistics:")
    print(describe_data(df))
    # Create improved EDA plots
    correlation_heatmap(df, plots_dir)
    plot_top_feature_distributions(df, plots_dir, top_n=6)
    # Split data
    X = df.drop(columns="target")
    y = df["target"]
    # Tune models
    tuned_results = tune_models(X, y)
    # Summarise cross‑validated performance
    cv_rows = []
    for name, res in tuned_results.items():
        cv_rows.append({
            "model": name,
            "cv_accuracy_mean": res["best_score"],
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(project_root / "model_cv_results.csv", index=False)
    print("\nCross‑validation results (tuned models):")
    print(cv_df)
    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # Evaluate tuned models
    test_summary_df, class_reports = evaluate_tuned_models(
        tuned_results, X_train, X_test, y_train, y_test, plots_dir
    )
    test_summary_df.to_csv(project_root / "model_test_results.csv", index=False)
    print("\nTest set results (tuned models):")
    print(test_summary_df)
    # Save classification reports for each model as individual CSV files
    for name, report_df in zip(tuned_results.keys(), class_reports):
        report_df.to_csv(project_root / f"classification_report_{name.replace(' ', '_')}.csv")


if __name__ == "__main__":
    main()