#!/usr/bin/env python3
"""
Nested Stratified 5-Fold Cross-Validation pipeline for outcome prediction.

Experiment 2: rigorous evaluation with outer loop for performance estimation
and inner loop (Optuna) for hyperparameter tuning on training fold only.
Feature selection and scaling are performed inside each outer fold to prevent leakage.

Outputs: analysis/model_results_nested_cv/ (or --output-dir).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import (
    ANALYSIS_OUTPUT_PATH,
    AUDIT_PATH,
    PATIENT_TEMPORAL_DATASET_PATH,
)
from src.modeling.dataset_loader import load_dataset, data_quality_checks
from src.modeling.evaluation import compute_metrics
from src.modeling.feature_selection import (
    remove_highly_correlated,
    rank_features_multi_method,
    select_top_k,
)
from src.modeling.hyperparameter_search import run_optuna
from src.modeling.interpretability import (
    compute_shap_importance,
    nci_standalone_analysis,
    shap_summary_plot,
)
from src.modeling.model_training import (
    get_lightgbm_factory,
    get_rf_factory,
    get_xgboost_factory,
    train_final_model,
)

# Default output directory for nested CV (separate from run_ml_pipeline.py).
DEFAULT_NESTED_CV_OUTPUT = os.path.join(ANALYSIS_OUTPUT_PATH, "model_results_nested_cv")

# Single model type for nested CV (LightGBM as default; can add CLI later).
DEFAULT_MODEL_NAME = "lightgbm"


def _get_factory_with_params(model_name: str, best_params: dict):
    """Return model factory with optional best_params (from Optuna)."""
    if model_name in ("rf", "randomforest"):
        return get_rf_factory(**best_params)
    if model_name in ("xgboost", "xgb"):
        return get_xgboost_factory(**best_params)
    return get_lightgbm_factory(**best_params)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run nested stratified 5-fold CV pipeline for outcome prediction."
    )
    parser.add_argument("--data", type=str, default=None, help="Path to patient_temporal_dataset.parquet")
    parser.add_argument("--metadata", type=str, default=None, help="Path to metadata CSV for outcome merge")
    parser.add_argument("--outcome", type=str, default="Outcome", help="Target column name")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Results directory (default: {DEFAULT_NESTED_CV_OUTPUT})",
    )
    parser.add_argument("--top-k", type=int, default=40, help="Number of features to select (30-50)")
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--n-folds", type=int, default=5, help="Outer and inner fold count")
    parser.add_argument("--optuna-trials", type=int, default=30, help="Optuna trials per outer fold (0 = use defaults)")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP (faster)")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        choices=("rf", "xgboost", "lightgbm"),
        help="Model to use for nested CV",
    )
    args = parser.parse_args()

    data_path = args.data or PATIENT_TEMPORAL_DATASET_PATH
    metadata_path = args.metadata or AUDIT_PATH
    output_dir = args.output_dir or DEFAULT_NESTED_CV_OUTPUT
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Nested Stratified 5-Fold CV Pipeline")
    print("=" * 60)

    print("\nStage 1 — Data loading")
    X, y, feature_names = load_dataset(data_path, metadata_path=metadata_path, outcome_column=args.outcome)
    print(f"Loaded: {X.shape[0]} patients, {X.shape[1]} features. Class counts: {dict(y.value_counts())}")

    print("\nStage 2 — Data quality (global, no leakage)")
    X_clean, dropped, report = data_quality_checks(
        X, max_missing_frac=0.5, remove_zero_variance=True, impute_remaining=True
    )
    print(f"Dropped {len(dropped)} columns. Remaining: {X_clean.shape[1]} features.")
    X_clean = X_clean.reset_index(drop=True)
    y = y.reindex(X_clean.index).dropna()
    X_clean = X_clean.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)

    n_outer = args.n_folds
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)

    fold_metrics = []
    all_roc_rows = []
    all_cm_rows = []
    best_params_per_fold = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_clean, y)):
        print(f"\n--- Outer fold {outer_fold + 1}/{n_outer} ---")
        X_train_outer = X_clean.iloc[train_idx].copy()
        y_train_outer = y.iloc[train_idx].copy()
        X_test_outer = X_clean.iloc[test_idx].copy()
        y_test_outer = y.iloc[test_idx].copy()

        # Feature processing ONLY on train_outer
        X_train_corr, dropped_corr = remove_highly_correlated(
            X_train_outer, threshold=args.correlation_threshold
        )
        test_cols = [c for c in X_test_outer.columns if c in X_train_corr.columns]
        X_test_corr = X_test_outer[test_cols].copy()

        ranking = rank_features_multi_method(X_train_corr, y_train_outer, random_state=42)
        k = min(args.top_k, len(ranking))
        selected = select_top_k(ranking, k=k)
        X_train_sel = X_train_corr[selected].copy()
        X_test_sel = X_test_corr[[c for c in selected if c in X_test_corr.columns]].copy()
        if X_test_sel.shape[1] != len(selected):
            missing = set(selected) - set(X_test_sel.columns)
            for m in missing:
                X_test_sel[m] = 0.0
            X_test_sel = X_test_sel[selected]

        # Optuna on train_outer only (inner 5-fold CV)
        if args.optuna_trials > 0:
            try:
                best_params, _ = run_optuna(
                    X_train_sel,
                    y_train_outer,
                    model_name=args.model,
                    n_trials=args.optuna_trials,
                    n_splits=args.n_folds,
                    random_state=42,
                )
                best_params_per_fold.append(best_params)
            except Exception as e:
                print(f"  Optuna failed: {e}. Using default params.")
                best_params_per_fold.append({})
        else:
            best_params_per_fold.append({})

        factory = _get_factory_with_params(args.model, best_params_per_fold[-1])

        # Scale on train_outer, transform test_outer
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sel)
        X_test_s = scaler.transform(X_test_sel)
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

        model = factory()
        model.fit(X_train_s, y_train_outer)

        y_proba = model.predict_proba(X_test_s)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        m = compute_metrics(y_test_outer.values, y_pred, y_proba)
        m["fold"] = outer_fold
        fold_metrics.append(m)
        print(
            f"  ROC-AUC={m['roc_auc']:.4f} Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
            f"Sens={m['sensitivity']:.4f} Spec={m['specificity']:.4f}"
        )

        fpr, tpr, _ = roc_curve(y_test_outer, y_proba)
        for i, (fp, tp) in enumerate(zip(fpr, tpr)):
            all_roc_rows.append({"fold": outer_fold, "fpr": fp, "tpr": tp})
        cm = confusion_matrix(y_test_outer, y_pred, labels=[0, 1])
        for i in range(2):
            for j in range(2):
                all_cm_rows.append({"fold": outer_fold, "row": i, "col": j, "value": int(cm[i, j])})

    df_metrics = pd.DataFrame(fold_metrics)
    summary = {
        "mean_roc_auc": float(df_metrics["roc_auc"].mean()),
        "std_roc_auc": float(df_metrics["roc_auc"].std()),
        "mean_accuracy": float(df_metrics["accuracy"].mean()),
        "mean_f1": float(df_metrics["f1"].mean()),
        "mean_sensitivity": float(df_metrics["sensitivity"].mean()),
        "mean_specificity": float(df_metrics["specificity"].mean()),
    }

    df_metrics.to_csv(os.path.join(output_dir, "nested_cv_metrics.csv"), index=False)
    with open(os.path.join(output_dir, "nested_cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(all_roc_rows).to_csv(os.path.join(output_dir, "per_fold_roc_curves.csv"), index=False)
    pd.DataFrame(all_cm_rows).to_csv(os.path.join(output_dir, "per_fold_confusion_matrices.csv"), index=False)

    print("\n" + "=" * 60)
    print("Nested CV summary (mean ± std)")
    print("=" * 60)
    print(f"  ROC-AUC:    {summary['mean_roc_auc']:.4f} ± {summary['std_roc_auc']:.4f}")
    print(f"  Accuracy:   {summary['mean_accuracy']:.4f}")
    print(f"  F1:         {summary['mean_f1']:.4f}")
    print(f"  Sensitivity: {summary['mean_sensitivity']:.4f}")
    print(f"  Specificity: {summary['mean_specificity']:.4f}")

    # Optional final model: global feature selection + Optuna on full data
    print("\n--- Optional final model (full data) ---")
    X_corr, _ = remove_highly_correlated(X_clean, threshold=args.correlation_threshold)
    ranking_global = rank_features_multi_method(X_corr, y, random_state=42)
    selected_global = select_top_k(ranking_global, k=min(args.top_k, len(ranking_global)))
    X_sel_global = X_corr[selected_global].copy()

    if args.optuna_trials > 0:
        try:
            best_params_final, _ = run_optuna(
                X_sel_global, y, model_name=args.model,
                n_trials=args.optuna_trials, n_splits=args.n_folds, random_state=42,
            )
        except Exception:
            best_params_final = {}
    else:
        best_params_final = {}
    final_factory = _get_factory_with_params(args.model, best_params_final)
    final_model, final_scaler, _ = train_final_model(
        X_sel_global, y, final_factory, scale=True, random_state=42
    )
    import joblib
    joblib.dump(final_model, os.path.join(output_dir, "final_model.joblib"))
    if final_scaler is not None:
        joblib.dump(final_scaler, os.path.join(output_dir, "final_scaler.joblib"))
    pd.Series(selected_global).to_csv(
        os.path.join(output_dir, "selected_features.csv"), index=False, header=["feature"]
    )
    print("  Saved: final_model.joblib, final_scaler.joblib, selected_features.csv")

    # SHAP on final model
    if not args.no_shap:
        print("\n--- Interpretability (SHAP) ---")
        try:
            _, imp_df = compute_shap_importance(
                final_model, X_sel_global, selected_global, n_samples=200
            )
            imp_df.to_csv(os.path.join(output_dir, "feature_importance_nested_cv.csv"), index=False)
            shap_summary_plot(
                final_model, X_sel_global, selected_global,
                save_path=os.path.join(output_dir, "shap_summary_nested_cv.png"),
                n_samples=min(200, len(X_sel_global)),
            )
            print("  Saved: feature_importance_nested_cv.csv, shap_summary_nested_cv.png")
        except Exception as e:
            print(f"  SHAP failed: {e}")
    else:
        print("\n  SHAP skipped (--no-shap).")

    # NCI standalone (optional, same as main pipeline)
    nci_df = nci_standalone_analysis(X_clean, y)
    if len(nci_df) > 0:
        nci_df.to_csv(os.path.join(output_dir, "nci_standalone_roc_auc.csv"), index=False)
        print("  Saved: nci_standalone_roc_auc.csv")

    print("\nAll outputs saved to:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
