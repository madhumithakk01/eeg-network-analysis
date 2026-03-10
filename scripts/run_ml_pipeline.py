#!/usr/bin/env python3
"""
Full ML pipeline for neurological outcome prediction from patient-level temporal features.

Stages: load data -> quality checks -> correlation filter -> feature selection ->
Stratified 5-Fold CV -> scaling -> train RF/XGBoost/LightGBM -> Optuna tuning ->
final evaluation -> SHAP interpretability -> NCI standalone analysis -> save results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import (
    AUDIT_PATH,
    MODEL_RESULTS_PATH,
    PATIENT_TEMPORAL_DATASET_PATH,
)
from src.modeling.dataset_loader import load_dataset, data_quality_checks
from src.modeling.evaluation import run_cross_validation, compute_metrics
from src.modeling.feature_selection import (
    remove_highly_correlated,
    rank_features_multi_method,
    select_top_k,
    select_top_k_with_nci,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full ML pipeline for outcome prediction.")
    parser.add_argument("--data", type=str, default=None, help="Path to patient_temporal_dataset.parquet")
    parser.add_argument("--metadata", type=str, default=None, help="Path to metadata CSV for outcome merge")
    parser.add_argument("--outcome", type=str, default="Outcome", help="Target column name")
    parser.add_argument("--output-dir", type=str, default=None, help="Results directory (default: MODEL_RESULTS_PATH)")
    parser.add_argument("--top-k", type=int, default=40, help="Number of features to select (30-50)")
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--optuna-trials", type=int, default=0, help="0 = skip Optuna")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP (faster)")
    args = parser.parse_args()

    data_path = args.data or PATIENT_TEMPORAL_DATASET_PATH
    metadata_path = args.metadata or AUDIT_PATH
    output_dir = args.output_dir or MODEL_RESULTS_PATH
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 1 — Data loading")
    print("=" * 60)
    print(f"Dataset path: {data_path}")
    X, y, feature_names = load_dataset(data_path, metadata_path=metadata_path, outcome_column=args.outcome)
    n_rows = X.shape[0]
    try:
        df_check = pd.read_parquet(data_path)
        n_patients = int(df_check["patient_id"].nunique()) if "patient_id" in df_check.columns else n_rows
    except Exception:
        n_patients = n_rows
    print(f"Loaded: {n_rows} rows, {n_patients} unique patients, {X.shape[1]} features. Class counts: {dict(y.value_counts())}")
    if n_rows < 250:
        print("WARNING: Dataset has fewer than 250 patients. The full cohort may not be loaded. Expected ~294.")

    print("\nStage 2 — Data quality")
    X_clean, dropped, report = data_quality_checks(
        X, max_missing_frac=0.5, remove_zero_variance=True, impute_remaining=True
    )
    print(f"Dropped {len(dropped)} columns. Remaining: {X_clean.shape[1]} features.")
    feature_names = [c for c in feature_names if c in X_clean.columns]

    print("\nStage 3 — Correlation filter (threshold={})".format(args.correlation_threshold))
    X_corr, dropped_corr = remove_highly_correlated(X_clean, threshold=args.correlation_threshold)
    print(f"Dropped {len(dropped_corr)} highly correlated. Remaining: {X_corr.shape[1]}")

    print("\nStage 4 — Feature selection (top k, NCI-preserved)")
    ranking = rank_features_multi_method(X_corr, y, random_state=42)
    effective_k = min(args.top_k, 25) if n_rows >= 150 else min(args.top_k, 15)
    selected = select_top_k_with_nci(ranking, k=effective_k, available_columns=X_corr.columns.tolist())
    X_sel = X_corr[selected].copy()
    ranking.to_csv(os.path.join(output_dir, "feature_ranking.csv"), index=False)
    print(f"Selected {len(selected)} features.")

    print("\nStage 5–7 — Stratified 5-Fold CV + Scaling + Model training")
    n_splits = args.n_folds
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    results = {}
    for name, get_factory in [
        ("RandomForest", lambda: get_rf_factory(random_state=42)),
        ("XGBoost", lambda: get_xgboost_factory(random_state=42, scale_pos_weight=scale_pos_weight)),
        ("LightGBM", lambda: get_lightgbm_factory(random_state=42, is_unbalance=True)),
    ]:
        try:
            cv_result = run_cross_validation(
                X_sel, y, get_factory(), n_splits=n_splits, random_state=42, scale=True, use_youden_threshold=True
            )
            results[name] = cv_result
            print(
                f"  {name}: mean ROC-AUC = {cv_result['mean_roc_auc']:.4f} ± {cv_result['std_roc_auc']:.4f} "
                f"| F1 = {cv_result['mean_f1']:.4f} | Sens = {cv_result['mean_sensitivity']:.4f} "
                f"| Spec = {cv_result['mean_specificity']:.4f}"
            )
        except Exception as e:
            print(f"  {name}: failed - {e}")
            results[name] = None

    # Best model by ROC-AUC
    best_name = max(
        (k for k, v in results.items() if v is not None),
        key=lambda k: results[k]["mean_roc_auc"],
        default="RandomForest",
    )
    print(f"\nBest model (ROC-AUC): {best_name}")

    # Optuna (optional)
    if args.optuna_trials > 0:
        print("\nStage 8 — Hyperparameter optimization (Optuna)")
        model_key = best_name.replace(" ", "").lower()
        if "randomforest" in model_key:
            model_key = "rf"
        elif "xgboost" in model_key:
            model_key = "xgboost"
        elif "lightgbm" in model_key:
            model_key = "lightgbm"
        try:
            best_params, study = run_optuna(
                X_sel, y, model_name=model_key, n_trials=args.optuna_trials, n_splits=n_splits
            )
            print(f"  Best params: {best_params}")
            if best_name == "RandomForest":
                factory = get_rf_factory(**best_params)
            elif best_name == "XGBoost":
                factory = get_xgboost_factory(**{**best_params, "scale_pos_weight": scale_pos_weight})
            else:
                factory = get_lightgbm_factory(**{**best_params, "is_unbalance": True})
            cv_result = run_cross_validation(X_sel, y, factory(), n_splits=n_splits, random_state=42, scale=True, use_youden_threshold=True)
            results[best_name] = cv_result
        except Exception as e:
            print(f"  Optuna failed: {e}")

    print("\nStage 9 — Final model and evaluation summary")
    if best_name == "RandomForest":
        final_factory = get_rf_factory(random_state=42)
    elif best_name == "XGBoost":
        final_factory = get_xgboost_factory(random_state=42, scale_pos_weight=scale_pos_weight)
    else:
        final_factory = get_lightgbm_factory(random_state=42, is_unbalance=True)
    final_model, scaler, _ = train_final_model(X_sel, y, final_factory, scale=True, random_state=42)
    cv_result = results.get(best_name)
    metrics = {
        "mean_roc_auc": cv_result["mean_roc_auc"] if cv_result else 0.0,
        "std_roc_auc": cv_result["std_roc_auc"] if cv_result else 0.0,
        "mean_f1": cv_result["mean_f1"] if cv_result else 0.0,
        "mean_sensitivity": cv_result["mean_sensitivity"] if cv_result else 0.0,
        "mean_specificity": cv_result["mean_specificity"] if cv_result else 0.0,
        "mean_accuracy": cv_result["mean_accuracy"] if cv_result else 0.0,
    }
    with open(os.path.join(output_dir, "cv_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Mean ROC-AUC: {metrics['mean_roc_auc']:.4f} ± {metrics['std_roc_auc']:.4f}")
    print(f"  Mean F1: {metrics['mean_f1']:.4f} | Sensitivity: {metrics['mean_sensitivity']:.4f} | Specificity: {metrics['mean_specificity']:.4f}")

    # Save model and scaler (joblib is a dependency of scikit-learn)
    import joblib
    joblib.dump(final_model, os.path.join(output_dir, "best_model.joblib"))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    pd.Series(selected).to_csv(os.path.join(output_dir, "selected_features.csv"), index=False, header=["feature"])

    # ROC curves and confusion matrices
    if cv_result:
        for i, (fpr, tpr) in enumerate(cv_result["roc_curves"]):
            pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
                os.path.join(output_dir, f"roc_fold_{i}.csv"), index=False
            )
        for i, cm in enumerate(cv_result["confusion_matrices"]):
            pd.DataFrame(cm).to_csv(os.path.join(output_dir, f"confusion_fold_{i}.csv"), index=False)

    print("\nStage 10 — Model interpretation (SHAP)")
    if not args.no_shap:
        try:
            _, imp_df = compute_shap_importance(final_model, X_sel[selected], selected, n_samples=200)
            imp_df.to_csv(os.path.join(output_dir, "shap_importance.csv"), index=False)
            top20 = imp_df.head(20)
            print("  Top 20 features (mean |SHAP|):")
            for _, row in top20.iterrows():
                print(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")
            shap_summary_plot(
                final_model, X_sel[selected], selected,
                save_path=os.path.join(output_dir, "shap_summary.png"),
                n_samples=min(200, len(X_sel)),
            )
        except Exception as e:
            print(f"  SHAP failed: {e}")
    else:
        print("  Skipped (--no-shap).")

    print("\nStage 11 — NCI standalone analysis")
    nci_df = nci_standalone_analysis(X_clean, y)
    if len(nci_df) > 0:
        nci_df.to_csv(os.path.join(output_dir, "nci_standalone_roc_auc.csv"), index=False)
        print(nci_df.to_string(index=False))
    else:
        print("  No NCI columns found in dataset.")

    print("\nStage 12 — Outputs saved to", output_dir)
    print("  best_model.joblib, scaler.joblib, selected_features.csv")
    print("  cv_metrics.json, feature_ranking.csv, shap_importance.csv, nci_standalone_roc_auc.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
