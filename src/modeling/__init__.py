"""ML training and evaluation for outcome prediction."""

from .dataset_loader import data_quality_checks, load_dataset
from .evaluation import compute_metrics, run_cross_validation
from .feature_selection import (
    rank_features_multi_method,
    remove_highly_correlated,
    select_top_k,
)
from .hyperparameter_search import run_optuna
from .interpretability import compute_shap_importance, nci_standalone_analysis, shap_summary_plot
from .model_training import (
    get_lightgbm_factory,
    get_rf_factory,
    get_xgboost_factory,
    train_final_model,
)

__all__ = [
    "data_quality_checks",
    "load_dataset",
    "compute_metrics",
    "run_cross_validation",
    "rank_features_multi_method",
    "remove_highly_correlated",
    "select_top_k",
    "run_optuna",
    "compute_shap_importance",
    "nci_standalone_analysis",
    "shap_summary_plot",
    "get_lightgbm_factory",
    "get_rf_factory",
    "get_xgboost_factory",
    "train_final_model",
]
