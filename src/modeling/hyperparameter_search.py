"""
Hyperparameter optimization with Optuna using Stratified 5-Fold CV (ROC-AUC).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold

from .model_training import get_lightgbm_factory, get_rf_factory, get_xgboost_factory


def _objective_rf(trial, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> float:
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 4, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    factory = get_rf_factory(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    model = factory()
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def _objective_xgb(trial, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> float:
    try:
        import xgboost as xgb
    except ImportError:
        return 0.5
    n_estimators = trial.suggest_int("n_estimators", 50, 250)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def _objective_lgb(trial, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> float:
    try:
        import lightgbm as lgb
    except ImportError:
        return 0.5
    n_estimators = trial.suggest_int("n_estimators", 50, 250)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        verbose=-1,
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def run_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "rf",
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[dict, Any]:
    """
    Run Optuna optimization for given model (rf, xgboost, lightgbm).
    Returns (best_params, optuna_study).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if model_name == "rf":
        objective = lambda t: _objective_rf(t, X_arr, y_arr, cv)
    elif model_name in ("xgboost", "xgb"):
        objective = lambda t: _objective_xgb(t, X_arr, y_arr, cv)
    elif model_name in ("lightgbm", "lgb"):
        objective = lambda t: _objective_lgb(t, X_arr, y_arr, cv)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state, n_startup_trials=10))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study
