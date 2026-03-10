"""
Model training: Random Forest, XGBoost, LightGBM with Stratified K-Fold CV.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .evaluation import run_cross_validation


def get_rf_factory(**kwargs) -> Callable[[], Any]:
    """Return a callable that returns a new RandomForestClassifier."""
    def factory():
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 12),
            min_samples_split=kwargs.get("min_samples_split", 5),
            min_samples_leaf=kwargs.get("min_samples_leaf", 2),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1,
        )
    return factory


def get_xgboost_factory(**kwargs) -> Callable[[], Any]:
    """Return a callable that returns a new XGBoost classifier."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is required. pip install xgboost")
    def factory():
        return xgb.XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 150),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            random_state=kwargs.get("random_state", 42),
            use_label_encoder=False,
            eval_metric="logloss",
        )
    return factory


def get_lightgbm_factory(**kwargs) -> Callable[[], Any]:
    """Return a callable that returns a new LightGBM classifier."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is required. pip install lightgbm")
    def factory():
        return lgb.LGBMClassifier(
            n_estimators=kwargs.get("n_estimators", 150),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            random_state=kwargs.get("random_state", 42),
            verbose=-1,
        )
    return factory


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], Any],
    scale: bool = True,
    random_state: int = 42,
) -> tuple[Any, Any | None, dict]:
    """
    Train final model on full data. Returns (model, scaler, metrics_dict).
    metrics_dict is empty; run CV separately for reported metrics.
    """
    X = X.copy()
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    if scale:
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
    else:
        scaler = None
        X_s = np.asarray(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)
    model = model_factory()
    model.fit(X_s, y)
    return model, scaler, {}
