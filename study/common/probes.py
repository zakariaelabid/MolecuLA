from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _finite_xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return X[mask], y[mask], mask


def fit_linear_probes(
    Z: np.ndarray,
    panel: pd.DataFrame,
    properties: list[str],
    split: np.ndarray,
    model_kind: str = "ridge",
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    rows = []
    models = {}
    for prop in properties:
        X, y, finite = _finite_xy(Z, panel[prop])
        split_f = np.asarray(split)[finite]
        train = split_f == "train"
        estimator = Ridge(alpha=alpha) if model_kind == "ridge" else LinearRegression()
        pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        pipe.fit(X[train], y[train])
        row = {"property": prop, "model_kind": model_kind, "alpha": alpha if model_kind == "ridge" else np.nan}
        for split_name in ["train", "val", "test"]:
            mask = split_f == split_name
            row[f"r2_{split_name}"] = r2_score(y[mask], pipe.predict(X[mask])) if mask.sum() > 1 else np.nan
        rows.append(row)
        models[prop] = pipe
    return pd.DataFrame(rows), models


def residualize_properties(
    panel: pd.DataFrame,
    properties: list[str],
    confounds: list[str],
    split: np.ndarray,
    alpha: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    rows = []
    C_all = out[confounds].to_numpy(dtype=float)
    for prop in properties:
        y_all = out[prop].to_numpy(dtype=float)
        X, y, finite = _finite_xy(C_all, y_all)
        split_f = np.asarray(split)[finite]
        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])
        pipe.fit(X[split_f == "train"], y[split_f == "train"])
        residual = np.full(len(out), np.nan, dtype=float)
        residual[finite] = y - pipe.predict(X)
        out[f"resid_{prop}"] = residual
        row = {"property": prop, "residual_column": f"resid_{prop}", "alpha": alpha}
        for split_name in ["train", "val", "test"]:
            mask = split_f == split_name
            row[f"confound_r2_{split_name}"] = r2_score(y[mask], pipe.predict(X[mask])) if mask.sum() > 1 else np.nan
        rows.append(row)
    return out, pd.DataFrame(rows)


def run_mlp_probes(
    Z: np.ndarray,
    panel: pd.DataFrame,
    properties: list[str],
    split: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (256, 128),
    random_state: int = 42,
    max_iter: int = 200,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    rows = []
    models = {}
    for prop in properties:
        X, y, finite = _finite_xy(Z, panel[prop])
        split_f = np.asarray(split)[finite]
        train = split_f == "train"
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layer_sizes,
                        random_state=random_state,
                        max_iter=max_iter,
                        early_stopping=True,
                    ),
                ),
            ]
        )
        pipe.fit(X[train], y[train])
        row = {"property": prop, "hidden_layer_sizes": str(hidden_layer_sizes)}
        for split_name in ["train", "val", "test"]:
            mask = split_f == split_name
            row[f"r2_{split_name}"] = r2_score(y[mask], pipe.predict(X[mask])) if mask.sum() > 1 else np.nan
        rows.append(row)
        models[prop] = pipe
    return pd.DataFrame(rows), models
