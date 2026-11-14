"""Metrics and diagnostics."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def residuals(y_true, y_pred) -> np.ndarray:
    """Prediction error = predicted - actual (same sign convention as the notebook)."""
    return np.asarray(y_pred) - np.asarray(y_true)


def feature_importance(model, feature_cols) -> pd.DataFrame:
    """RF: impurity importance. Linear: |standardised coefficient|, normalised."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        kind = "Impurity importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
        total = values.sum()
        values = values / total if total else values
        kind = "Relative |coefficient|"
    else:
        return pd.DataFrame(columns=["Feature", "Importance", "Kind"])
    out = pd.DataFrame({"Feature": feature_cols, "Importance": values, "Kind": kind})
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)
