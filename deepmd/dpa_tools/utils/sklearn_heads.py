# utils/sklearn_heads.py
#
# Single source of truth for building sklearn predictor heads.
# Used by DPAFineTuner._fit_sklearn() and cv._build_sklearn_head().

from __future__ import annotations


def build_sklearn_head(predictor_type: str, seed: int = 42):
    """Build an sklearn estimator for the given predictor type.

    Parameters
    ----------
    predictor_type : str
        One of ``"rf"``, ``"linear"`` / ``"ridge"``, or ``"mlp"``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    estimator
        An sklearn-compatible regressor (NOT wrapped in a Pipeline).

    Raises
    ------
    ValueError
        If *predictor_type* is not recognised.
    """
    if predictor_type in ("linear", "ridge"):
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0, random_state=seed)

    if predictor_type == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=100, random_state=seed)

    if predictor_type == "mlp":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=(512, 512, 256),
            max_iter=2000,
            alpha=0.0,
            learning_rate_init=1e-3,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )

    raise ValueError(
        f"Unknown predictor type: {predictor_type!r}. "
        "Supported: 'rf', 'linear'/'ridge', 'mlp'."
    )
