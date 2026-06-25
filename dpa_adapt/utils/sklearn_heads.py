# SPDX-License-Identifier: LGPL-3.0-or-later
# utils/sklearn_heads.py
#
# Single source of truth for building sklearn predictor heads.
# Used by DPAFineTuner._fit_sklearn() and cv._build_sklearn_head().

from typing import (
    Any,
)


def build_sklearn_head(predictor_type: str, seed: int = 42, n_outputs: int = 1) -> Any:
    """Build an sklearn estimator for the given predictor type.

    Parameters
    ----------
    predictor_type : str
        One of ``"rf"``, ``"linear"`` / ``"ridge"``, or ``"mlp"``.
    seed : int
        Random seed for reproducibility.
    n_outputs : int
        Number of output dimensions.  When > 1, ``"rf"`` and ``"ridge"``
        are automatically wrapped in ``MultiOutputRegressor``.  ``"mlp"``
        supports multi-output natively and ignores this parameter.

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
        from sklearn.linear_model import (
            Ridge,
        )

        est = Ridge(alpha=1.0, random_state=seed)
        if n_outputs > 1:
            from sklearn.multioutput import (
                MultiOutputRegressor,
            )

            return MultiOutputRegressor(est)
        return est

    if predictor_type == "rf":
        from sklearn.ensemble import (
            RandomForestRegressor,
        )

        est = RandomForestRegressor(n_estimators=100, random_state=seed)
        if n_outputs > 1:
            from sklearn.multioutput import (
                MultiOutputRegressor,
            )

            return MultiOutputRegressor(est)
        return est

    if predictor_type == "mlp":
        from sklearn.neural_network import (
            MLPRegressor,
        )

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
