# dpa_tools/conditions.py
"""Condition manager for scalar condition inputs (e.g. temperature, pressure)."""

import pickle

import numpy as np


class DPAConditionError(Exception):
    """Raised when conditions are missing, mismatched, or used before fit."""
    pass


class ConditionManager:
    """Fit a StandardScaler per named condition key, then produce a
    normalized (n, d_total) array for downstream concatenation.
    """

    def __init__(self):
        self._scalers = None
        self._keys = None

    def fit(self, conditions: dict[str, np.ndarray]) -> None:
        from sklearn.preprocessing import StandardScaler

        self._scalers = {}
        self._keys = sorted(conditions.keys())
        for key in self._keys:
            scaler = StandardScaler()
            scaler.fit(np.asarray(conditions[key]).reshape(-1, 1))
            self._scalers[key] = scaler

    def transform(self, conditions: dict[str, np.ndarray]) -> np.ndarray:
        if self._scalers is None:
            raise DPAConditionError(
                "ConditionManager.transform() called before fit()."
            )
        parts = []
        for key in self._keys:
            if key not in conditions:
                raise DPAConditionError(
                    f"Condition key {key!r} was present at fit time "
                    f"but is missing from transform()."
                )
            x = self._scalers[key].transform(
                np.asarray(conditions[key]).reshape(-1, 1)
            )
            parts.append(x)
        return np.hstack(parts)

    def fit_transform(self, conditions: dict[str, np.ndarray]) -> np.ndarray:
        self.fit(conditions)
        return self.transform(conditions)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"scalers": self._scalers, "keys": self._keys}, f)

    @classmethod
    def load(cls, path: str) -> "ConditionManager":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._scalers = data["scalers"]
        obj._keys = data["keys"]
        return obj
