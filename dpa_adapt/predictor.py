# dpa_adapt/predictor.py

import numpy as np

from dpa_adapt.conditions import DPAConditionError
from dpa_adapt.data.loader import load_data
from dpa_adapt.utils.dotdict import DotDict


def _unwrap_multioutput(est):
    """If *est* is a ``MultiOutputRegressor``, return the wrapped estimator."""
    from sklearn.multioutput import MultiOutputRegressor

    if isinstance(est, MultiOutputRegressor):
        return est.estimator
    return est


def _is_rf(est):
    from sklearn.ensemble import RandomForestRegressor

    return isinstance(_unwrap_multioutput(est), RandomForestRegressor)


def _is_ridge(est):
    from sklearn.linear_model import Ridge

    return isinstance(_unwrap_multioutput(est), Ridge)


def _is_mlp(est):
    from sklearn.neural_network import MLPRegressor

    return isinstance(est, MLPRegressor)


class DPAPredictor:
    """
    Read-only inference wrapper for a frozen DPA+sklearn bundle.

    Parameters
    ----------
    model_path : str
        Path to a frozen model file produced by ``DPAFineTuner.freeze()``.
    n_committee : int
        Number of committee members for uncertainty estimation.
        Default 1 uses the single estimator from the bundle unchanged.
    """

    def __init__(self, model_path: str, n_committee: int = 1):
        from dpa_adapt._backend import load_torch_file

        bundle = load_torch_file(model_path)

        # Reject bundles from future versions we cannot read.
        fmt = bundle.get("format_version")
        if fmt is not None and fmt != 1:
            raise ValueError(
                f"Unsupported frozen-model format version {fmt}. "
                "This version of dpa_adapt only supports format_version 1. "
                "Re-freeze the model with the current dpa_adapt version."
            )

        # Detect models frozen with dpa_adapt <0.2 (missing modern metadata).
        if "predictor" in bundle and "pooling" not in bundle:
            raise ValueError(
                "This model was frozen with dpa_adapt <0.2. "
                "Re-freeze with the current version: "
                "model.freeze(output_dir)."
            )

        self._predictor         = bundle["predictor"]
        self._target_key        = bundle["target_key"]  # str or list[str]
        self._type_map          = bundle["type_map"]
        self._task_dim          = bundle["task_dim"]
        self._pretrained        = bundle["pretrained"]
        self._model_branch      = bundle.get("model_branch")
        self._pooling           = bundle["pooling"]
        self._condition_manager = bundle.get("condition_manager")
        self.n_committee        = n_committee

        # Detect estimator type from the final pipeline step.
        final_est = self._predictor.steps[-1][1]
        if _is_rf(final_est):
            self._estimator_type = "rf"
        elif _is_ridge(final_est):
            self._estimator_type = "ridge"
        elif _is_mlp(final_est):
            self._estimator_type = "mlp"
        else:
            self._estimator_type = "unknown"

        from dpa_adapt.finetuner import DPAFineTuner

        # TODO: replace with dedicated DescriptorExtractor class after refactor.
        # For now, DPAFineTuner is reused purely as a descriptor feature extractor.
        self._extractor = DPAFineTuner(
            pretrained=self._pretrained,
            model_branch=self._model_branch,
            predictor="linear",
            pooling=self._pooling,
        )

    def fit(self, data, target_key=None, labels=None, fmt=None, conditions=None):
        """Train committee members for uncertainty estimation.

        Only valid when *n_committee* > 1.  Clones the frozen sklearn
        pipeline *n_committee* times with different random seeds and
        stores the ensemble as ``self.estimators_``.  Also computes
        ``self.uncertainty_threshold_`` (95th-percentile train-set std).
        """
        if self.n_committee <= 1:
            raise RuntimeError(
                "fit() requires n_committee > 1. "
                "The single-estimator predictor is ready to use as-is."
            )

        from sklearn.base import clone

        from dpa_adapt.conditions import ConditionManager
        from dpa_adapt.finetuner import _load_labels

        if target_key is not None and labels is not None:
            raise ValueError("target_key and labels are mutually exclusive")
        if target_key is None and labels is None:
            raise ValueError("Either target_key or labels must be provided")

        systems = load_data(data, fmt=fmt)
        if self._extractor._model is None:
            self._extractor._model = self._extractor._load_descriptor_model()
        self._extractor._validate_type_map(self._type_map, systems)
        features = self._extractor._extract_features(systems)

        if self._condition_manager is not None:
            if conditions is None:
                raise DPAConditionError(
                    "This model was fit with conditions. "
                    "Pass conditions= to fit()."
                )
            X_cond = self._condition_manager.transform(conditions)
            features = np.concatenate([features, X_cond], axis=1)
        elif conditions is not None:
            raise DPAConditionError(
                "This model was fit without conditions."
            )

        if labels is not None:
            y = np.asarray(labels)
        else:
            y = _load_labels(systems, target_key)

        y_flat = y.ravel() if y.ndim == 1 or y.shape[-1] == 1 else y

        self.estimators_ = []
        for seed in range(self.n_committee):
            est = clone(self._predictor)
            try:
                est[-1].set_params(random_state=seed)
            except ValueError:
                pass
            est.fit(features, y_flat)
            self.estimators_.append(est)

        preds = np.array([e.predict(features) for e in self.estimators_])
        preds = preds.reshape(self.n_committee, -1, self._task_dim)
        self.uncertainty_threshold_ = float(
            np.percentile(np.std(preds, axis=0), 95)
        )

    def _extract_and_condition(self, data, fmt, conditions):
        """Shared feature extraction + condition concatenation."""
        systems = load_data(data, fmt=fmt)
        # Load the model first so the checkpoint type_map is available, then
        # validate before extracting features (extraction relies on the data
        # type_map being a subset of the checkpoint's).
        if self._extractor._model is None:
            self._extractor._model = self._extractor._load_descriptor_model()
        self._extractor._validate_type_map(self._type_map, systems)
        features = self._extractor._extract_features(systems)

        if self._condition_manager is not None:
            if conditions is None:
                raise DPAConditionError(
                    "This model was fit with conditions. "
                    "Pass conditions= to predict()."
                )
            X_cond = self._condition_manager.transform(conditions)
            features = np.concatenate([features, X_cond], axis=1)
        elif conditions is not None:
            raise DPAConditionError(
                "This model was fit without conditions."
            )

        return features

    def predict(self, data, fmt=None, conditions=None, return_uncertainty=False) -> DotDict:
        """
        Run inference on ``data``.

        Parameters
        ----------
        data : str | list[str]
            Path(s) to deepmd/npy system directories.
        fmt : str, optional
            Reserved for future format support.
        conditions : dict[str, np.ndarray], optional
            Named condition arrays.  Required when the model was fit with
            conditions; must be absent otherwise.
        return_uncertainty : bool
            When True, include ``"uncertainty"`` (per-sample std) in the
            result.  Behaviour depends on estimator type and committee
            configuration.

        Returns
        -------
        DotDict
            ``predictions`` : np.ndarray, shape (n_frames, task_dim)
            ``uncertainty`` : np.ndarray, shape (n_frames, task_dim)  (if requested)
        """
        features = self._extract_and_condition(data, fmt, conditions)

        if return_uncertainty:
            return self._predict_with_uncertainty(features)

        if self.n_committee > 1:
            preds = np.array([e.predict(features) for e in self.estimators_])
            preds = preds.reshape(self.n_committee, -1, self._task_dim)
            return DotDict({"predictions": np.mean(preds, axis=0)})

        raw = self._predictor.predict(features)
        predictions = np.asarray(raw).reshape(-1, self._task_dim)
        return DotDict({"predictions": predictions})

    def _predict_with_uncertainty(self, features):
        """Per-estimator uncertainty dispatch."""
        if self._estimator_type == "rf":
            X_t = features
            for _, step in self._predictor.steps[:-1]:
                X_t = step.transform(X_t)
            rf = self._predictor.steps[-1][1]
            tree_preds = np.array([t.predict(X_t) for t in rf.estimators_])
            tree_preds = tree_preds.reshape(
                len(rf.estimators_), -1, self._task_dim,
            )
            return DotDict({
                "predictions": np.mean(tree_preds, axis=0),
                "uncertainty": np.std(tree_preds, axis=0),
            })

        if self._estimator_type in ("ridge", "linear"):
            raise ValueError(
                "Ridge regression has a unique closed-form solution and "
                "cannot produce uncertainty estimates. "
                "Use estimator='rf' or estimator='mlp' for uncertainty."
            )

        if self.n_committee > 1:
            preds = np.array([e.predict(features) for e in self.estimators_])
            preds = preds.reshape(self.n_committee, -1, self._task_dim)
            return DotDict({
                "predictions": np.mean(preds, axis=0),
                "uncertainty": np.std(preds, axis=0),
            })

        raise RuntimeError(
            f"Uncertainty estimation requires either estimator='rf' "
            f"or n_committee > 1 (for committee-based uncertainty). "
            f"Got estimator_type={self._estimator_type!r} "
            f"with n_committee={self.n_committee}."
        )

    def evaluate(self, data, fmt=None, conditions=None) -> DotDict:
        """
        Predict on ``data`` and compute evaluation metrics against stored labels.

        Parameters
        ----------
        data : str | list[str]
            Path(s) to deepmd/npy system directories with label files.
        fmt : str, optional
            Reserved for future format support.
        conditions : dict[str, np.ndarray], optional
            Named condition arrays.  Required when the model was fit with
            conditions; must be absent otherwise.

        Returns
        -------
        DotDict
            mae, rmse, r2 : float
            predictions   : np.ndarray, shape (n_frames, task_dim)
            labels        : np.ndarray, shape (n_frames, task_dim)
        """
        from dpa_adapt.finetuner import _load_labels
        from dpa_adapt.data.errors import DPADataError

        result      = self.predict(data, fmt=fmt, conditions=conditions)
        predictions = result.predictions

        systems = load_data(data, fmt=fmt)
        labels  = _load_labels(systems, self._target_key)
        labels  = labels.reshape(predictions.shape)

        if predictions.shape != labels.shape:
            raise DPADataError(
                f"Shape mismatch: predictions {predictions.shape} vs "
                f"labels {labels.shape}."
            )

        err    = predictions - labels
        if isinstance(self._target_key, list):
            # Per-property metrics
            keys = self._target_key
            mae, rmse, r2 = {}, {}, {}
            for i, key in enumerate(keys):
                e_i = err[:, i]
                mae[key] = float(np.mean(np.abs(e_i)))
                rmse[key] = float(np.sqrt(np.mean(e_i ** 2)))
                ss_res_i = np.sum(e_i ** 2)
                ss_tot_i = np.sum((labels[:, i] - labels[:, i].mean()) ** 2)
                r2[key] = float(1.0 - ss_res_i / ss_tot_i) if ss_tot_i > 0 else float("nan")
        else:
            mae    = float(np.mean(np.abs(err)))
            rmse   = float(np.sqrt(np.mean(err ** 2)))
            ss_res = np.sum(err ** 2)
            ss_tot = np.sum((labels - labels.mean()) ** 2)
            r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        return DotDict({
            "mae":         mae,
            "rmse":        rmse,
            "r2":          r2,
            "predictions": predictions,
            "labels":      labels,
        })
