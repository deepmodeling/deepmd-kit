# SPDX-License-Identifier: LGPL-3.0-or-later
# cv.py
#
# sklearn-style split and cross-validation for dpdata systems.
# Leak-proof: all operations group by formula / user-provided groups so that
# the same formula never appears in both train and validation/test.

from __future__ import (
    annotations,
)

import json
import logging
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
from sklearn.pipeline import (
    make_pipeline,
)
from sklearn.preprocessing import (
    StandardScaler,
)

from dpa_adapt.data.loader import (
    _get_source,
    _resolve_label_key,
)

_LOG = logging.getLogger("dpa_adapt.cv")


# ---------------------------------------------------------------------------
# internal: formula / group helpers
# ---------------------------------------------------------------------------


def _extract_formula(system: Any) -> str:
    """Extract the formula name from a system.

    Uses the source path stored during loading (``_dpa_source`` attribute).
    Falls back to a system hash when no source path is available.
    """
    source = _get_source(system)
    if source is not None:
        return Path(source).resolve().parent.name
    return f"sys_{id(system)}"


def _formula_to_group(systems: list) -> list[str]:
    """Return one group label per system, derived from its path formula."""
    return [_extract_formula(s) for s in systems]


def _group_indices(groups: list[str]) -> dict[str, list[int]]:
    """Map each unique group to the list of system indices belonging to it."""
    mapping: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        mapping.setdefault(g, []).append(i)
    return mapping


# ---------------------------------------------------------------------------
# internal: manifest parsing
# ---------------------------------------------------------------------------


def _build_fold_groups(
    manifest_path: str,
) -> tuple[list[set[str]], set[str]]:
    """Parse a split_manifest.json into fold groups and test set.

    Returns
    -------
    folds : list[set[str]]
        One set of formula names per fold.
    test : set[str]
        Held-out test formulas (may be empty).
    """
    m = json.loads(Path(manifest_path).read_text())
    folds: list[set[str]] = []
    test: set[str] = set()

    for tag in ("co", "ni"):
        tag_data = m.get(tag, {})
        test.update(tag_data.get("test", []))
        parts = tag_data.get("parts", [])
        for i, part in enumerate(parts):
            if i >= len(folds):
                folds.append(set())
            folds[i].update(part)

    folds = [f for f in folds if f]
    return folds, test


# ---------------------------------------------------------------------------
# internal: sklearn head builder (delegates to shared factory)
# ---------------------------------------------------------------------------


def _build_sklearn_head(predictor_type: str, seed: int = 42) -> Any:
    """Map a predictor type string to an sklearn estimator.

    Delegates to ``dpa_adapt.utils.sklearn_heads.build_sklearn_head``.
    """
    from dpa_adapt.utils.sklearn_heads import (
        build_sklearn_head,
    )

    return build_sklearn_head(predictor_type, seed=seed)


# ---------------------------------------------------------------------------
# internal: per-system lazy assembly (avoids loading all descriptors at once)
# ---------------------------------------------------------------------------


def _load_system_labels(system: Any, label_key: str) -> np.ndarray:
    """Load labels for a single system, shape (n_frames, ...)."""
    resolved = _resolve_label_key(label_key)
    return np.asarray(system.data[resolved])


def _assemble_from_per_system_cache(
    systems: list,
    groups: list[str],
    selected_groups: set[str],
    label_key: str,
    granularity: str,
    pretrained: str,
    model_branch: str | None,
    pooling: str,
    type_map: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build X, y for systems whose group is in *selected_groups*.

    Reads one system's descriptors at a time from the per-system cache.
    Peak memory is proportional to the fold, not the full dataset.

    Parameters
    ----------
    systems : list[dpdata.System]
        All systems (same order as *groups*).
    groups : list[str]
        Group label per system.
    selected_groups : set[str]
        Which groups to include.
    label_key : str
        Label key in system data (e.g. ``"energies"``).
    granularity : str
        ``"frame"`` or ``"composition"``.
    pretrained : str
        Path to the pretrained model checkpoint.
    model_branch : str or None
        Model branch name for descriptor extraction.
    pooling : str
        Pooling strategy for descriptor aggregation.
    type_map : list[str] or tuple[str, ...] or None
        Optional type map for the system.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray (1D)
    """
    from dpa_adapt.data.desc_cache import (
        get_per_system_descriptor,
    )

    X_list, y_list = [], []

    for system, grp in zip(systems, groups, strict=True):
        if grp not in selected_groups:
            continue
        desc = get_per_system_descriptor(
            system,
            pretrained=pretrained,
            model_branch=model_branch,
            pooling=pooling,
            type_map=type_map,
        )  # (n_frames, feat_dim)
        lab = _load_system_labels(system, label_key)  # (n_frames, ...)
        if granularity == "composition":
            desc = desc.mean(axis=0, keepdims=True)
            lab = lab.mean(axis=0, keepdims=True)
        X_list.append(desc)
        y_list.append(lab)

    if not X_list:
        return np.empty((0, 0)), np.empty((0,))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).ravel()
    return X, y


# ---------------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------------


def train_test_split(
    systems: list,
    manifest: str | None = None,
    group_by: str | list[str] | None = None,
    test_size: float = 0.1,
    valid_size: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Split systems into train / valid / test, leak-proof by group.

    Exactly one of *manifest* or *group_by* must be provided.

    Parameters
    ----------
    systems : list
        dpdata systems (from ``load_data()`` or ``load_dataset()``).
    manifest : str, optional
        Path to a ``split_manifest.json``.  When provided, the splits are read
        from the manifest.
    group_by : str or list[str], optional
        ``"formula"`` — extract formula from each system's source path.
        ``list[str]`` — explicit group label per system (same length as
        *systems*).
    test_size : float
        Fraction of groups held out for test (ignored when *manifest* used).
    valid_size : float
        Fraction of remaining groups held out for validation.
    seed : int
        Random seed.

    Returns
    -------
    train, valid, test : list
        Three disjoint lists of systems.
    """
    n = len(systems)
    if n == 0:
        return [], [], []

    # --- manifest path ---
    if manifest is not None:
        folds, test_formulas = _build_fold_groups(manifest)
        if not folds:
            raise ValueError("Manifest contains no non-empty folds.")

        valid_formulas = folds[-1]
        train_formulas: set[str] = set()
        for f in folds[:-1]:
            train_formulas.update(f)

        grp = _formula_to_group(systems)
        train = [s for s, g in zip(systems, grp, strict=True) if g in train_formulas]
        valid = [s for s, g in zip(systems, grp, strict=True) if g in valid_formulas]
        test = [s for s, g in zip(systems, grp, strict=True) if g in test_formulas]
        return train, valid, test

    # --- group_by ---
    if group_by is None:
        raise ValueError(
            "Either manifest= or group_by= must be provided "
            "to ensure leak-proof splitting."
        )

    if isinstance(group_by, str) and group_by == "formula":
        groups = _formula_to_group(systems)
    elif isinstance(group_by, (list, tuple)):
        if len(group_by) != n:
            raise ValueError(
                f"group_by list length ({len(group_by)}) must match systems ({n})."
            )
        groups = list(group_by)
    else:
        raise ValueError(
            f"group_by must be 'formula' or a list of strings; got {group_by!r}."
        )

    unique_groups = sorted(set(groups))
    n_groups = len(unique_groups)
    if n_groups <= 1:
        raise ValueError(f"Only {n_groups} unique group(s) found; cannot split.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_groups)
    shuffled = [unique_groups[i] for i in perm]

    n_test = max(1, int(np.ceil(n_groups * test_size)))
    n_valid = max(1, int(np.ceil((n_groups - n_test) * valid_size)))

    test_groups = set(shuffled[:n_test])
    valid_groups = set(shuffled[n_test : n_test + n_valid])
    train_groups = set(shuffled[n_test + n_valid :])

    train = [s for s, g in zip(systems, groups, strict=True) if g in train_groups]
    valid = [s for s, g in zip(systems, groups, strict=True) if g in valid_groups]
    test = [s for s, g in zip(systems, groups, strict=True) if g in test_groups]

    return train, valid, test


# ---------------------------------------------------------------------------
# cross_validate
# ---------------------------------------------------------------------------


def cross_validate(
    model: Any,
    systems: list,
    label_key: str = "energy",
    cv: str | int = 5,
    group_by: str | list[str] | None = "formula",
    granularity: str = "frame",
    allow_expensive_cv: bool = False,
    min_groups_warn: int = 30,
    seed: int = 42,
    manifest: str | None = None,
) -> dict:
    """Leak-proof cross-validation for dpdata systems.

    For ``frozen_sklearn`` (the default code path for now), descriptors are
    extracted **once** and a cheap sklearn head is trained per fold — even
    ``cv=5`` completes in seconds.

    Training paradigms (``frozen_head`` / ``finetune`` / ``mft``)
    are expensive: each fold re-trains a full DeePMD model.  To prevent
    accidental hour-long runs, *allow_expensive_cv* must be explicitly set
    to ``True`` for those strategies when *cv* is an integer >= 2.  Otherwise
    a ``ValueError`` is raised.  Non-blocking warnings about estimated runtime
    are printed regardless.

    Parameters
    ----------
    model : DPAFineTuner
        Estimator instance with a ``strategy`` attribute.
    systems : list[str]
        Validated system directory paths.
    label_key : str
        Label filename under ``set.*/`` (default ``"energy"``).
    cv : str or int
        ``"holdout"`` — single train/valid split.  Training paradigms default
        to this.
        ``int >= 2`` — k-fold GroupKFold CV.  ``frozen_sklearn`` defaults to 5.
    group_by : str or list[str] or None
        ``"formula"`` (default) — extract formula from system path.
        ``list[str]`` — explicit groups.
        ``None`` — no grouping (random split; not recommended for small data).
    granularity : str
        ``"frame"`` (default) — one prediction per frame.
        ``"composition"`` — mean-pool descriptors and labels per formula,
        yielding one prediction per independent sample.
    allow_expensive_cv : bool
        Must be ``True`` to run k-fold CV on a training paradigm.  Ignored
        for ``frozen_sklearn``.
    min_groups_warn : int
        Emit a warning when the number of independent groups is below this
        threshold.  Default 30 is an empirical guideline (small-sample CV
        variance is large; see Hastie et al. ESL §7.10).  Set to 0 to disable.
    seed : int
        Random seed for sklearn heads.
    manifest : str, optional
        Path to a ``split_manifest.json``.  When provided, fold definitions
        are read from the manifest (deterministic, reproducible).  The *cv*
        parameter is ignored — the number of folds equals the number of parts
        in the manifest.  Test formulas in the manifest are excluded from CV.

    Returns
    -------
    dict
        Keys: ``train_mae``, ``test_mae``, ``test_rmse``, ``test_r2``,
        ``aggregate`` (mean/std dict), ``n_independent``, ``warnings``
        (list[str]), ``granularity``.
    """
    # ---- resolve strategy ----
    strategy = getattr(model, "strategy", "frozen_sklearn")
    is_cheap = strategy == "frozen_sklearn"

    if granularity not in ("frame", "composition"):
        raise ValueError(
            f"granularity must be 'frame' or 'composition'; got {granularity!r}."
        )

    # ---- resolve groups ----
    if group_by is None:
        groups = [f"sys_{i}" for i in range(len(systems))]
    elif isinstance(group_by, str) and group_by == "formula":
        groups = _formula_to_group(systems)
    elif isinstance(group_by, (list, tuple)):
        if len(group_by) != len(systems):
            raise ValueError(
                f"group_by list length ({len(group_by)}) must match "
                f"systems ({len(systems)})."
            )
        groups = list(group_by)
    else:
        raise ValueError(f"Invalid group_by: {group_by!r}")

    gmap = _group_indices(groups)
    unique_groups = sorted(gmap.keys())
    n_groups = len(unique_groups)

    # ---- resolve cv ----
    if cv == "holdout":
        n_splits = 1
    elif isinstance(cv, int) and cv >= 2:
        n_splits = cv
    else:
        raise ValueError(f"cv must be 'holdout' or an int >= 2; got {cv!r}.")

    # ---- expensive-cv guard (NON-interactive!) ----
    if not is_cheap and n_splits >= 2 and not allow_expensive_cv:
        raise ValueError(
            f"{strategy} {n_splits}-fold CV requires re-training the model "
            f"{n_splits} times, which may take hours on a single GPU. "
            f"Pass allow_expensive_cv=True to proceed, or use "
            f"cv='holdout' for a single train/valid split."
        )
    if not is_cheap and n_splits >= 2:
        _LOG.warning(
            "%s %d-fold CV will train %d models. "
            "Estimated %s. This is a non-blocking warning — training proceeds.",
            strategy,
            n_splits,
            n_splits,
            _estimate_runtime(strategy, n_splits),
        )

    # ---- build fold assignments ----
    fold_assignments: list[tuple[set[str], set[str]]] = []

    if manifest is not None:
        # Deterministic folds from split_manifest.json.
        # Each part is a validation fold; test formulas are excluded.
        manifest_folds, test_formulas = _build_fold_groups(manifest)
        if not manifest_folds:
            raise ValueError("Manifest contains no non-empty folds.")

        # Exclude test formulas from CV
        if test_formulas:
            _LOG.info(
                "Excluding %d test formula(s) from cross_validate: %s",
                len(test_formulas),
                sorted(test_formulas)[:10],
            )

        for fi, fold_formulas in enumerate(manifest_folds):
            val_groups = set(fold_formulas)
            train_groups: set[str] = set()
            for fj, other in enumerate(manifest_folds):
                if fj != fi:
                    train_groups.update(other)
            # Remove test formulas from both sides
            val_groups -= test_formulas
            train_groups -= test_formulas
            if val_groups and train_groups:
                fold_assignments.append((train_groups, val_groups))

        n_splits = len(fold_assignments)
    else:
        # Deterministic GroupKFold: sort groups, split by index (no shuffle).
        # Reproducible given the same set of systems and groups.
        groups_sorted = list(unique_groups)  # already sorted from dict keys

        if n_splits == 1:
            n_val = max(1, n_groups // 5)
            val_groups = set(groups_sorted[:n_val])
            train_groups = set(groups_sorted[n_val:])
            fold_assignments.append((train_groups, val_groups))
        else:
            fold_size = n_groups // n_splits
            for fi in range(n_splits):
                start = fi * fold_size
                end = start + fold_size if fi < n_splits - 1 else n_groups
                val_groups = set(groups_sorted[start:end])
                train_groups = set(groups_sorted[:start]) | set(groups_sorted[end:])
                fold_assignments.append((train_groups, val_groups))

    # ---- ensure per-system descriptor cache (once, lazy) ----
    # This reuses existing desc_mean.npy when present, extracts only missing
    # systems one-by-one.  Peak memory is one system's descriptors at a time.
    if is_cheap:
        from dpa_adapt.data.desc_cache import (
            ensure_per_system_cache,
        )

        ensure_per_system_cache(
            systems,
            pretrained=model.pretrained,
            model_branch=model.model_branch,
            pooling=model.pooling,
            type_map=getattr(model, "type_map", None),
        )

    # ---- per-fold loop (reads per-system cache on demand) ----
    train_mae_list, test_mae_list = [], []
    test_rmse_list, test_r2_list = [], []

    for train_groups, val_groups in fold_assignments:
        if is_cheap:
            Xtr, ytr = _assemble_from_per_system_cache(
                systems,
                groups,
                train_groups,
                label_key,
                granularity,
                pretrained=model.pretrained,
                model_branch=model.model_branch,
                pooling=model.pooling,
                type_map=getattr(model, "type_map", None),
            )
            Xva, yva = _assemble_from_per_system_cache(
                systems,
                groups,
                val_groups,
                label_key,
                granularity,
                pretrained=model.pretrained,
                model_branch=model.model_branch,
                pooling=model.pooling,
                type_map=getattr(model, "type_map", None),
            )
            if Xtr.shape[0] == 0 or Xva.shape[0] == 0:
                continue

            predictor_type = getattr(model, "_predictor_type", None)
            if predictor_type is None:
                predictor_type = getattr(model, "predictor", "linear")
                # Map the public API name to the internal _predictor_type
                if predictor_type == "ridge":
                    predictor_type = "linear"
            head = make_pipeline(
                StandardScaler(),
                _build_sklearn_head(predictor_type, seed=seed),
            )
            head.fit(Xtr, ytr)

            pred_tr = head.predict(Xtr)
            pred_va = head.predict(Xva)

            train_mae_list.append(float(np.mean(np.abs(pred_tr - ytr))))
            test_mae_list.append(float(np.mean(np.abs(pred_va - yva))))
            test_rmse_list.append(float(np.sqrt(np.mean((pred_va - yva) ** 2))))
            if len(yva) >= 3:
                ss_res = np.sum((pred_va - yva) ** 2)
                ss_tot = np.sum((yva - yva.mean()) ** 2)
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            else:
                r2 = float("nan")
            test_r2_list.append(r2)

            # Release fold arrays before the next fold
            del Xtr, ytr, Xva, yva, pred_tr, pred_va
        else:
            # Training paradigms — delegate to per-fold fit/evaluate.
            # Phase 2 will wire this to DPATrainer / MFTFineTuner.
            raise NotImplementedError(
                "cross_validate for training paradigms "
                "(frozen_head / finetune / mft) is not yet "
                "implemented. Use frozen_sklearn for now."
            )

    # ---- warnings ----
    warnings: list[str] = []
    if min_groups_warn > 0 and n_groups < min_groups_warn:
        warnings.append(
            f"Only {n_groups} independent groups; CV metrics have high "
            f"variance. Report per-fold values, not just mean ± std. "
            f"(min_groups_warn={min_groups_warn}, set to 0 to suppress)"
        )
    if granularity == "frame" and n_groups < 100:
        warnings.append(
            "granularity='frame': labels repeat within each group. "
            "n_independent is the true sample size."
        )

    # ---- aggregate ----
    agg = {}
    for name, lst in [
        ("mae", test_mae_list),
        ("rmse", test_rmse_list),
        ("r2", test_r2_list),
    ]:
        vals = [v for v in lst if not np.isnan(v)]
        if vals:
            agg[f"{name}_mean"] = float(np.mean(vals))
            agg[f"{name}_std"] = float(np.std(vals))

    return {
        "train_mae": train_mae_list,
        "test_mae": test_mae_list,
        "test_rmse": test_rmse_list,
        "test_r2": test_r2_list,
        "aggregate": agg,
        "n_independent": n_groups,
        "warnings": warnings,
        "granularity": granularity,
    }


# ---------------------------------------------------------------------------
# internal: runtime estimate
# ---------------------------------------------------------------------------


def _estimate_runtime(strategy: str, n_splits: int) -> str:
    per_run = {
        "frozen_head": "~5-15 min/run",
        "finetune": "~10-30 min/run",
        "mft": "~20-60 min/run",
    }.get(strategy, "unknown")
    return f"{n_splits} x {per_run}"
