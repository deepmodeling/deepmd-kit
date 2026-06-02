# dpa_tools/trainer.py
"""
DPATrainer: drives ``dp --pt train`` for Scratch / FT / LP adaptation modes,
mirroring the comparison setup of arXiv:2601.08486 (Table 3 / Fig 2).

Mode is selected by constructor arguments:

| Mode    | ``pretrained``   | ``freeze_backbone`` |
| ------- | ---------------- | ------------------- |
| Scratch | ``None``         | ``False``           |
| FT      | path to ckpt     | ``False``           |
| LP      | path to ckpt     | ``True``            |

MFT lives in :class:`dpa_tools.mft.MFTFineTuner`; the sklearn-head Path B
lives in :class:`dpa_tools.finetuner.DPAFineTuner`.
"""

from __future__ import annotations

import copy
import glob as _glob
import json
import logging
import os
import re
import subprocess
from typing import Optional, Union

_LOG = logging.getLogger("dpa_tools.trainer")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fallback descriptor config used when pretrained=None (Scratch mode).
# Must match DPA-3.1-3M exactly. Source: ckpt _extra_state.model_params.shared_dict.
DPA3_DESCRIPTOR_DEFAULT = {
    "type": "dpa3",
    "repflow": {
        "n_dim": 128, "e_dim": 64, "a_dim": 32, "nlayers": 16,
        "e_rcut": 6.0, "e_rcut_smth": 5.3, "e_sel": 1200,
        "a_rcut": 4.0, "a_rcut_smth": 3.5, "a_sel": 300,
        "axis_neuron": 4, "skip_stat": True,
        "a_compress_rate": 1, "a_compress_e_rate": 2,
        "a_compress_use_split": True,
        "update_angle": True, "smooth_edge_update": True,
        "use_dynamic_sel": True, "sel_reduce_factor": 10.0,
        "update_style": "res_residual",
        "update_residual": 0.1, "update_residual_init": "const",
        "n_multi_edge_message": 1, "optim_update": True,
        "use_exp_switch": True,
        "fix_stat_std": 0.3,
    },
    # Paper qm9_gap input.json uses "silut:3.0" (alias of "custom_silu:3.0";
    # verified identical output in deepmd-kit 3.1.3).
    "activation_function": "silut:3.0",
    "precision": "float32",
    "use_tebd_bias": False,
    "concat_output_tebd": False,
    "exclude_types": [],
    "env_protection": 0.0,
    "trainable": True,
    "use_econf_tebd": False,
}

DEFAULT_FITTING_NET = {
    "type": "property",
    "neuron": [240, 240, 240],
    "activation_function": "tanh",   # paper Table 8
    "resnet_dt": True,
    "precision": "float32",
}

_VALID_LOSSES = ("mse", "smooth_mae")


# ---------------------------------------------------------------------------
# DPATrainer
# ---------------------------------------------------------------------------

class DPATrainer:
    """
    Drive ``dp --pt train`` for Scratch / FT / LP downstream adaptation.

    Parameters
    ----------
    pretrained : str or None
        Path to a DPA pretrained checkpoint (.pt). ``None`` means Scratch.
    init_branch : str
        Branch name in the checkpoint used to initialize the descriptor.
        Only consulted when ``pretrained`` is given.
    freeze_backbone : bool
        If True, freeze the descriptor (LP mode). Requires ``pretrained``.
    property_name : str
        Name of the property npy file under ``set.000/`` (e.g. ``"homo"``).
        Must be a valid Python identifier.
    task_dim : int
        Output dimensionality of the property head. Must be ``>= 1``.
    intensive : bool
        Whether the property is intensive (mean-pool) or extensive (sum).
    train_systems, valid_systems : str or list[str]
        Globs (or list of globs) resolving to deepmd/npy system directories.
        Both required.
    type_map : list[str]
        Element symbols. Required; no auto-inference.
    fitting_net_params : dict, optional
        Overrides for the property head config (shallow-merged onto the
        defaults). The defaults are ``DEFAULT_FITTING_NET`` plus
        ``property_name``, ``task_dim``, ``intensive``, ``seed``.
    learning_rate, stop_lr : float
        Exp-decay LR endpoints.
    max_steps : int
        Total training steps.
    batch_size : str or int
        DeepMD-kit batch_size spec (e.g. ``"auto:512"``).
    loss_function : str
        ``"mse"`` or ``"smooth_mae"``.
    seed : int
        Random seed.
    output_dir : str
        Directory for checkpoints, input.json, and manifests.
    save_freq, disp_freq : int
        DeepMD-kit save/display intervals.
    """

    def __init__(
        self,
        # ---- pretraining / freezing ----
        pretrained: Optional[str] = None,
        init_branch: str = "SPICE2",
        freeze_backbone: bool = False,
        # ---- downstream task ----
        property_name: str = "homo",
        task_dim: int = 1,
        intensive: bool = True,
        # ---- data ----
        train_systems: Union[str, list, None] = None,
        valid_systems: Union[str, list, None] = None,
        type_map: Optional[list] = None,
        # ---- model overrides ----
        fitting_net_params: Optional[dict] = None,
        # ---- training ----
        learning_rate: float = 1e-3,
        stop_lr: float = 1e-5,
        max_steps: int = 100_000,
        batch_size: Union[str, int] = "auto:512",
        loss_function: str = "mse",
        seed: int = 42,
        # ---- output ----
        output_dir: str = "./dpa_output",
        save_freq: int = 10_000,
        disp_freq: int = 1_000,
    ):
        # ---- validation ----
        if train_systems is None:
            raise ValueError("train_systems is required (got None).")
        if valid_systems is None:
            raise ValueError("valid_systems is required (got None).")
        if type_map is None:
            raise ValueError(
                "type_map is required. Pass an explicit list of element "
                "symbols (e.g. the SPICE2 full periodic table). "
                "Auto-inference is intentionally not supported."
            )
        if not isinstance(type_map, list) or not all(isinstance(x, str) for x in type_map):
            raise ValueError("type_map must be a list of element symbol strings.")
        if freeze_backbone and pretrained is None:
            raise ValueError(
                "LP requires a pretrained checkpoint to freeze. "
                "Set freeze_backbone=False for Scratch, or pass a pretrained ckpt."
            )
        if pretrained is not None and not os.path.isfile(pretrained):
            raise ValueError(
                f"pretrained checkpoint not found: {pretrained!r}."
            )
        if not isinstance(property_name, str) or not property_name.isidentifier():
            raise ValueError(
                f"property_name must be a valid Python identifier "
                f"(no spaces or slashes); got {property_name!r}."
            )
        if not isinstance(task_dim, int) or task_dim < 1:
            raise ValueError(f"task_dim must be an int >= 1; got {task_dim!r}.")
        if loss_function not in _VALID_LOSSES:
            raise ValueError(
                f"loss_function must be one of {_VALID_LOSSES}; "
                f"got {loss_function!r}."
            )

        self.pretrained = pretrained
        self.init_branch = init_branch
        self.freeze_backbone = freeze_backbone
        self.property_name = property_name
        self.task_dim = task_dim
        self.intensive = intensive
        self.train_systems = train_systems
        self.valid_systems = valid_systems
        self.type_map = type_map
        self.fitting_net_params = fitting_net_params
        self.learning_rate = learning_rate
        self.stop_lr = stop_lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.seed = seed
        self.output_dir = output_dir
        self.save_freq = save_freq
        self.disp_freq = disp_freq

    # ----- mode label (debugging convenience) -----
    @property
    def mode(self) -> str:
        if self.pretrained is None:
            return "Scratch"
        return "LP" if self.freeze_backbone else "FT"

    # ----- descriptor sourcing -----
    def _read_descriptor_from_ckpt(self) -> dict:
        import torch

        sd = torch.load(self.pretrained, map_location="cpu", weights_only=False)
        try:
            descriptor = (
                sd["model"]["_extra_state"]["model_params"]
                ["shared_dict"]["dpa3_descriptor"]
            )
        except (KeyError, TypeError) as e:
            raise RuntimeError(
                f"Could not locate dpa3_descriptor in checkpoint {self.pretrained}: "
                f"missing key {e!r}. Expected path sd['model']['_extra_state']"
                "['model_params']['shared_dict']['dpa3_descriptor']."
            ) from e
        return copy.deepcopy(descriptor)

    def _get_descriptor(self) -> dict:
        if self.pretrained is not None:
            descriptor = self._read_descriptor_from_ckpt()
        else:
            descriptor = copy.deepcopy(DPA3_DESCRIPTOR_DEFAULT)
        # Paper alignment (qm9_gap input.json): silut:3.0 activation (alias of
        # the ckpt's custom_silu:3.0) + explicit fix_stat_std=0.3. Enforced on
        # both the ckpt-read and scratch paths so the emitted JSON matches the
        # paper repo verbatim.
        descriptor["activation_function"] = "silut:3.0"
        descriptor["repflow"]["fix_stat_std"] = 0.3
        # LP: freeze the descriptor by setting trainable=False on the descriptor
        # block. DeepMD-kit 3.1.3 honors this field in the `--finetune` code path
        # (verified by reading deepmd.pt.train.training; the descriptor's
        # `requires_grad_` is set from this flag at init). If a future deepmd-kit
        # version changes this, switch to passing `--freeze-descriptor` to the
        # CLI or use `dp --pt freeze` as a post-processing step.
        descriptor["trainable"] = not self.freeze_backbone
        return descriptor

    # ----- glob expansion -----
    @staticmethod
    def _expand_systems(spec, label: str) -> list:
        if isinstance(spec, str):
            patterns = [spec]
        else:
            patterns = list(spec)
        resolved: list = []
        for pat in patterns:
            matches = sorted(_glob.glob(pat))
            resolved.extend(matches)
        # de-duplicate while preserving order
        seen = set()
        unique = []
        for p in resolved:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        if not unique:
            raise ValueError(
                f"{label} resolved to 0 systems from patterns={patterns!r}. "
                f"Check the glob and that the directories exist."
            )
        if len(unique) < 50:
            _LOG.warning(
                "%s resolved to only %d systems (patterns=%r). "
                "MFT-paper BOOM splits typically yield 500/300 for train/valid.",
                label, len(unique), patterns,
            )
        return unique

    # ----- config build -----
    def _build_fitting_net(self) -> dict:
        fn = copy.deepcopy(DEFAULT_FITTING_NET)
        fn.update({
            "property_name": self.property_name,
            "task_dim": self.task_dim,
            "intensive": self.intensive,
            # verified: deepmd.utils.argcheck.fitting_property() accepts seed
            # (inspect.getsource shows Argument("seed", [int, None], optional=True))
            "seed": self.seed,
        })
        # NB: dim_case_embd is intentionally NOT injected for FT/LP. The paper
        # qm9_gap input.json omits it: single-task `--finetune` (without
        # --model-branch) copies only the backbone and random-inits the
        # property head at [128, 240], so there is no [159, 240] checkpoint
        # head to size-match against. An explicit user value still wins.
        if self.fitting_net_params:
            fn.update(self.fitting_net_params)
        return fn

    def _build_config(self) -> dict:
        # Seed propagation in DeepMD-kit v3.1.3 (deepmd/utils/argcheck.py):
        #   - model.descriptor.seed   verified: descrpt_dpa3_args() L1428
        #   - model.fitting_net.seed  verified: fitting_property() L1966
        #   - training.seed           verified: training_args() L3856
        # A top-level "seed" was previously added as a "v0/v1 compat default"
        # but deepmd 3.1.3 dargs is strict-mode and rejects unknown root keys
        # (ArgumentKeyError at root location). Do NOT re-add it.
        train_sys = self._expand_systems(self.train_systems, "train_systems")
        valid_sys = self._expand_systems(self.valid_systems, "valid_systems")
        self._resolved_train_systems = train_sys
        self._resolved_valid_systems = valid_sys

        descriptor = self._get_descriptor()
        descriptor["seed"] = self.seed  # verified: descrpt_dpa3_args (deepmd v3.1.3)
        fitting_net = self._build_fitting_net()

        return {
            "model": {
                "type_map": self.type_map,
                "descriptor": descriptor,
                "fitting_net": fitting_net,
            },
            "loss": {
                "type": "property",
                "loss_func": self.loss_function,
                "metric": ["mae", "rmse"],
            },
            "learning_rate": {
                "type": "exp",
                "start_lr": self.learning_rate,
                "stop_lr": self.stop_lr,
                # Paper qm9_gap: decay_steps=1000 (we previously used 5000).
                "decay_steps": 1000,
            },
            "training": {
                "training_data": {
                    "systems": train_sys,
                    "batch_size": self.batch_size,
                },
                "validation_data": {
                    "systems": valid_sys,
                    "batch_size": self.batch_size,
                },
                "numb_steps": self.max_steps,
                "seed": self.seed,  # verified: training_args (deepmd v3.1.3)
                # Paper qm9_gap: gradient_max_norm=5.0 (gradient clipping).
                "gradient_max_norm": 5.0,
                "disp_freq": self.disp_freq,
                "save_freq": self.save_freq,
                # Absolute path so checkpoints land in output_dir without
                # depending on the caller's cwd (we no longer pass --output).
                "save_ckpt": os.path.join(self.output_dir, "model.ckpt"),
            },
        }

    # ----- CLI build -----
    def _build_cmd(self, input_json: str) -> list:
        # Paper qm9_gap uses `dp --pt train <json> --finetune <ckpt>` with NO
        # --model-branch: single-task fine-tune copies the backbone and
        # random-inits the property head. Passing --model-branch would try to
        # copy a branch's [159, 240] property head and fail with a size
        # mismatch. `--skip-neighbor-stat` is kept (paper omits it, but our
        # data-stat pass is too slow); deepmd honors `training.save_ckpt` from
        # the JSON so no `--output` flag is needed.
        cmd = ["dp", "--pt", "train", input_json]
        cmd += ["--skip-neighbor-stat"]
        if self.pretrained is not None:
            cmd += ["--finetune", self.pretrained]
        return cmd

    # ----- checkpoint discovery -----
    def _find_latest_checkpoint(self) -> tuple:
        """
        Return ``(Path | None, int)`` for the checkpoint with the largest
        step in ``output_dir``, or ``(None, 0)`` if none exist.
        """
        from pathlib import Path
        ckpts = list(Path(self.output_dir).glob("model.ckpt-*.pt"))
        if not ckpts:
            return None, 0

        def step_of(p):
            return int(p.stem.split("-")[-1])

        latest = max(ckpts, key=step_of)
        return latest, step_of(latest)

    def _final_ckpt_path(self) -> Optional[str]:
        latest, _ = self._find_latest_checkpoint()
        return str(latest) if latest is not None else None

    # ----- fit -----
    def fit(self) -> str:
        """
        Run ``dp --pt train``.

        Returns
        -------
        str
            Path to the final ``model.ckpt-<step>.pt``.

        Notes
        -----
        Idempotency: training is skipped if a checkpoint at step
        ``>= max_steps`` exists in ``output_dir``. If ``max_steps`` is
        increased between runs (i.e. only a shorter checkpoint exists),
        training is restarted from scratch (or from ``pretrained``) —
        checkpoint resumption is not supported.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        latest, step = self._find_latest_checkpoint()
        if latest is not None and step >= self.max_steps:
            _LOG.info(
                "Skipping training: found %s (step %d) >= max_steps=%d",
                latest, step, self.max_steps,
            )
            return str(latest)

        config = self._build_config()
        input_json = os.path.join(self.output_dir, "input.json")
        with open(input_json, "w") as f:
            json.dump(config, f, indent=2)

        manifest_train = os.path.join(self.output_dir, "manifest_train.txt")
        with open(manifest_train, "w") as f:
            f.write("\n".join(self._resolved_train_systems) + "\n")
        manifest_valid = os.path.join(self.output_dir, "manifest_valid.txt")
        with open(manifest_valid, "w") as f:
            f.write("\n".join(self._resolved_valid_systems) + "\n")

        cmd = self._build_cmd(input_json)
        # fit() deliberately echoes the CLI so the user can rerun it manually.
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        ckpt = self._final_ckpt_path()
        if ckpt is None:
            raise RuntimeError(
                f"Training finished but no model.ckpt-*.pt was found in "
                f"{self.output_dir}."
            )
        return ckpt

    # ----- evaluate -----
    def evaluate(self, test_systems: Union[str, list]) -> dict:
        """
        Run ``dp --pt test`` on the trained checkpoint.

        Parameters
        ----------
        test_systems : str or list[str]
            Glob (or list of globs) resolving to deepmd/npy system dirs.

        Returns
        -------
        dict
            ``{'rmse': float, 'mae': float, 'n_frames': int, 'n_systems': int,
            '_raw_stdout': str, '_parser_pattern_used': str}``.
            Raises ``RuntimeError`` if neither RMSE nor MAE can be parsed.

        Notes
        -----
        Uses ``dp --pt test -f <datafile>`` (single-value flag taking a path
        to a file listing one system per line). Previously used multiple
        ``-s`` flags, but argparse honored only the last one and the parser
        silently succeeded with a single-system result.
        """
        ckpt = self._final_ckpt_path()
        if ckpt is None:
            raise RuntimeError(
                f"No checkpoint found in {self.output_dir}; call fit() first."
            )
        systems = self._expand_systems(test_systems, "test_systems")

        # Write the resolved system paths to a datafile and pass via -f.
        # This is dp --pt test's native multi-system input mode (see
        # `dp --pt test --help`).
        os.makedirs(self.output_dir, exist_ok=True)
        datafile = os.path.join(self.output_dir, "test_systems.txt")
        with open(datafile, "w") as f:
            f.write("\n".join(systems) + "\n")

        cmd = ["dp", "--pt", "test", "-m", ckpt, "-f", datafile, "-n", "999999"]
        _LOG.info(
            "Running: %s  (with %d systems listed in %s)",
            " ".join(cmd), len(systems), datafile,
        )

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # DeepMD-kit logs PROPERTY MAE/RMSE to stderr (Python logging default).
        # Feed both streams to the parser.
        combined = result.stdout + "\n" + result.stderr

        parsed = self._parse_test_output(combined)

        # Sanity check: extract the "# number of systems : N" line and verify
        # it matches our resolved list.
        n_sys_match = re.search(
            r"number of systems\s*[:=]?\s*(\d+)", combined, re.IGNORECASE
        )
        if n_sys_match:
            n_found = int(n_sys_match.group(1))
            parsed["n_systems"] = n_found
            if n_found != len(systems):
                _LOG.warning(
                    "dp test reports %d systems but %d were resolved; "
                    "some systems may have been skipped (missing labels?)",
                    n_found, len(systems),
                )
        else:
            parsed["n_systems"] = 0
            _LOG.warning(
                "Could not extract 'number of systems' from dp test output; "
                "inspect _raw_stdout."
            )

        return parsed

    # ----- test-output parsing -----
    # Calibrated against real deepmd-kit 3.1.3 `dp --pt test` stderr (property
    # task). Sample line: "PROPERTY RMSE           : 6.065579e-02 units"
    # The output appears twice — once per system, once in "weighted average of
    # errors" — so the parser uses findall and takes the LAST match (Fix 3).
    #
    # Refactored: replaced fragile multi-pattern regex fallback chain with a
    # single well-anchored regex per metric type, auto-detected from the output.
    # Generic \brmse\b / \bmae\b fallback patterns removed; unparseable output
    # now raises RuntimeError with the last 50 lines of stdout+stderr.
    _PROPERTY_RMSE_RE = re.compile(
        r"PROPERTY\s+RMSE\s+:\s*([0-9eE.+-]+)", re.IGNORECASE
    )
    _PROPERTY_MAE_RE = re.compile(
        r"PROPERTY\s+MAE\s+:\s*([0-9eE.+-]+)", re.IGNORECASE
    )
    _ENERGY_RMSE_RE = re.compile(
        r"Energy\s+RMSE\s+:\s*([0-9eE.+-]+)\s*\S+", re.IGNORECASE
    )
    _ENERGY_MAE_RE = re.compile(
        r"Energy\s+MAE\s+:\s*([0-9eE.+-]+)\s*\S+", re.IGNORECASE
    )
    _N_FRAMES_PATTERNS = [
        re.compile(r"number of test data\s*[:=]?\s*(\d+)", re.IGNORECASE),
        re.compile(r"#\s*of test data\s*[:=]?\s*(\d+)", re.IGNORECASE),
        re.compile(r"\bn_frames\b\s*[:=]?\s*(\d+)", re.IGNORECASE),
    ]

    @classmethod
    def _parse_test_output(cls, stdout: str) -> dict:
        """
        Extract ``rmse``, ``mae``, ``n_frames`` from ``dp --pt test`` stdout.

        Auto-detects output format — ``PROPERTY MAE`` / ``PROPERTY RMSE`` for
        property tasks, ``Energy MAE`` / ``Energy RMSE`` for ener tasks —
        and applies a single well-anchored regex per metric type.  No generic
        fallback patterns are used; if parsing fails a ``RuntimeError`` is
        raised with the last 50 lines of the combined output.

        Refactored: replaced fragile multi-pattern regex fallback chain with
        format-aware, single-pattern-per-metric parsing.
        """
        # Auto-detect output format from the presence of known metric labels.
        if "PROPERTY MAE" in stdout or "PROPERTY RMSE" in stdout:
            mae_re = cls._PROPERTY_MAE_RE
            rmse_re = cls._PROPERTY_RMSE_RE
            tag = "PROPERTY"
        elif "Energy MAE" in stdout or "Energy RMSE" in stdout:
            mae_re = cls._ENERGY_MAE_RE
            rmse_re = cls._ENERGY_RMSE_RE
            tag = "Energy"
        else:
            tail = "\n".join(stdout.splitlines()[-50:])
            raise RuntimeError(
                "Could not parse MAE or RMSE from `dp --pt test` output. "
                "No PROPERTY MAE/RMSE or Energy MAE/RMSE lines found.\n"
                "----- last 50 lines of combined stdout+stderr -----\n"
                f"{tail}\n"
                "----------------------"
            )

        # Take the LAST match. dp --pt test prints per-system errors followed by
        # a "weighted average of errors" block; the weighted average is what we
        # want when multiple systems are evaluated together. For a single-system
        # test, the per-system and weighted lines have the same value.
        mae_matches = mae_re.findall(stdout)
        rmse_matches = rmse_re.findall(stdout)

        if not mae_matches and not rmse_matches:
            tail = "\n".join(stdout.splitlines()[-50:])
            raise RuntimeError(
                f"Detected {tag} output format but could not extract numeric "
                "MAE or RMSE values.\n"
                "----- last 50 lines of combined stdout+stderr -----\n"
                f"{tail}\n"
                "----------------------"
            )

        mae = float(mae_matches[-1]) if mae_matches else float("nan")
        rmse = float(rmse_matches[-1]) if rmse_matches else float("nan")

        # TODO: for the total across systems we'd need to sum all matches;
        # here we take the last (per-system) match. `n_frames` is currently
        # only used for logging, so this approximation is acceptable.
        n_frames = 0
        for pat in cls._N_FRAMES_PATTERNS:
            matches = pat.findall(stdout)
            if matches:
                n_frames = int(matches[-1])
                break

        pattern_used = f"{tag} MAE (last); {tag} RMSE (last)"
        return {
            "rmse": rmse,
            "mae": mae,
            "n_frames": n_frames,
            "_raw_stdout": stdout,
            "_parser_pattern_used": pattern_used,
        }
