import glob as _glob
import os
import re
import subprocess
import sys


class MFTFineTuner:
    """
    Multi-task fine-tuning via dp --pt train.

    Jointly optimizes a downstream property head and an aux force-field head
    on a shared DPA descriptor, preventing representation collapse (per
    arXiv:2601.08486).

    Refactored: ``fitting_net_params`` is now lazily resolved from the
    checkpoint on first access rather than eagerly in ``__init__``, so
    constructing an ``MFTFineTuner`` no longer triggers ``torch.load``
    unless ``fit()`` (or any other accessor) actually needs the value.

    Parameters
    ----------
    pretrained : str
        Path to the DPA pretrained checkpoint (.pt).
    aux_branch : str
        Branch name in the checkpoint to initialize the aux head.
        Default: 'MP_traj_v024_alldata_mixu' (general materials coverage).
        Run `dp --pt show <checkpoint> model-branch` to list all options.
    aux_prob : float
        Sampling weight for the aux branch. Positive real number; DeepMD-kit
        normalizes it against DOWNSTREAM weight of 1.0. This is the primary
        experimental variable for sensitivity analysis.
        Example: aux_prob=0.5 → aux:downstream ≈ 1:2 sampling ratio.
    aux_type_map : list[str]
        Element symbols for the aux data directory.
    downstream_type_map : list[str]
        Element symbols for the downstream data directory.
    fitting_net_params : dict, optional
        Fitting net architecture for the aux branch. Must match the
        checkpoint exactly. When omitted (the default), it is read
        automatically from the pretrained checkpoint at
        ``sd['model']['_extra_state']['model_params']['model_dict'][aux_branch]['fitting_net']``.
        Pass an explicit dict only if you need to override the checkpoint's
        config (e.g. for experiments).
    downstream_task_type : str
        Either ``"ener"`` (force-field head, the legacy default) or
        ``"property"`` (intensive scalar head, e.g. HOMO/LUMO). Selects how
        the DOWNSTREAM branch's fitting_net and loss are built:

        * ``"ener"`` — DOWNSTREAM reuses the aux fitting_net dict and an
          ener-style loss with force/virial prefs. This is what the
          mp_data sensitivity-analysis MFT experiments rely on.
        * ``"property"`` — DOWNSTREAM gets a fresh ``type: property``
          fitting_net (using ``property_name``, ``task_dim``, ``intensive``)
          and a property-style MSE loss with no force/virial prefs. This
          is what arXiv:2601.08486 Table 3 / Fig 2 reports for HOMO/LUMO.
        Required for paper-faithful BOOM evaluation on QM9. Default
        ``"ener"`` preserves back-compat with existing sensitivity-analysis
        callers.
    property_name : str, optional
        Required when ``downstream_task_type="property"``. Name of the
        per-system property file (e.g. ``"homo"`` reads ``set.*/homo.npy``).
        Must be a valid Python identifier.
    task_dim : int
        Output dimensionality of the property head. Default ``1``.
    intensive : bool
        Whether the property is intensive (mean-pool) or extensive (sum).
        Default ``True`` (correct for HOMO/LUMO and most molecular
        properties).
    learning_rate : float
        Initial learning rate.
    stop_lr : float
        Final learning rate.
    max_steps : int
        Total training steps.
    batch_size : str | int
        Batch size (e.g. "auto:32" or 32).
    seed : int
        Random seed.
    output_dir : str
        Directory for checkpoints and logs.
    save_freq : int
        Checkpoint save interval (steps).
    disp_freq : int
        Log display interval (steps).
    """

    def __init__(
        self,
        pretrained,
        aux_branch="MP_traj_v024_alldata_mixu",
        aux_prob=0.5,
        aux_type_map=None,
        downstream_type_map=None,
        fitting_net_params=None,
        downstream_task_type="ener",
        property_name=None,
        task_dim=1,
        intensive=True,
        learning_rate=1e-3,
        stop_lr=1e-5,
        max_steps=50000,
        batch_size="auto:32",
        aux_batch_size=None,
        downstream_batch_size=None,
        seed=42,
        fparam_dim: int = 0,
        output_dir="./mft_output",
        save_freq=10000,
        disp_freq=1000,
    ):
        if downstream_task_type not in ("ener", "property"):
            raise ValueError(
                f"downstream_task_type must be 'ener' or 'property'; "
                f"got {downstream_task_type!r}."
            )
        if downstream_task_type == "property":
            if not isinstance(property_name, str) or not property_name.isidentifier():
                raise ValueError(
                    "property_name is required when "
                    "downstream_task_type='property' and must be a valid "
                    f"Python identifier; got {property_name!r}."
                )
            if not isinstance(task_dim, int) or task_dim < 1:
                raise ValueError(
                    f"task_dim must be an int >= 1; got {task_dim!r}."
                )
        if not isinstance(fparam_dim, int) or fparam_dim < 0:
            raise ValueError(
                f"fparam_dim must be a non-negative int; got {fparam_dim!r}."
            )

        self.pretrained = pretrained
        self.aux_branch = aux_branch
        self.aux_prob = aux_prob
        self.aux_type_map = aux_type_map
        self.downstream_type_map = downstream_type_map
        # Lazy: only load from ckpt when fitting_net_params is first accessed.
        self._fitting_net_params = fitting_net_params
        self._fitting_net_params_resolved = (fitting_net_params is not None)
        self.downstream_task_type = downstream_task_type
        self.property_name = property_name
        self.task_dim = task_dim
        self.intensive = intensive
        self.learning_rate = learning_rate
        self.stop_lr = stop_lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.aux_batch_size = aux_batch_size
        self.downstream_batch_size = downstream_batch_size
        self.seed = seed
        self.fparam_dim = fparam_dim
        self.output_dir = output_dir
        self.save_freq = save_freq
        self.disp_freq = disp_freq

        # populated by fit()
        self.train_data = None
        self.aux_data = None
        self.valid_data = None

    # ------------------------------------------------------------------
    # Lazy fitting_net_params resolution
    #
    # Refactored: torch.load is deferred from __init__ to first access
    # so that constructing an MFTFineTuner is cheap.  The checkpoint is
    # only read when fit() (via MFTConfigManager) or user code accesses
    # fitting_net_params and the value was not explicitly provided.
    # ------------------------------------------------------------------

    @property
    def fitting_net_params(self):
        if (
            self._fitting_net_params is None
            and not self._fitting_net_params_resolved
        ):
            self._fitting_net_params = self._read_fitting_net_from_ckpt(
                self.pretrained, self.aux_branch
            )
            self._fitting_net_params_resolved = True
        return self._fitting_net_params

    @fitting_net_params.setter
    def fitting_net_params(self, value):
        self._fitting_net_params = value

    @staticmethod
    def _read_fitting_net_from_ckpt(pretrained, aux_branch):
        """
        Pull fitting_net config for ``aux_branch`` out of a DPA multi-task
        checkpoint. Raises ValueError listing available branches if
        ``aux_branch`` isn't present.
        """
        import torch

        sd = torch.load(pretrained, map_location="cpu", weights_only=False)
        try:
            model_dict = sd["model"]["_extra_state"]["model_params"]["model_dict"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(
                f"Could not locate model_dict in checkpoint {pretrained}: "
                f"missing key {e!r}. Expected path "
                "sd['model']['_extra_state']['model_params']['model_dict']."
            ) from e
        if aux_branch not in model_dict:
            available = sorted(model_dict.keys())
            raise ValueError(
                f"aux_branch {aux_branch!r} not found in checkpoint {pretrained}. "
                f"Available branches: {available}. "
                f"Run `dp --pt show {pretrained} model-branch` to inspect."
            )
        return model_dict[aux_branch]["fitting_net"]

    def _resolve_type_maps(self, train_data, aux_data):
        """Auto-infer aux_type_map from checkpoint and validate data type_maps.

        Called by fit() when the user has not explicitly provided aux_type_map
        or downstream_type_map.  Reads the checkpoint's global type_map (118
        elements for DPA-3.1-3M), validates that each dataset's elements are
        a subset, and sets ``self.aux_type_map`` and
        ``self.downstream_type_map``.
        """
        from dpa_adapt.data.loader import load_data
        from dpa_adapt.data.type_map import (
            read_checkpoint_type_map,
            read_data_type_map_union,
            validate_type_map_subset,
        )

        self.aux_type_map = read_checkpoint_type_map(
            self.pretrained, branch=self.aux_branch,
        )

        try:
            train_systems = load_data(train_data)
        except Exception:
            train_systems = []
        try:
            aux_systems = load_data(aux_data)
        except Exception:
            aux_systems = []

        for label, systems in [
            ("downstream", train_systems),
            ("aux", aux_systems),
        ]:
            if not systems:
                continue
            try:
                elements = read_data_type_map_union(systems)
            except ValueError:
                continue  # no atom_names — deepmd uses raw atom indices
            validate_type_map_subset(
                elements, self.aux_type_map, label=f"{label} data",
            )

        try:
            self.downstream_type_map = read_data_type_map_union(train_systems)
        except ValueError:
            self.downstream_type_map = []

    def fit(self, train_data, aux_data, valid_data=None):
        """
        Run MFT training.

        Parameters
        ----------
        train_data : str or list[str]
            Downstream deepmd/npy directory (or list of directories).
            DeepMD-kit requires the standard label filename ``energy.npy``
            under each ``set.*`` subdir. If the raw data uses a custom name
            like ``e_form.npy``, create a symlink before calling fit():

                ln -sf set.000/e_form.npy set.000/energy.npy

            force.npy is optional (loss weight applies regardless; set to 0
            if absent).
        aux_data : str or list[str]
            Aux deepmd/npy directory. Must have energy.npy + force.npy.
        valid_data : str, optional
            Validation deepmd/npy directory.
        """
        self.train_data = train_data
        self.aux_data = aux_data
        self.valid_data = valid_data

        if self.fparam_dim > 0:
            from dpa_adapt.trainer import DPATrainer
            DPATrainer._validate_fparam(train_data, self.fparam_dim)

        import glob
        train_dirs = train_data if isinstance(train_data, list) else [train_data]
        for sys_path in train_dirs:
            e_form_sets = glob.glob(os.path.join(sys_path, "set.*", "e_form.npy"))
            for e_form_path in e_form_sets:
                energy_path = os.path.join(os.path.dirname(e_form_path), "energy.npy")
                if not os.path.exists(energy_path):
                    print(
                        f"WARNING: {e_form_path} exists but {energy_path} is missing. "
                        f"DeepMD-kit expects energy.npy — create a symlink: "
                        f"ln -sf e_form.npy {energy_path}"
                    )

        os.makedirs(self.output_dir, exist_ok=True)

        # Auto-infer type_maps when not explicitly provided.
        # Without this, the global type_map in mft_input.json is [] and
        # deepmd hits a CUDA device-side assert "index out of bounds" when
        # gathering real_atom_types (local indices) against an empty map.
        if not self.aux_type_map:
            self._resolve_type_maps(train_data, aux_data)

        from dpa_adapt.config.manager import MFTConfigManager
        cm = MFTConfigManager(self)
        config = cm.build()
        input_json = os.path.join(self.output_dir, "mft_input.json")
        cm.save(config, input_json)
        cmd = cm.build_cmd(input_json)

        log_path = os.path.join(self.output_dir, "train.log")
        print(f"Running: {cmd}")
        print(f"Log: {log_path}")

        with open(log_path, "w") as log_f:
            process = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.output_dir,
            )
            for line in process.stdout:
                print(line, end="")
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()
            process.wait()

        if process.returncode != 0:
            raise RuntimeError(
                f"dp train failed (return code {process.returncode}). "
                f"See {log_path} for details."
            )

    # ----- evaluate -----
    # `dp --pt test` for a multi-task ckpt requires a frozen .pth produced by
    # `dp --pt freeze --head <downstream head>` (property | DOWNSTREAM).
    # Feeding the raw .pt silently yields all-zero predictions. The frozen file
    # is cached in `output_dir` so a second evaluate() call is fast.
    #
    # The "Energy MAE/Natoms" line is per-atom; downstream BOOM analysis wants
    # per-molecule "Energy MAE". The regex below requires whitespace between
    # "MAE" and ":" so the "/Natoms" variant cannot match. dp prints per-system
    # blocks followed by a "weighted average of errors" block — we use findall
    # and take the LAST occurrence.
    _ENERGY_MAE_RE = re.compile(
        r"Energy\s+MAE\s+:\s*([0-9eE.+-]+)\s*\S+", re.IGNORECASE
    )
    _ENERGY_RMSE_RE = re.compile(
        r"Energy\s+RMSE\s+:\s*([0-9eE.+-]+)\s*\S+", re.IGNORECASE
    )
    _PROPERTY_MAE_RE = re.compile(
        r"PROPERTY\s+MAE\s+:\s*([0-9eE.+-]+)\s*\S*", re.IGNORECASE
    )
    _PROPERTY_RMSE_RE = re.compile(
        r"PROPERTY\s+RMSE\s+:\s*([0-9eE.+-]+)\s*\S*", re.IGNORECASE
    )
    _N_SYSTEMS_RE = re.compile(
        r"number of systems\s*[:=]?\s*(\d+)", re.IGNORECASE
    )

    @property
    def _downstream_head(self):
        """Branch/head name of the downstream task. Paper property mode uses
        "property" (matching MFTConfigManager); legacy ener mode keeps
        "DOWNSTREAM"."""
        return (
            "property"
            if getattr(self, "downstream_task_type", "ener") == "property"
            else "DOWNSTREAM"
        )

    def _freeze_ckpt(self):
        """
        Freeze ``model.ckpt-{max_steps}.pt`` to ``frozen_<head>.pth`` in
        ``output_dir`` (head = "property" or "DOWNSTREAM"). Skips if the frozen
        file already exists.

        Returns the absolute path to the frozen .pth.
        """
        head = self._downstream_head
        frozen_name = f"frozen_{head}.pth"
        frozen_path = os.path.join(self.output_dir, frozen_name)
        if os.path.exists(frozen_path):
            return frozen_path

        ckpt = os.path.join(self.output_dir, f"model.ckpt-{self.max_steps}.pt")
        if not os.path.isfile(ckpt):
            raise RuntimeError(
                f"Expected checkpoint {ckpt} not found; cannot freeze. "
                f"Did fit() complete successfully?"
            )

        # `dp --pt freeze -c .` picks up the checkpoint file from cwd, so we
        # must cd into output_dir.
        freeze_cmd = (
            f"dp --pt freeze -c . -o {frozen_name} --head {head}"
        )
        result = subprocess.run(
            freeze_cmd, shell=True,
            capture_output=True, text=True,
            cwd=self.output_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"dp --pt freeze failed (return code {result.returncode}).\n"
                f"cmd: {freeze_cmd}\n"
                f"cwd: {self.output_dir}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        if not os.path.exists(frozen_path):
            raise RuntimeError(
                f"dp --pt freeze reported success but {frozen_path} was not "
                f"created.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return frozen_path

    @staticmethod
    def _resolve_test_data(test_data):
        """
        Normalize ``test_data`` (single path, glob string, or list of paths/
        globs) to a flat list of system directories.
        """
        if isinstance(test_data, str):
            patterns = [test_data]
        else:
            patterns = list(test_data)

        resolved = []
        for pat in patterns:
            if _glob.has_magic(pat):
                matches = sorted(_glob.glob(pat))
                if not matches:
                    raise RuntimeError(
                        f"Glob pattern {pat!r} resolved to 0 systems."
                    )
                resolved.extend(matches)
            else:
                resolved.append(pat)

        # de-duplicate preserving order
        seen = set()
        unique = []
        for p in resolved:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        if not unique:
            raise RuntimeError(
                f"test_data {test_data!r} resolved to 0 systems."
            )
        return unique

    def evaluate(self, test_data):
        """
        Evaluate the downstream head of the MFT checkpoint via ``dp --pt test``.

        Pipeline:
          1. ``dp --pt freeze --head <head>`` to produce ``frozen_<head>.pth``
             (head = "property" in paper property mode, "DOWNSTREAM" in legacy
             ener mode; cached in ``output_dir``).
          2. Resolve ``test_data`` (str path, glob string, or list) to a flat
             list of system directories.
          3. Write the list to a datafile and call ``dp --pt test -m <pth>
             -f <datafile> -n 999999`` once. (Spawning one dp test per system
             is unacceptably slow — ~9s/process × hundreds of systems.)
          4. Parse the LAST occurrence of MAE / RMSE from the combined
             stdout+stderr — this is the weighted average across all systems.
             For ener tasks the keywords are ``Energy MAE`` / ``Energy RMSE``
             (the "Energy MAE/Natoms" variant is rejected by requiring
             whitespace between MAE and ``:``). For property tasks the
             keywords are ``PROPERTY MAE`` / ``PROPERTY RMSE``. The parser
             auto-detects the format from the output.

        Parameters
        ----------
        test_data : str or list[str]
            Either a single system path, a glob string, or a list of paths /
            globs.

        Returns
        -------
        dict
            ``{"mae": float, "rmse": float, "n_systems": int,
            "_parser_pattern_used": str, "_raw_stdout": str}``.

        Notes
        -----
        The DeepMD-kit output labels the unit as ``eV`` regardless of the
        actual training units; callers using Hartree-trained checkpoints
        should treat the returned numbers as Hartree.
        """
        frozen_path = self._freeze_ckpt()

        systems = self._resolve_test_data(test_data)

        os.makedirs(self.output_dir, exist_ok=True)
        datafile = os.path.join(self.output_dir, "test_systems.txt")
        with open(datafile, "w") as f:
            f.write("\n".join(systems) + "\n")

        cmd = [
            "dp", "--pt", "test",
            "-m", frozen_path,
            "-f", datafile,
            "-n", "999999",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        combined = result.stdout + "\n" + result.stderr

        return self._parse_test_output(combined, n_resolved=len(systems))

    @classmethod
    def _parse_test_output(cls, combined: str, n_resolved: int = 0) -> dict:
        """
        Extract weighted-average ``mae`` / ``rmse`` (last match) and
        ``n_systems`` from ``dp --pt test`` output.

        Auto-detects output format: "PROPERTY MAE" / "PROPERTY RMSE" for
        property tasks, "Energy MAE" / "Energy RMSE" for ener tasks.

        Raises ``RuntimeError`` with diagnostic context if neither MAE nor
        RMSE can be parsed — silent NaN returns previously masked the Bug-1
        all-zero failure for months, so we fail loudly instead.
        """
        if "PROPERTY MAE" in combined or "PROPERTY RMSE" in combined:
            mae_matches = cls._PROPERTY_MAE_RE.findall(combined)
            rmse_matches = cls._PROPERTY_RMSE_RE.findall(combined)
            tag = "PROPERTY"
        else:
            mae_matches = cls._ENERGY_MAE_RE.findall(combined)
            rmse_matches = cls._ENERGY_RMSE_RE.findall(combined)
            tag = "Energy"

        if not mae_matches and not rmse_matches:
            tail = "\n".join(combined.splitlines()[-100:])
            raise RuntimeError(
                "Could not parse Energy MAE or RMSE from `dp --pt test` "
                "output. The most common cause is feeding a raw .pt ckpt "
                "instead of a frozen .pth, which silently produces zero "
                "predictions and no MAE/RMSE lines. Re-check the freeze "
                "step.\n----- last 100 lines of combined stdout+stderr -----\n"
                f"{tail}\n----------------------"
            )

        mae = float(mae_matches[-1]) if mae_matches else float("nan")
        rmse = float(rmse_matches[-1]) if rmse_matches else float("nan")

        n_sys_match = cls._N_SYSTEMS_RE.search(combined)
        n_systems = int(n_sys_match.group(1)) if n_sys_match else n_resolved

        pattern_used = f"{tag} MAE (last); {tag} RMSE (last)"
        return {
            "mae": mae,
            "rmse": rmse,
            "n_systems": n_systems,
            "_parser_pattern_used": pattern_used,
            "_raw_stdout": combined,
        }
