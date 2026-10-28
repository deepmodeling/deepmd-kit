# SPDX-License-Identifier: LGPL-3.0-or-later
import collections
import logging
import warnings
from collections.abc import (
    Callable,
)
from functools import (
    cached_property,
)
from types import (
    SimpleNamespace,
)
from typing import (
    Any,
)

import numpy as np

import deepmd.utils.random as dp_random
from deepmd.common import (
    make_default_mesh,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils.data import (
    DataRequirementItem,
    DeepmdData,
)
from deepmd.utils.data_config import (
    DataConfig,
    check_type_map_consistency,
    resolve_data_groups,
    resolve_data_sources,
)
from deepmd.utils.data_path import (
    is_lmdb,
)
from deepmd.utils.data_path import process_systems as process_systems
from deepmd.utils.data_sampling import (
    data_child_seeds,
    system_probabilities,
    uniform_sampling_weights,
)
from deepmd.utils.data_sampling import prob_sys_size_ext as prob_sys_size_ext
from deepmd.utils.data_sampling import process_sys_probs as process_sys_probs
from deepmd.utils.out_stat import (
    compute_stats_from_redu,
)

log = logging.getLogger(__name__)


class DeepmdDataSystem:
    """Class for manipulating many data systems.

    It is implemented with the help of DeepmdData
    """

    def __init__(
        self,
        systems: list[str],
        batch_size: int,
        test_size: int,
        rcut: float | None = None,
        set_prefix: str = "set",
        shuffle_test: bool = True,
        type_map: list[str] | None = None,
        optional_type_map: bool = True,
        modifier: Any | None = None,
        trn_all_set: bool = False,
        sys_probs: list[float] | None = None,
        auto_prob_style: str = "prob_sys_size",
        sort_atoms: bool = True,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        systems
            Specifying the paths to systems
        batch_size
            The batch size
        test_size
            The size of test data
        rcut
            The cut-off radius. Not used.
        set_prefix
            Prefix for the directories of different sets
        shuffle_test
            If the test data are shuffled
        type_map
            Gives the name of different atom types
        optional_type_map
            If the type_map.raw in each system is optional
        modifier
            Data modifier that has the method `modify_data`
        trn_all_set
            Use all sets as training dataset. Otherwise, if the number of sets is more than 1, the last set is left for test.
        sys_probs : list of float
            The probabilitis of systems to get the batch.
            Summation of positive elements of this list should be no greater than 1.
            Element of this list can be negative, the probability of the corresponding system is determined
                automatically by the number of batches in the system.
        auto_prob_style : str
            Determine the probability of systems automatically. The method is assigned by this key and can be
            - "prob_uniform" : every system supplies the same sampled frame share.
            - "prob_sys_size" : the probability of a system is proportional to the number of batches in the system
            - "prob_sys_size;stt_idx:end_idx:weight;stt_idx:end_idx:weight;..." :
                                the list of systems is divided into blocks. A block is specified by `stt_idx:end_idx:weight`,
                                where `stt_idx` is the starting index of the system, `end_idx` is then ending (not including) index of the system,
                                the probabilities of the systems in this block sums up to `weight`, and the relatively probabilities within this block is proportional
                to the number of batches in the system.
        sort_atoms : bool
            Sort atoms by atom types. Required to enable when the data is directly fed to
            descriptors except mixed types.
        """
        # init data
        del rcut
        self.system_dirs = systems
        self.nsystems = len(self.system_dirs)
        if self.nsystems <= 0:
            raise ValueError("No systems provided")
        self.data_systems = []
        for ii in self.system_dirs:
            self.data_systems.append(
                DeepmdData(
                    ii,
                    set_prefix=set_prefix,
                    shuffle_test=shuffle_test,
                    type_map=type_map,
                    optional_type_map=optional_type_map,
                    modifier=modifier,
                    trn_all_set=trn_all_set,
                    sort_atoms=sort_atoms,
                )
            )
        # check mix_type format
        error_format_msg = (
            "if one of the system is of mixed_type format, "
            "then all of the systems should be of mixed_type format!"
        )
        if self.data_systems[0].mixed_type:
            for data_sys in self.data_systems[1:]:
                assert data_sys.mixed_type, error_format_msg
            self.mixed_type = True
        else:
            for data_sys in self.data_systems[1:]:
                assert not data_sys.mixed_type, error_format_msg
            self.mixed_type = False
        # batch size
        self.batch_size = batch_size
        is_auto_bs = False
        self.mixed_systems = False
        if isinstance(self.batch_size, int):
            self.batch_size = self.batch_size * np.ones(self.nsystems, dtype=int)
        elif isinstance(self.batch_size, str):
            words = self.batch_size.split(":")
            if "auto" == words[0]:
                is_auto_bs = True
                rule = 32
                if len(words) == 2:
                    rule = int(words[1])
                self.batch_size = self._make_auto_bs(rule)
            elif "mixed" == words[0]:
                self.mixed_type = True
                self.mixed_systems = True
                if len(words) == 2:
                    rule = int(words[1])
                else:
                    raise RuntimeError("batch size must be specified for mixed systems")
                self.batch_size = rule * np.ones(self.nsystems, dtype=int)
            elif "max" == words[0]:
                # Determine batch size so that batch_size * natoms <= rule, at least 1
                if len(words) != 2:
                    raise RuntimeError("batch size must be specified for max systems")
                rule = int(words[1])
                bs = []
                for ii in self.data_systems:
                    ni = ii.get_natoms()
                    bsi = rule // ni
                    if bsi == 0:
                        bsi = 1
                    bs.append(bsi)
                self.batch_size = bs
            elif "filter" == words[0]:
                # Remove systems with natoms > rule, then set batch size like "max:rule"
                if len(words) != 2:
                    raise RuntimeError(
                        "batch size must be specified for filter systems"
                    )
                rule = int(words[1])
                filtered_data_systems = []
                filtered_system_dirs = []
                for sys_dir, data_sys in zip(
                    self.system_dirs, self.data_systems, strict=True
                ):
                    if data_sys.get_natoms() <= rule:
                        filtered_data_systems.append(data_sys)
                        filtered_system_dirs.append(sys_dir)
                if len(filtered_data_systems) == 0:
                    raise RuntimeError(
                        f"No system left after removing systems with more than {rule} atoms"
                    )
                if len(filtered_data_systems) != len(self.data_systems):
                    warnings.warn(
                        f"Remove {len(self.data_systems) - len(filtered_data_systems)} systems with more than {rule} atoms"
                    )
                self.data_systems = filtered_data_systems
                self.system_dirs = filtered_system_dirs
                self.nsystems = len(self.data_systems)
                bs = []
                for ii in self.data_systems:
                    ni = ii.get_natoms()
                    bsi = rule // ni
                    if bsi == 0:
                        bsi = 1
                    bs.append(bsi)
                self.batch_size = bs
            elif words[0] == "mix":
                raise RuntimeError(
                    "the 'mix' batch_size rule packs frames of unequal atom "
                    "count into one batch and is only available for LMDB "
                    "datasets on the pt and pt_expt backends"
                )
            else:
                raise RuntimeError("unknown batch_size rule " + words[0])
        elif isinstance(self.batch_size, list):
            pass
        else:
            raise RuntimeError("invalid batch_size")
        assert isinstance(self.batch_size, (list, np.ndarray))
        assert len(self.batch_size) == self.nsystems

        # natoms, nbatches
        ntypes = []
        for ii in self.data_systems:
            ntypes.append(ii.get_ntypes())
        self.sys_ntypes = max(ntypes)
        self.natoms = []
        self.natoms_vec = []
        self.nbatches = []
        type_map_list = []
        for ii in range(self.nsystems):
            self.natoms.append(self.data_systems[ii].get_natoms())
            self.natoms_vec.append(
                self.data_systems[ii].get_natoms_vec(self.sys_ntypes).astype(int)
            )
            self.nbatches.append(
                self.data_systems[ii].get_sys_numb_batch(self.batch_size[ii])
            )
            type_map_list.append(self.data_systems[ii].get_type_map())
        self.type_map = check_type_map_consistency(type_map_list)

        # ! altered by Marián Rynik
        # test size
        # now test size can be set as a percentage of systems data or test size
        # can be set for each system individually in the same manner as batch
        # size. This enables one to use systems with diverse number of
        # structures and different number of atoms.
        self.test_size = test_size
        if isinstance(self.test_size, int):
            self.test_size = self.test_size * np.ones(self.nsystems, dtype=int)
        elif isinstance(self.test_size, str):
            words = self.test_size.split("%")
            try:
                percent = int(words[0])
            except ValueError as e:
                raise RuntimeError("unknown test_size rule " + words[0]) from e
            self.test_size = self._make_auto_ts(percent)
        elif isinstance(self.test_size, list):
            pass
        else:
            raise RuntimeError("invalid test_size")
        assert isinstance(self.test_size, (list, np.ndarray))
        assert len(self.test_size) == self.nsystems

        # init pick idx
        self.pick_idx = 0

        # derive system probabilities
        self.sys_probs = None
        self.set_sys_probs(sys_probs, auto_prob_style)

        # check batch and test size
        for ii in range(self.nsystems):
            chk_ret = self.data_systems[ii].check_batch_size(self.batch_size[ii])
            if chk_ret is not None and not is_auto_bs and not self.mixed_systems:
                warnings.warn(
                    f"system {self.system_dirs[ii]} required batch size is larger than the size of the dataset {chk_ret[0]} ({self.batch_size[ii]} > {chk_ret[1]})"
                )
            chk_ret = self.data_systems[ii].check_test_size(self.test_size[ii])
            if chk_ret is not None and not is_auto_bs and not self.mixed_systems:
                warnings.warn(
                    f"system {self.system_dirs[ii]} required test size is larger than the size of the dataset {chk_ret[0]} ({self.test_size[ii]} > {chk_ret[1]})"
                )

    def _load_test(self, ntests: int = -1) -> None:
        self.test_data = collections.defaultdict(list)
        for ii in range(self.nsystems):
            test_system_data = self.data_systems[ii].get_test(ntests=ntests)
            for nn in test_system_data:
                self.test_data[nn].append(test_system_data[nn])

    @cached_property
    def default_mesh(self) -> list[np.ndarray]:
        """Mesh for each system."""
        return [
            make_default_mesh(
                self.data_systems[ii].pbc, self.data_systems[ii].mixed_type
            )
            for ii in range(self.nsystems)
        ]

    def compute_energy_shift(
        self, rcond: float | None = None, key: str = "energy"
    ) -> tuple[np.ndarray, np.ndarray]:
        sys_ener = []
        for ss in self.data_systems:
            sys_ener.append(ss.avg(key))
        sys_ener = np.concatenate(sys_ener)
        sys_tynatom = np.array(self.natoms_vec, dtype=GLOBAL_NP_FLOAT_PRECISION)
        sys_tynatom = np.reshape(sys_tynatom, [self.nsystems, -1])
        sys_tynatom = sys_tynatom[:, 2:]
        energy_shift, _ = compute_stats_from_redu(
            sys_ener.reshape(-1, 1),
            sys_tynatom,
            rcond=rcond,
        )
        return energy_shift.ravel()

    def add_dict(self, adict: dict[str, dict[str, Any]]) -> None:
        """Add items to the data system by a `dict`.
        `adict` should have items like
        .. code-block:: python.

           adict[key] = {
               "ndof": ndof,
               "atomic": atomic,
               "must": must,
               "high_prec": high_prec,
               "type_sel": type_sel,
               "repeat": repeat,
           }

        For the explanation of the keys see `add`
        """
        for kk in adict:
            self.add(
                kk,
                adict[kk]["ndof"],
                atomic=adict[kk]["atomic"],
                must=adict[kk]["must"],
                high_prec=adict[kk]["high_prec"],
                type_sel=adict[kk]["type_sel"],
                repeat=adict[kk]["repeat"],
                default=adict[kk]["default"],
                dtype=adict[kk].get("dtype"),
                output_natoms_for_type_sel=adict[kk].get(
                    "output_natoms_for_type_sel", False
                ),
                special_shape=adict[kk].get("special_shape"),
                length_scale=adict[kk].get("length_scale", "absolute"),
            )

    def add_data_requirements(
        self, data_requirements: list[DataRequirementItem]
    ) -> None:
        """Add items to the data system by a list of `DataRequirementItem`."""
        self.add_dict({rr.key: rr.dict for rr in data_requirements})

    def add(
        self,
        key: str,
        ndof: int,
        atomic: bool = False,
        must: bool = False,
        high_prec: bool = False,
        type_sel: list[int] | None = None,
        repeat: int = 1,
        default: float = 0.0,
        dtype: np.dtype | None = None,
        output_natoms_for_type_sel: bool = False,
        special_shape: str | None = None,
        length_scale: str = "absolute",
    ) -> None:
        """Add a data item that to be loaded.

        Parameters
        ----------
        key
            The key of the item. The corresponding data is stored in `sys_path/set.*/key.npy`
        ndof
            The number of dof
        atomic
            The item is an atomic property.
            If False, the size of the data should be nframes x ndof
            If True, the size of data should be nframes x natoms x ndof
        must
            The data file `sys_path/set.*/key.npy` must exist.
            If must is False and the data file does not exist, the `data_dict[find_key]` is set to 0.0
        high_prec
            Load the data and store in float64, otherwise in float32
        type_sel
            Select certain type of atoms
        repeat
            The data will be repeated `repeat` times.
        default, default=0.
            Default value of data
        dtype
            The dtype of data, overwrites `high_prec` if provided
        output_natoms_for_type_sel : bool
            If True and type_sel is True, the atomic dimension will be natoms instead of nsel
        special_shape : str, optional
            Name of a loader-defined non-standard shape contract.
        length_scale : str, optional
            Per-element length scale a derived geometric item measures its
            default against, either ``"absolute"`` or ``"covalent"``.
        """
        for ii in self.data_systems:
            ii.add(
                key,
                ndof,
                atomic=atomic,
                must=must,
                high_prec=high_prec,
                repeat=repeat,
                type_sel=type_sel,
                default=default,
                dtype=dtype,
                output_natoms_for_type_sel=output_natoms_for_type_sel,
                special_shape=special_shape,
                length_scale=length_scale,
            )

    def reduce(self, key_out: str, key_in: str) -> None:
        """Generate a new item from the reduction of another atom.

        Parameters
        ----------
        key_out
            The name of the reduced item
        key_in
            The name of the data item to be reduced
        """
        for ii in self.data_systems:
            ii.reduce(key_out, key_in)

    def get_data_dict(self, ii: int = 0) -> dict:
        return self.data_systems[ii].get_data_dict()

    def set_sys_probs(
        self,
        sys_probs: list[float] | None = None,
        auto_prob_style: str = "prob_sys_size",
    ) -> None:
        self.sys_probs = system_probabilities(
            self.nbatches,
            sys_probs=sys_probs,
            auto_prob=auto_prob_style,
            batch_nframes=(
                self.mean_batch_nframes
                if sys_probs is None and auto_prob_style == "prob_uniform"
                else None
            ),
        )

    @cached_property
    def mean_batch_nframes(self) -> np.ndarray:
        """Return native mean batch sizes for per-system selection."""
        if self.mixed_systems:
            return np.ones(self.nsystems, dtype=np.float64)
        return np.asarray(
            [
                system.get_mean_batch_nframes(size)
                for system, size in zip(self.data_systems, self.batch_size, strict=True)
            ],
            dtype=np.float64,
        )

    @cached_property
    def uniform_sampling_mass(self) -> float:
        """Return the summed systems-per-frame weights of the native streams."""
        if self.mixed_systems:
            return float(self.nsystems / self.batch_size[0])
        return float(uniform_sampling_weights(self.mean_batch_nframes).sum())

    @property
    def uniform_nominal_mass(self) -> float:
        """Return the systems-per-frame weight averaged over native passes."""
        return self.uniform_sampling_mass

    def get_batch(self, sys_idx: int | None = None) -> dict:
        # batch generation style altered by Ziyao Li:
        # one should specify the "sys_prob" and "auto_prob_style" params
        # via set_sys_prob() function. The sys_probs this function uses is
        # defined as a private variable, self.sys_probs, initialized in __init__().
        # This is to optimize the (vain) efforts in evaluating sys_probs every batch.
        """Get a batch of data from the data systems.

        Parameters
        ----------
        sys_idx : int
            The index of system from which the batch is get.
            If sys_idx is not None, `sys_probs` and `auto_prob_style` are ignored
            If sys_idx is None, automatically determine the system according to `sys_probs` or `auto_prob_style`, see the following.
            This option does not work for mixed systems.

        Returns
        -------
        dict
            The batch data
        """
        if not self.mixed_systems:
            b_data = self.get_batch_standard(sys_idx)
        else:
            b_data = self.get_batch_mixed()
        return b_data

    def state_dict(self) -> dict[str, Any]:
        """Return logical sampling progress for all systems.

        The caller must save the shared ``dp_random`` state separately,
        once for all training and validation data sources.
        """
        return {
            "version": 1,
            "batch_size": [int(size) for size in self.batch_size],
            "sys_probs": np.asarray(self.sys_probs).tolist(),
            "mixed_systems": self.mixed_systems,
            "pick_idx": int(self.pick_idx),
            "systems": [data.state_dict() for data in self.data_systems],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore progress without materializing any frame data."""
        if state.get("version") != 1:
            raise ValueError("Unsupported DeepmdDataSystem progress version")
        if state.get("batch_size") != [int(size) for size in self.batch_size]:
            raise ValueError("DeepmdDataSystem progress has different batch sizes")
        try:
            saved_probs = np.array(state.get("sys_probs"), dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "DeepmdDataSystem progress has invalid probabilities"
            ) from error
        if (
            saved_probs.shape != (self.nsystems,)
            or not np.all(np.isfinite(saved_probs))
            or np.any(saved_probs < 0)
            or not np.allclose(saved_probs, self.sys_probs, rtol=1e-12, atol=0)
        ):
            raise ValueError("DeepmdDataSystem progress has different probabilities")
        if state.get("mixed_systems") != self.mixed_systems:
            raise ValueError("DeepmdDataSystem progress has a different batching mode")
        states = state.get("systems", [])
        if len(states) != self.nsystems:
            raise ValueError("DeepmdDataSystem progress has different systems")
        pick_idx = state.get("pick_idx")
        if type(pick_idx) is not int or not 0 <= pick_idx < self.nsystems:
            raise ValueError("Invalid DeepmdDataSystem progress pick_idx")
        # Validate every system before changing any reader's progress.
        for data, data_state in zip(self.data_systems, states, strict=True):
            data._validate_state_dict(data_state)
        for data, data_state in zip(self.data_systems, states, strict=True):
            data.load_state_dict(data_state)
        # Tolerate recomputation roundoff, but retain the checkpoint's exact
        # selection thresholds so the restored RNG resumes the same stream.
        # Preserve an exactly matching dtype too: NumPy uses it to choose
        # the tolerance for checking whether probabilities sum to one.
        if np.array_equal(saved_probs, self.sys_probs):
            saved_probs = np.array(self.sys_probs, copy=True)
        self.sys_probs = saved_probs
        self.pick_idx = pick_idx
        self.__dict__.pop("test_data", None)

    def get_batch_standard(self, sys_idx: int | None = None) -> dict:
        """Get a batch of data from the data systems in the standard way.

        Parameters
        ----------
        sys_idx : int
            The index of system from which the batch is get.
            If sys_idx is not None, `sys_probs` and `auto_prob_style` are ignored
            If sys_idx is None, automatically determine the system according to `sys_probs` or `auto_prob_style`, see the following.

        Returns
        -------
        dict
            The batch data
        """
        if sys_idx is not None:
            self.pick_idx = sys_idx
        else:
            self.pick_idx = dp_random.choice(
                np.arange(self.nsystems, dtype=np.int32), p=self.sys_probs
            )
        b_data = self.data_systems[self.pick_idx].get_batch(
            self.batch_size[self.pick_idx]
        )
        b_data["natoms_vec"] = self.natoms_vec[self.pick_idx]
        b_data["default_mesh"] = self.default_mesh[self.pick_idx]
        return b_data

    def get_batch_mixed(self) -> dict:
        """Get a batch of data from the data systems in the mixed way.

        Returns
        -------
        dict
            The batch data
        """
        # mixed systems have a global batch size
        batch_size = self.batch_size[0]
        batch_data = []
        for _ in range(batch_size):
            self.pick_idx = dp_random.choice(
                np.arange(self.nsystems, dtype=np.int32), p=self.sys_probs
            )
            bb_data = self.data_systems[self.pick_idx].get_batch(1)
            bb_data["natoms_vec"] = self.natoms_vec[self.pick_idx]
            bb_data["default_mesh"] = self.default_mesh[self.pick_idx]
            batch_data.append(bb_data)
        b_data = self._merge_batch_data(batch_data)
        return b_data

    def _merge_batch_data(self, batch_data: list[dict]) -> dict:
        """Merge batch data from different systems.

        Parameters
        ----------
        batch_data : list of dict
            A list of batch data from different systems.

        Returns
        -------
        dict
            The merged batch data.
        """
        b_data = {}
        max_natoms = max(bb["natoms_vec"][0] for bb in batch_data)
        # natoms_vec
        natoms_vec = np.zeros(2 + self.get_ntypes(), dtype=int)
        natoms_vec[0:3] = max_natoms
        b_data["natoms_vec"] = natoms_vec
        # real_natoms_vec
        real_natoms_vec = np.vstack([bb["natoms_vec"] for bb in batch_data])
        b_data["real_natoms_vec"] = real_natoms_vec
        # type
        type_vec = np.full((len(batch_data), max_natoms), -1, dtype=int)
        for ii, bb in enumerate(batch_data):
            type_vec[ii, : bb["type"].shape[1]] = bb["type"][0]
        b_data["type"] = type_vec
        # default_mesh
        default_mesh = np.mean([bb["default_mesh"] for bb in batch_data], axis=0)
        b_data["default_mesh"] = default_mesh
        # other data
        data_dict = self.get_data_dict(0)
        for kk, vv in data_dict.items():
            if kk not in batch_data[0]:
                continue
            b_data["find_" + kk] = batch_data[0]["find_" + kk]
            if vv.get("special_shape") == "hessian" or kk == "hessian":
                # A Hessian is a (3 * natoms, 3 * natoms) matrix, so neither
                # branch below pads it correctly: concatenating raises on
                # ragged systems and copying a flat prefix would scatter the
                # rows. Embed each frame's block in the top-left corner of the
                # padded square instead; the padded rows and columns stay zero
                # and are dropped by the loss mask.
                padded_dof = max_natoms * 3
                merged = np.zeros(
                    (len(batch_data), padded_dof, padded_dof),
                    dtype=batch_data[0][kk].dtype,
                )
                for ii, bb in enumerate(batch_data):
                    frame_dof = bb["natoms_vec"][0] * 3
                    merged[ii, :frame_dof, :frame_dof] = bb[kk][0].reshape(
                        frame_dof, frame_dof
                    )
                b_data[kk] = merged.reshape(len(batch_data), -1)
            elif not vv["atomic"]:
                b_data[kk] = np.concatenate([bb[kk] for bb in batch_data], axis=0)
            else:
                b_data[kk] = np.zeros(
                    (len(batch_data), max_natoms * vv["ndof"] * vv["repeat"]),
                    dtype=batch_data[0][kk].dtype,
                )
                for ii, bb in enumerate(batch_data):
                    b_data[kk][ii, : bb[kk].shape[1]] = bb[kk][0]
        return b_data

    # ! altered by Marián Rynik
    def get_test(
        self, sys_idx: int | None = None, n_test: int = -1
    ) -> dict[str, np.ndarray]:  # depreciated
        """Get test data from the the data systems.

        Parameters
        ----------
        sys_idx
            The test dat of system with index `sys_idx` will be returned.
            If is None, the currently selected system will be returned.
        n_test
            Number of test data. If set to -1 all test data will be get.
        """
        if not hasattr(self, "test_data"):
            self._load_test(ntests=n_test)
        if sys_idx is not None:
            idx = sys_idx
        else:
            idx = self.pick_idx

        test_system_data = {}
        for nn in self.test_data:
            test_system_data[nn] = self.test_data[nn][idx]
        test_system_data["natoms_vec"] = self.natoms_vec[idx]
        test_system_data["default_mesh"] = self.default_mesh[idx]
        return test_system_data

    def get_sys_ntest(self, sys_idx: int | None = None) -> int:
        """Get number of tests for the currently selected system,
        or one defined by sys_idx.
        """
        if sys_idx is not None:
            return self.test_size[sys_idx]
        else:
            return self.test_size[self.pick_idx]

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    def get_nbatches(self) -> int:
        """Get the total number of batches."""
        return self.nbatches

    def get_ntypes(self) -> int:
        """Get the number of types."""
        return self.sys_ntypes

    def get_nsystems(self) -> int:
        """Get the number of data systems."""
        return self.nsystems

    def get_sys(self, idx: int) -> DeepmdData:
        """Get a certain data system."""
        return self.data_systems[idx]

    def get_batch_size(self) -> int:
        """Get the batch size."""
        return self.batch_size

    def get_batch_pass_length(self) -> int:
        """Return the local sampling-pass length, including system probabilities.

        NumPy data systems are replicated across ranks, so a reader's pass
        length is independent of the distributed optimizer's world size.
        """
        from deepmd.dpmodel.utils.training_utils import (
            compute_total_numb_batch,
        )

        return compute_total_numb_batch(self.nbatches, self.sys_probs)

    def get_epoch_batch_counts(self, world_size: int) -> np.ndarray:
        """Return optimizer workloads for the replicated NumPy data reader."""
        return np.asarray(self.nbatches, dtype=np.float64) / world_size

    def print_summary(self, name: str) -> None:
        print_summary(
            name,
            self.nsystems,
            self.system_dirs,
            self.natoms,
            self.batch_size,
            self.nbatches,
            self.sys_probs,
            [ii.pbc for ii in self.data_systems],
        )

    def _make_auto_bs(self, rule: int) -> list[int]:
        bs = []
        for ii in self.data_systems:
            ni = ii.get_natoms()
            bsi = rule // ni
            if bsi * ni < rule:
                bsi += 1
            bs.append(bsi)
        return bs

    # ! added by Marián Rynik
    def _make_auto_ts(self, percent: float) -> list[int]:
        ts = []
        for ii in range(self.nsystems):
            ni = self.batch_size[ii] * self.nbatches[ii]
            tsi = int(ni * percent / 100)
            ts.append(tsi)

        return ts


def _format_name_length(name: str, width: int) -> str:
    if len(name) <= width:
        return "{: >{}}".format(name, width)
    else:
        name = name[-(width - 3) :]
        name = "-- " + name
        return name


def print_summary(
    name: str,
    nsystems: int,
    system_dirs: list[str],
    natoms: list[int],
    batch_size: list[int],
    nbatches: list[int],
    sys_probs: list[float],
    pbc: list[bool],
    e_max: list[int] | None = None,
) -> None:
    """Print summary of systems.

    Parameters
    ----------
    name : str
        The name of the system
    nsystems : int
        The number of systems
    system_dirs : list of str
        The directories of the systems
    natoms : list of int
        The number of atoms
    batch_size : list of int
        The batch size
    nbatches : list of int
        The number of batches
    sys_probs : list of float
        The probabilities
    pbc : list of bool
        The periodic boundary conditions
    e_max : list of int, optional
        The maximal number of valid edges per frame for each system.
    """
    # width 65
    sys_width = 42
    log.info(
        f"---Summary of DataSystem: {name.capitalize():13s}-----------------------------------------------"
    )
    log.info("Found %d System(s):", nsystems)
    use_e_max = e_max is not None and len(e_max) == nsystems
    if use_e_max:
        emax_width = max(5, len(str(max(e_max))))
        log.info(
            "%s  %-6s  %-*s  %-6s  %-6s  %-9s  %-3s",
            _format_name_length("system", sys_width),
            "natoms",
            emax_width,
            "e_max",
            "bch_sz",
            "n_bch",
            "prob",
            "pbc",
        )
    else:
        log.info(
            "%s  %-6s  %-6s  %-6s  %-9s  %-3s",
            _format_name_length("system", sys_width),
            "natoms",
            "bch_sz",
            "n_bch",
            "prob",
            "pbc",
        )
    for ii in range(nsystems):
        if use_e_max:
            log.info(
                "%s  %6d  %*d  %6d  %6d  %9.3e  %3s",
                _format_name_length(system_dirs[ii], sys_width),
                natoms[ii],
                emax_width,
                e_max[ii],
                batch_size[ii],
                nbatches[ii],
                sys_probs[ii],
                "T" if pbc[ii] else "F",
            )
        else:
            log.info(
                "%s  %6d  %6d  %6d  %9.3e  %3s",
                _format_name_length(system_dirs[ii], sys_width),
                natoms[ii],
                batch_size[ii],
                nbatches[ii],
                sys_probs[ii],
                "T" if pbc[ii] else "F",
            )
    log.info(
        "--------------------------------------------------------------------------------------"
    )


def get_data(
    jdata: DataConfig,
    rcut: float,
    type_map: list[str] | None,
    modifier: Any | None,
    multi_task_mode: bool = False,
    *,
    lmdb_factory: Callable[[str, int | str, str | None], Any] | None = None,
    seed: int | None = None,
) -> Any:
    """Build native data readers and compose independently configured groups.

    Parameters
    ----------
    jdata
        A data dictionary or a nonempty list of group dictionaries.
    rcut
        The cut-off radius, not used
    type_map
        The type map
    modifier
        The data modifier
    multi_task_mode
        If in multi task mode
    lmdb_factory
        Optional backend adapter accepting an LMDB path, batch size, and
        internal auto-probability rule. Without it, only NPY/HDF5 sources
        are discovered.
    seed
        Seed for group selection. If omitted, draws from the shared training
        random generator seed the group-selection streams.

    Returns
    -------
    object
        A native data system for a single source, or a grouped data system
        with the same batch, statistics, and sampling-metadata interface.
    """
    from deepmd.utils.data_group import (
        GroupedDataSystem,
    )

    groups = resolve_data_groups(
        jdata, formats=("deepmd", "lmdb") if lmdb_factory is not None else ("deepmd",)
    )
    needs_composition = len(groups) > 1 or any(
        len(group.systems) > 1 and any(is_lmdb(path) for path in group.systems)
        for group in groups
    )
    seeds = (
        iter(data_child_seeds(seed, len(groups) + 1)) if needs_composition else iter(())
    )
    group_sources = []
    for group in groups:
        sources = resolve_data_sources(group)
        has_lmdb = any(source.format == "lmdb" for source in sources)
        children = []
        for source in sources:
            if source.format == "deepmd":
                child = DeepmdDataSystem(
                    systems=list(source.systems),
                    batch_size=source.batch_size,
                    test_size=1,
                    shuffle_test=True,
                    rcut=rcut,
                    type_map=type_map,
                    optional_type_map=not multi_task_mode,
                    modifier=modifier,
                    trn_all_set=True,
                    sys_probs=group.sys_probs if not has_lmdb else None,
                    auto_prob_style=group.auto_prob or "prob_sys_size",
                )
            else:
                assert lmdb_factory is not None
                child = lmdb_factory(
                    source.systems[0], source.batch_size, group.auto_prob
                )
            children.append(child)
        group_sources.append(children)

    sources = [source for children in group_sources for source in children]
    numpy_sources = [
        source for source in sources if isinstance(source, DeepmdDataSystem)
    ]
    if len({source.mixed_type for source in numpy_sources}) > 1:
        raise ValueError("All NumPy data systems must agree on mixed_type.")
    ntypes = max(source.get_ntypes() for source in sources)
    for source in numpy_sources:
        source.sys_ntypes = ntypes
        source.natoms_vec = [
            system.get_natoms_vec(ntypes).astype(int) for system in source.data_systems
        ]
    datasets = [
        (
            children[0]
            if len(children) == 1
            else GroupedDataSystem(
                children,
                [None] * len(children),
                seed=next(seeds),
                uniform=group.auto_prob == "prob_uniform",
            )
        )
        for group, children in zip(groups, group_sources, strict=True)
    ]
    return (
        datasets[0]
        if len(datasets) == 1
        else GroupedDataSystem(
            datasets, [group.prob for group in groups], seed=next(seeds)
        )
    )


def get_neighbor_stat_data(jdata: DataConfig, type_map: list[str] | None) -> Any:
    """Expose every discovered source to backend neighbor-statistic kernels."""
    sources = []
    for group in resolve_data_groups(jdata):
        for source in resolve_data_sources(group):
            if source.format == "deepmd":
                sources.append(
                    DeepmdDataSystem(
                        list(source.systems),
                        source.batch_size,
                        1,
                        type_map=type_map,
                        trn_all_set=True,
                    )
                )
            else:
                from deepmd.dpmodel.utils.lmdb_data import (
                    make_neighbor_stat_data,
                )

                sources.append(make_neighbor_stat_data(source.systems[0], type_map))
    return SimpleNamespace(
        data_systems=[system for source in sources for system in source.data_systems],
        system_dirs=[path for source in sources for path in source.system_dirs],
        mixed_type=any(source.mixed_type for source in sources),
        get_ntypes=lambda: max(source.get_ntypes() for source in sources),
        get_batch=lambda: None,
    )
