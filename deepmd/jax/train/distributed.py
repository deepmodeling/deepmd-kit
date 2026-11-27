# SPDX-License-Identifier: LGPL-3.0-or-later
"""Global-array data parallelism and ZeRO stage-one optimizer-state placement."""

import logging
from collections.abc import (
    Callable,
    Sequence,
)
from functools import (
    wraps,
)
from typing import (
    Any,
)

import numpy as np
from jax.sharding import (
    Mesh,
    NamedSharding,
    PartitionSpec,
    Sharding,
    SingleDeviceSharding,
)
from packaging.version import (
    Version,
)

from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)
from deepmd.dpmodel.loss.ener_spin import (
    EnergySpinLoss,
)
from deepmd.dpmodel.loss.property import (
    PropertyLoss,
)
from deepmd.dpmodel.train.data import (
    next_training_batch,
)
from deepmd.dpmodel.train.sharding import (
    ShardingPolicy,
)
from deepmd.dpmodel.utils.batch import (
    normalize_batch,
)
from deepmd.dpmodel.utils.dist_check import (
    pair_margin_frame_mask,
    select_frames,
)
from deepmd.dpmodel.utils.neighbor_graph.graph import (
    frame_id_from_n_node,
)
from deepmd.jax.env import (
    OptState,
    jax,
    jnp,
    nnx,
)
from deepmd.jax.utils.graph_batch import (
    atomic_batch_keys,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

log = logging.getLogger(__name__)


def _round_capacity(size: int, multiple: int) -> int:
    return ((size + multiple - 1) // multiple) * multiple


def _pad_axis(value: Any, capacity: int, axis: int = 0, fill: int = 0) -> Any:
    if value.shape[axis] == capacity:
        return value
    widths = [(0, 0)] * value.ndim
    widths[axis] = (0, capacity - value.shape[axis])
    xp = np if isinstance(value, np.ndarray) else jnp
    return xp.pad(value, widths, constant_values=fill)


class DistributedTraining:
    """
    Place one global training batch and state over the local JAX device mesh.

    Parameters
    ----------
    training_params : dict
        Training configuration containing ``zero_stage``.
    devices : sequence of jax.Device, optional
        Local devices participating in training. The default uses all visible
        local devices; ``CUDA_VISIBLE_DEVICES`` controls GPU visibility.

    Notes
    -----
    Stage zero replicates the complete training state. Stage one replicates
    parameters and gradients and partitions eligible optimizer arrays along
    one divisible dimension. Scalar clocks and arrays without a divisible
    dimension remain replicated. Model losses and gradient clipping retain a
    global-array view. Each logical device's input batch retains its own
    masked loss normalization, and their objectives are averaged before the
    optimization update, matching PyTorch data-parallel training.

    Sparse graph nodes, edges, and frames have separate partitioned axes.
    The compiler inserts graph gathers and reduction collectives according to
    those global indices; no frame or neighbor is dropped to fit a shard.
    """

    def __init__(
        self,
        training_params: dict[str, Any],
        devices: Sequence[jax.Device] | None = None,
    ) -> None:
        if jax.process_count() > 1:
            raise NotImplementedError(
                "JAX distributed training currently uses one controller for local devices. "
                "Run one process with the participating GPUs visible."
            )
        devices = tuple(jax.local_devices() if devices is None else devices)
        if not devices:
            raise ValueError("JAX training requires at least one local device.")
        self.device_count = len(devices)
        if self.device_count > 1 and Version(jax.__version__) < Version("0.7.2"):
            raise NotImplementedError(
                f"Multi-device training requires JAX 0.7.2 or newer; found {jax.__version__}. "
                "Upgrade jax and jaxlib, or run with one visible device."
            )
        self.policy = ShardingPolicy.from_training_params(
            training_params, is_distributed=self.device_count > 1
        )
        if self.policy.shards_parameters:
            raise NotImplementedError(
                "JAX training supports zero_stage=0 or 1; parameter sharding at "
                f"zero_stage={self.policy.stage} is not implemented."
            )
        self.mesh = Mesh(np.asarray(devices), ("data",))
        self.replicated = (
            NamedSharding(self.mesh, PartitionSpec())
            if self.is_distributed
            else SingleDeviceSharding(devices[0])
        )
        self._reported_batch = False
        if self.is_distributed:
            budgets = (
                {
                    key: value.get("training_data", {}).get("batch_size", "auto")
                    for key, value in training_params["data_dict"].items()
                }
                if "data_dict" in training_params
                else training_params.get("training_data", {}).get("batch_size", "auto")
            )
            log.info(
                "JAX training uses %d local devices, zero_stage=%d, and per-device "
                "batch budget %s. One update combines one input batch per device "
                "and averages their independently normalized losses.",
                self.device_count,
                self.policy.stage,
                budgets,
            )

    @property
    def is_distributed(self) -> bool:
        return self.device_count > 1

    def data_sharding(self, axis: int = 0) -> Sharding:
        """Shard the selected array dimension across the device mesh."""
        if not self.is_distributed:
            return self.replicated
        return NamedSharding(self.mesh, PartitionSpec(*([None] * axis), "data"))

    def epoch_length(self, data: Any) -> int:
        """Return updates per sampler pass at one original batch per device."""
        return (
            data.get_batch_pass_length() + self.device_count - 1
        ) // self.device_count

    def get_batch_group(
        self, data: Any, *, task_key: str, filter_enabled: bool = False
    ) -> tuple[dict[str, Any], ...]:
        """Read synchronized logical-device batches and apply frame filtering.

        Parameters
        ----------
        data : Any
            Reader exposing ``get_batch`` and ``get_batch_pass_length``.
        task_key : str
            Training-task identifier used in failure diagnostics.
        filter_enabled : bool
            Whether the reader supplies a derived pair-clearance margin.
            Validation readers are never filtered.

        Returns
        -------
        tuple of dict
            One normalized, nonempty batch for each logical device.

        Raises
        ------
        RuntimeError
            If enabled filtering lacks derived margins or no complete valid
            device group is found within one sampling pass.

        Notes
        -----
        Any empty filtered batch discards the entire device group, matching
        PyTorch's all-rank validity reduction. Validation callers omit the
        filter radius. The sampler pads each pass to a device multiple so
        groups never join different sampler epochs.
        """
        if not self.is_distributed:
            return (
                next_training_batch(
                    data, filter_enabled=filter_enabled, task_key=task_key
                ),
            )
        attempts = max(self.epoch_length(data), 1) if filter_enabled else 1
        for _ in range(attempts):
            raw_batches = tuple(data.get_batch() for _ in range(self.device_count))
            batches = tuple(normalize_batch(batch) for batch in raw_batches)
            masks = tuple(
                pair_margin_frame_mask(
                    batch, getattr(raw, "pair_filter_enabled", filter_enabled)
                )
                for raw, batch in zip(raw_batches, batches, strict=True)
            )
            if any(mask is not None and not mask.any() for mask in masks):
                continue
            return tuple(
                select_frames(batch, mask)
                if mask is not None and not mask.all()
                else batch
                for batch, mask in zip(batches, masks, strict=True)
            )
        raise RuntimeError(
            f"No globally valid training batch group for task {task_key!r} was found "
            f"within {attempts} attempts. Every device batch must retain at least one "
            "frame after pair-clearance filtering; lower "
            "`training_data.min_pair_dist` or clean the dataset."
        )

    def optimizer_sharding(self, value: Any) -> Sharding:
        """Partition independent matrix batches first, then a divisible tensor axis."""
        if not self.policy.shards_optimizer_state:
            return self.replicated
        candidates = [
            axis
            for axis, size in enumerate(value.shape)
            if size >= self.device_count and size % self.device_count == 0
        ]
        if not candidates:
            return self.replicated
        axis = (
            0
            if value.ndim >= 3 and 0 in candidates
            else max(candidates, key=lambda index: value.shape[index])
        )
        return self.data_sharding(axis)

    def _place_node(self, node: Any, *, optimizer: bool = False) -> None:
        sharding = (
            self.optimizer_sharding if optimizer else lambda value: self.replicated
        )
        state = nnx.state(node, OptState) if optimizer else nnx.state(node)
        state = jax.tree.map(
            lambda value: jax.device_put(value, sharding(value)),
            state,
        )
        nnx.update(node, state)

    def place_state(
        self,
        models: Any,
        optimizer: nnx.Optimizer,
        ema_models: Any = None,
        *,
        active: Any = None,
    ) -> None:
        """Place restored or initialized state while preserving NNX shared aliases."""
        self._place_node(models)
        self._place_node(optimizer, optimizer=True)
        if ema_models is not None:
            self._place_node(ema_models)
        if active is not None:
            # Activity masks are state snapshots, whose VariableState leaves
            # are immutable in older NNX releases. Replace the snapshot's
            # top-level entries instead of updating it as a live model graph.
            placed = jax.tree.map(
                lambda value: jax.device_put(value, self.replicated), active
            )
            for key in active:
                active[key] = placed[key]
        if not self.is_distributed:
            return
        arrays = jax.tree_util.tree_leaves(nnx.state(optimizer, OptState))
        total_bytes = sum(value.nbytes for value in arrays)
        sharded_bytes = sum(
            value.nbytes for value in arrays if not value.sharding.is_fully_replicated
        )
        resident_bytes = sum(
            value.addressable_shards[0].data.nbytes for value in arrays
        )
        log.info(
            "JAX optimizer state partitions %.1f%% of its bytes; persistent state "
            "per device is %.2f MiB from %.2f MiB globally. Scalar clocks and "
            "non-divisible tensors remain replicated.",
            100.0 * sharded_bytes / max(total_bytes, 1),
            resident_bytes / 2**20,
            total_bytes / 2**20,
        )

    def constrain_gradients(self, gradients: Any) -> Any:
        """Materialize globally reduced replicated gradients before global clipping."""
        if not self.is_distributed:
            return gradients
        return jax.tree.map(
            lambda value: jax.lax.with_sharding_constraint(value, self.replicated),
            gradients,
        )

    def evaluate_loss(
        self,
        loss: Callable[..., Any],
        learning_rate: Any,
        natoms: int,
        model_dict: dict[str, Any],
        label_dict: dict[str, Any],
        *,
        mae: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        """Average independently normalized device batches from one model call.

        Parameters
        ----------
        loss : Callable
            Shared array-API loss object.
        learning_rate : Any
            Current learning rate used by the objective's prefactors.
        natoms : int
            Static atom capacity supplied to the ordinary loss interface.
        model_dict : dict
            Combined model predictions, node masks and graph frame counts.
        label_dict : dict
            Combined labels and optional logical-device group metadata.
        mae : bool
            Request additional absolute-error diagnostics.

        Returns
        -------
        Any
            Scalar mean of the original device-batch objectives.
        dict
            Diagnostic metrics from logical device zero.

        Raises
        ------
        NotImplementedError
            If a grouped objective lacks complete frame and node masking.

        Notes
        -----
        ``data_group`` assigns every physical frame to its original input
        batch. ``group_find_*`` retains each batch's label availability.
        Padding frames have group ``-1``. Ungrouped batches retain the shared
        loss's ordinary global-array reduction, including validation batches.
        Diagnostic metrics come from logical device zero, matching PyTorch's
        chief-rank training curve; only the optimized objective is averaged.
        """
        if "data_group" not in label_dict:
            return loss(learning_rate, natoms, model_dict, label_dict, mae=mae)
        if not isinstance(loss, (EnergyLoss, EnergySpinLoss, PropertyLoss)) or (
            isinstance(loss, EnergyLoss) and (loss.has_h or loss.has_gf)
        ):
            raise NotImplementedError(
                "Grouped JAX data-parallel losses support energy, native-spin "
                "energy and property objectives without Hessian or generalized-force terms."
            )
        groups = label_dict["data_group"]
        frame_mask = label_dict.get(
            "frame_mask", jnp.ones_like(groups, dtype=jnp.bool_)
        )
        atom_mask = model_dict["mask"]
        if "n_node" in model_dict:
            frame_id = frame_id_from_n_node(
                model_dict["n_node"], n_total=atom_mask.shape[0]
            )
            node_group = groups[frame_id]
        else:
            node_group = jnp.broadcast_to(groups[:, None], atom_mask.shape)
        availability = {
            key.removeprefix("group_"): value
            for key, value in label_dict.items()
            if key.startswith("group_find_")
        }

        def local_loss(group: Any) -> tuple[Any, dict[str, Any]]:
            membership = node_group == group
            local_model = dict(model_dict)
            local_model["mask"] = atom_mask * membership
            if "mask_mag" in model_dict:
                magnetic_mask = model_dict["mask_mag"]
                local_model["mask_mag"] = magnetic_mask * membership.reshape(
                    *membership.shape, *([1] * (magnetic_mask.ndim - membership.ndim))
                )
            local_labels = dict(label_dict)
            local_labels["frame_mask"] = frame_mask & (groups == group)
            local_labels.update(
                {key: value[group] for key, value in availability.items()}
            )
            return loss(learning_rate, natoms, local_model, local_labels, mae=mae)

        values, metrics = jax.vmap(local_loss)(
            jnp.arange(self.device_count, dtype=jnp.int64)
        )
        return jnp.mean(values), jax.tree.map(lambda value: value[0], metrics)

    def _constrain_node(self, node: Any, *, optimizer: bool = False) -> None:
        sharding = (
            self.optimizer_sharding if optimizer else lambda value: self.replicated
        )
        state = nnx.state(node, OptState) if optimizer else nnx.state(node)
        state = jax.tree.map(
            lambda value: jax.lax.with_sharding_constraint(value, sharding(value)),
            state,
        )
        nnx.update(node, state)

    def jit_step(self, function: Callable[..., Any]) -> Callable[..., Any]:
        """Compile a training step with persistent model and optimizer layouts."""
        if not self.is_distributed:
            return nnx.jit(function)

        @wraps(function)
        def step(
            models: Any,
            optimizer: nnx.Optimizer,
            learning_rate: Any,
            batch: Any,
            active: Any,
            ema_models: Any,
        ) -> Any:
            result = function(
                models, optimizer, learning_rate, batch, active, ema_models
            )
            self._constrain_node(models)
            self._constrain_node(optimizer, optimizer=True)
            if ema_models is not None:
                self._constrain_node(ema_models)
            return result

        return nnx.jit(step)

    def prepare_batch(
        self,
        labels: dict[str, Any],
        inputs: dict[str, Any],
        *,
        data_requirements: Sequence[DataRequirementItem] = (),
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pad masked axes to device multiples and distribute the global arrays."""
        if "edge_index" in inputs:
            labels, inputs, label_axes, input_axes = self._pad_graph_batch(
                labels, inputs, data_requirements
            )
        else:
            labels, inputs, label_axes, input_axes = self._pad_dense_batch(
                labels, inputs
            )

        def place(values: dict[str, Any], axes: dict[str, int]) -> dict[str, Any]:
            result = {}
            for key, value in values.items():
                if value is None:
                    result[key] = None
                elif key.startswith("find_"):
                    result[key] = bool(value)
                else:
                    sharding = (
                        self.data_sharding(axes[key])
                        if key in axes
                        else self.replicated
                    )
                    result[key] = jax.device_put(value, sharding)
            return result

        return place(labels, label_axes), place(inputs, input_axes)

    def _pad_graph_batch(
        self,
        labels: dict[str, Any],
        inputs: dict[str, Any],
        data_requirements: Sequence[DataRequirementItem],
    ) -> tuple[dict, dict, dict, dict]:
        labels, inputs = labels.copy(), inputs.copy()
        node_capacity = _round_capacity(inputs["atype"].shape[0], self.device_count)
        frame_capacity = _round_capacity(inputs["n_node"].shape[0], self.device_count)
        edge_capacity = _round_capacity(inputs["edge_vec"].shape[0], self.device_count)
        if self.is_distributed and not self._reported_batch:
            log.info(
                "JAX global graph capacity: %d nodes, %d edges, %d frames; "
                "per-device partitions: %d nodes, %d edges, %d frames.",
                node_capacity,
                edge_capacity,
                frame_capacity,
                node_capacity // self.device_count,
                edge_capacity // self.device_count,
                frame_capacity // self.device_count,
            )
            if "data_group" in labels:
                groups = np.asarray(labels["data_group"])
                valid = groups >= 0
                counts = np.asarray(inputs["n_node"])
                log.info(
                    "JAX logical-device batches contain frames %s and physical atoms %s; "
                    "optimization averages their local losses and diagnostics use device zero.",
                    np.bincount(groups[valid], minlength=self.device_count).tolist(),
                    np.bincount(
                        groups[valid],
                        weights=counts[valid],
                        minlength=self.device_count,
                    )
                    .astype(np.int64)
                    .tolist(),
                )
            self._reported_batch = True
        atomic_keys = atomic_batch_keys(data_requirements)
        input_axes = {}
        for key, value in inputs.items():
            if value is None:
                continue
            axis = 1 if key == "edge_index" else 0
            if key in ("edge_index", "edge_vec", "edge_mask"):
                capacity = edge_capacity
            elif key in ("atype", "spin", "aparam"):
                capacity = node_capacity
            else:
                capacity = frame_capacity
            inputs[key] = _pad_axis(value, capacity, axis, -1 if key == "atype" else 0)
            input_axes[key] = axis
        label_axes = {}
        for key, value in labels.items():
            if (
                value is None
                or key.startswith(("find_", "group_find_"))
                or getattr(value, "ndim", 0) == 0
            ):
                continue
            capacity = node_capacity if key in atomic_keys else frame_capacity
            labels[key] = _pad_axis(
                value, capacity, fill=-1 if key in ("type", "data_group") else 0
            )
            label_axes[key] = 0
        return labels, inputs, label_axes, input_axes

    def _pad_dense_batch(
        self, labels: dict[str, Any], inputs: dict[str, Any]
    ) -> tuple[dict, dict, dict, dict]:
        labels, inputs = labels.copy(), inputs.copy()
        nframes = inputs["nlist"].shape[0]
        capacity = _round_capacity(nframes, self.device_count)
        labels.setdefault("frame_mask", np.ones(nframes, dtype=np.bool_))
        input_axes, label_axes = {}, {}
        for values, axes in ((inputs, input_axes), (labels, label_axes)):
            for key, value in values.items():
                if (
                    value is None
                    or key.startswith(("find_", "group_find_"))
                    or getattr(value, "ndim", 0) == 0
                ):
                    continue
                if value.shape[0] != nframes:
                    continue
                fill = (
                    -1
                    if key in ("nlist", "extended_atype", "atype", "type", "data_group")
                    else 0
                )
                values[key] = _pad_axis(value, capacity, fill=fill)
                axes[key] = 0
        return labels, inputs, label_axes, input_axes
