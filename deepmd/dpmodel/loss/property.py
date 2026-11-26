# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.loss.loss import (
    Loss,
)
from deepmd.dpmodel.loss.reduction import (
    absolute_error,
    masked_frame_mean,
)
from deepmd.dpmodel.utils.neighbor_graph.graph import (
    frame_id_from_n_node,
)
from deepmd.dpmodel.utils.neighbor_graph.segment import (
    segment_sum,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class PropertyLoss(Loss):
    r"""Loss on property predictions.

    Except for MAPE, the residual is evaluated in normalized property space,

    .. math::

       e_k=\frac{y_k-b_k}{s_k}-\frac{\hat y_k-b_k}{s_k}.

    Extensive properties are first divided by the number of real atoms.
    ``smooth_mae`` uses

    .. math::

       \ell_\beta(e)=\begin{cases}
       e^2/(2\beta),&|e|<\beta,\\
       |e|-\beta/2,&|e|\ge\beta.
       \end{cases}

    The other choices are the usual :math:`\sum|e|`, :math:`\sum e^2`,
    :math:`\sqrt{\langle e^2\rangle}`, and
    :math:`\langle |(y-\hat y)/(y+10^{-3})|\rangle`.

    Parameters
    ----------
    task_dim : int
        The output dimension of property fitting net.
    var_name : str
        The property variable name.
    loss_func : str
        The loss function: "smooth_mae", "mae", "mse", "rmse", "mape".
    metric : list[str]
        The metrics to report.
    beta : float
        The 'beta' parameter in 'smooth_mae' loss.
    out_bias : list or None
        Bias for normalization. ``None`` reads ``model_dict["_out_bias"]``
        when provided, otherwise zero.
    out_std : list or None
        Standard deviation for normalization. ``None`` reads
        ``model_dict["_out_std"]`` when provided, otherwise one.
    intensive : bool
        Whether the property is intensive.
    **kwargs
        Other keyword arguments.
    """

    def __init__(
        self,
        task_dim: int,
        var_name: str,
        loss_func: str = "smooth_mae",
        metric: list[str] | None = None,
        beta: float = 1.00,
        out_bias: list | None = None,
        out_std: list | None = None,
        intensive: bool = False,
        **kwargs: Any,
    ) -> None:
        if metric is None:
            metric = ["mae"]
        self.task_dim = task_dim
        self.var_name = var_name
        self.loss_func = loss_func
        self.metric = metric
        self.beta = beta
        self.out_bias = out_bias
        self.out_std = out_std
        self.intensive = intensive

    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, Array],
        label_dict: dict[str, Array],
        mae: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        r"""Evaluate the configured normalized property error.

        For extensive targets, both prediction and label are divided by the
        real atom count before the equations in :class:`PropertyLoss` are
        applied.
        """
        del learning_rate, mae
        var_name = self.var_name
        pred = model_dict[var_name]
        xp = array_api_compat.array_namespace(pred)
        dev = array_api_compat.device(pred)
        label = label_dict[var_name]
        frame_mask = label_dict.get("frame_mask")

        # Normalize by natoms for extensive properties (without mutating input)
        if not self.intensive:
            if "n_node" in model_dict:
                n_node = model_dict["n_node"]
                counts = xp.astype(n_node, pred.dtype)
                if "mask" in model_dict:
                    mask = xp.reshape(model_dict["mask"], (-1,))
                    counts = segment_sum(
                        xp.astype(mask, pred.dtype),
                        frame_id_from_n_node(n_node, n_total=mask.shape[0]),
                        n_node.shape[0],
                    )
                real_natoms = xp.reshape(
                    xp.where(counts > 0, counts, xp.ones_like(counts)),
                    (-1,) + (1,) * (pred.ndim - 1),
                )
                pred = pred / real_natoms
                label = label / real_natoms
            elif "mask" in model_dict:
                # Per-frame real atom count: shape [nf] → broadcast over [nf, task_dim].
                real_natoms = xp.reshape(
                    xp.astype(xp.sum(model_dict["mask"], axis=-1), pred.dtype),
                    (-1,) + (1,) * (pred.ndim - 1),
                )
                safe_natoms = xp.where(
                    real_natoms > 0, real_natoms, xp.ones_like(real_natoms)
                )
                pred = pred / safe_natoms
                label = label / safe_natoms
            else:
                pred = pred / natoms
                label = label / natoms

        # Get out_std and out_bias
        if self.out_std is not None:
            out_std = xp.asarray(self.out_std, dtype=pred.dtype, device=dev)
        elif "_out_std" in model_dict:
            out_std = xp.astype(model_dict["_out_std"], pred.dtype)
        else:
            out_std = xp.ones((self.task_dim,), dtype=pred.dtype, device=dev)
        if self.out_bias is not None:
            out_bias = xp.asarray(self.out_bias, dtype=pred.dtype, device=dev)
        elif "_out_bias" in model_dict:
            out_bias = xp.astype(model_dict["_out_bias"], pred.dtype)
        else:
            out_bias = xp.zeros((self.task_dim,), dtype=pred.dtype, device=dev)

        loss = xp.zeros((), dtype=pred.dtype, device=dev)
        more_loss = {}

        norm_pred = (pred - out_bias) / out_std
        norm_label = (label - out_bias) / out_std
        diff = norm_label - norm_pred

        def sum_frames(value: Array) -> Array:
            if frame_mask is None:
                return xp.sum(value)
            mask = xp.reshape(frame_mask, (-1,) + (1,) * (value.ndim - 1))
            return xp.sum(xp.where(mask, value, xp.zeros_like(value)))

        def root_mean_square(value: Array) -> Array:
            mean_square = masked_frame_mean(xp.square(value), frame_mask)
            zero = mean_square == 0
            safe_square = xp.where(zero, xp.ones_like(mean_square), mean_square)
            return xp.where(zero, xp.zeros_like(mean_square), xp.sqrt(safe_square))

        if self.loss_func == "smooth_mae":
            abs_diff = absolute_error(diff)
            smooth_l1 = xp.where(
                abs_diff < self.beta,
                0.5 * diff**2 / self.beta,
                abs_diff - 0.5 * self.beta,
            )
            loss = loss + sum_frames(smooth_l1)
        elif self.loss_func == "mae":
            loss = loss + sum_frames(absolute_error(diff))
        elif self.loss_func == "mse":
            loss = loss + sum_frames(xp.square(diff))
        elif self.loss_func == "rmse":
            loss = loss + root_mean_square(diff)
        elif self.loss_func == "mape":
            loss = loss + masked_frame_mean(
                absolute_error((label - pred) / (label + 1e-3)), frame_mask
            )
        else:
            raise RuntimeError(f"Unknown loss function : {self.loss_func}")

        # metrics (computed on un-normalized values)
        if "smooth_mae" in self.metric:
            abs_raw = absolute_error(label - pred)
            more_loss["smooth_mae"] = masked_frame_mean(
                xp.where(
                    abs_raw < self.beta,
                    0.5 * (label - pred) ** 2 / self.beta,
                    abs_raw - 0.5 * self.beta,
                ),
                frame_mask,
            )
        if "mae" in self.metric:
            more_loss["mae"] = masked_frame_mean(
                absolute_error(label - pred), frame_mask
            )
        if "mse" in self.metric:
            more_loss["mse"] = masked_frame_mean(xp.square(label - pred), frame_mask)
        if "rmse" in self.metric:
            more_loss["rmse"] = root_mean_square(label - pred)
        if "mape" in self.metric:
            more_loss["mape"] = masked_frame_mean(
                absolute_error((label - pred) / (label + 1e-3)), frame_mask
            )

        return loss, more_loss

    @property
    def training_metric_names(self) -> tuple[str, ...]:
        """Return the selected property metrics."""
        return tuple(
            name
            for name in ("smooth_mae", "mae", "mse", "rmse", "mape")
            if name in self.metric
        )

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        label_requirement.append(
            DataRequirementItem(
                self.var_name,
                ndof=self.task_dim,
                atomic=False,
                must=True,
                high_prec=True,
            )
        )
        return label_requirement

    def serialize(self) -> dict:
        """Serialize the loss module."""
        return {
            "@class": "PropertyLoss",
            "@version": 1,
            "task_dim": self.task_dim,
            "var_name": self.var_name,
            "loss_func": self.loss_func,
            "metric": self.metric,
            "beta": self.beta,
            "out_bias": self.out_bias,
            "out_std": self.out_std,
            "intensive": self.intensive,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "PropertyLoss":
        """Deserialize the loss module."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        return cls(**data)
