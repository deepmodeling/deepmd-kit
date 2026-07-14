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
        The bias for normalization.
    out_std : list or None
        The standard deviation for normalization.
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

        # Normalize by natoms for extensive properties (without mutating input)
        if not self.intensive:
            if "mask" in model_dict:
                # Per-frame real atom count: shape [nf] → broadcast over [nf, task_dim].
                real_natoms = xp.reshape(
                    xp.astype(xp.sum(model_dict["mask"], axis=-1), pred.dtype),
                    (-1,) + (1,) * (pred.ndim - 1),
                )
                pred = pred / real_natoms
                label = label / real_natoms
            else:
                pred = pred / natoms
                label = label / natoms

        # Get out_std and out_bias
        if self.out_std is not None:
            out_std = xp.asarray(self.out_std, dtype=pred.dtype, device=dev)
        else:
            out_std = xp.ones((self.task_dim,), dtype=pred.dtype, device=dev)
        if self.out_bias is not None:
            out_bias = xp.asarray(self.out_bias, dtype=pred.dtype, device=dev)
        else:
            out_bias = xp.zeros((self.task_dim,), dtype=pred.dtype, device=dev)

        loss = xp.zeros((), dtype=pred.dtype, device=dev)
        more_loss = {}

        norm_pred = (pred - out_bias) / out_std
        norm_label = (label - out_bias) / out_std
        diff = norm_label - norm_pred

        if self.loss_func == "smooth_mae":
            abs_diff = xp.abs(diff)
            smooth_l1 = xp.where(
                abs_diff < self.beta,
                0.5 * diff**2 / self.beta,
                abs_diff - 0.5 * self.beta,
            )
            loss = loss + xp.sum(smooth_l1)
        elif self.loss_func == "mae":
            loss = loss + xp.sum(xp.abs(diff))
        elif self.loss_func == "mse":
            loss = loss + xp.sum(xp.square(diff))
        elif self.loss_func == "rmse":
            loss = loss + xp.sqrt(xp.mean(xp.square(diff)))
        elif self.loss_func == "mape":
            loss = loss + xp.mean(xp.abs((label - pred) / (label + 1e-3)))
        else:
            raise RuntimeError(f"Unknown loss function : {self.loss_func}")

        # metrics (computed on un-normalized values)
        if "smooth_mae" in self.metric:
            abs_raw = xp.abs(label - pred)
            more_loss["smooth_mae"] = xp.mean(
                xp.where(
                    abs_raw < self.beta,
                    0.5 * (label - pred) ** 2 / self.beta,
                    abs_raw - 0.5 * self.beta,
                )
            )
        if "mae" in self.metric:
            more_loss["mae"] = xp.mean(xp.abs(label - pred))
        if "mse" in self.metric:
            more_loss["mse"] = xp.mean(xp.square(label - pred))
        if "rmse" in self.metric:
            more_loss["rmse"] = xp.sqrt(xp.mean(xp.square(label - pred)))
        if "mape" in self.metric:
            more_loss["mape"] = xp.mean(xp.abs((label - pred) / (label + 1e-3)))

        return loss, more_loss

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
