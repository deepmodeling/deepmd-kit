# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
from typing import (
    Callable,
    List,
    Optional,
    Union,
)

import numpy as np
import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.task.fitting import (
    GeneralFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.out_stat import (
    compute_stats_from_atomic,
    compute_stats_from_redu,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

log = logging.getLogger(__name__)


@GeneralFitting.register("polar")
class PolarFittingNet(GeneralFitting):
    """Construct a polar fitting net.

    Parameters
    ----------
    var_name : str
        The atomic property to fit, 'polar'.
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    embedding_width : int
        The dimension of rotation matrix, m1.
    neuron : List[int]
        Number of neurons in each hidden layers of the fitting net.
    resnet_dt : bool
        Using time-step in the ResNet construction.
    numb_fparam : int
        Number of frame parameters.
    numb_aparam : int
        Number of atomic parameters.
    activation_function : str
        Activation function.
    precision : str
        Numerical precision.
    mixed_types : bool
        If true, use a uniform fitting net for all atom types, otherwise use
        different fitting nets for different atom types.
    rcond : float, optional
        The condition number for the regression of atomic energy.
    seed : int, optional
        Random seed.
    fit_diag : bool
        Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to
        normal polarizability matrix by contracting with the rotation matrix.
    scale : List[float]
        The output of the fitting net (polarizability matrix) for type i atom will be scaled by scale[i]
    shift_diag : bool
        Whether to shift the diagonal part of the polarizability matrix. The shift operation is carried out after scale.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        embedding_width: int,
        neuron: List[int] = [128, 128, 128],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        rcond: Optional[float] = None,
        seed: Optional[int] = None,
        exclude_types: List[int] = [],
        fit_diag: bool = True,
        scale: Optional[Union[List[float], float]] = None,
        shift_diag: bool = True,
        **kwargs,
    ):
        self.embedding_width = embedding_width
        self.fit_diag = fit_diag
        self.scale = scale
        if self.scale is None:
            self.scale = [1.0 for _ in range(ntypes)]
        else:
            if isinstance(self.scale, list):
                assert (
                    len(self.scale) == ntypes
                ), "Scale should be a list of length ntypes."
            elif isinstance(self.scale, float):
                self.scale = [self.scale for _ in range(ntypes)]
            else:
                raise ValueError(
                    "Scale must be a list of float of length ntypes or a float."
                )
        self.scale = torch.tensor(
            self.scale, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        ).view(ntypes, 1)
        self.shift_diag = shift_diag
        self.constant_matrix = torch.zeros(
            ntypes, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )
        super().__init__(
            var_name=kwargs.pop("var_name", "polar"),
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            neuron=neuron,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            rcond=rcond,
            seed=seed,
            exclude_types=exclude_types,
            **kwargs,
        )
        self.old_impl = False  # this only supports the new implementation.

    def _net_out_dim(self):
        """Set the FittingNet output dim."""
        return (
            self.embedding_width
            if self.fit_diag
            else self.embedding_width * self.embedding_width
        )

    def __setitem__(self, key, value):
        if key in ["constant_matrix"]:
            self.constant_matrix = value
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        if key in ["constant_matrix"]:
            return self.constant_matrix
        else:
            return super().__getitem__(key)

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "polar"
        data["@version"] = 2
        data["embedding_width"] = self.embedding_width
        data["old_impl"] = self.old_impl
        data["fit_diag"] = self.fit_diag
        data["shift_diag"] = self.shift_diag
        data["@variables"]["scale"] = to_numpy_array(self.scale)
        data["@variables"]["constant_matrix"] = to_numpy_array(self.constant_matrix)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        return super().deserialize(data)

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [3, 3],
                    reduciable=True,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )
    
    # need to remove this and update UT test_polar_stat.py
    def compute_output_stats(
        self,
        merged: Union[Callable[[], List[dict]], List[dict]],
        stat_file_path: Optional[DPPath] = None,
    ) -> None:
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The path to the stat file.

        """
        if self.shift_diag:
            if stat_file_path is not None:
                stat_file_path = stat_file_path / "constant_matrix"
            if stat_file_path is not None and stat_file_path.is_file():
                constant_matrix = stat_file_path.load_numpy()
            else:
                if callable(merged):
                    # only get data for once
                    sampled = merged()
                else:
                    sampled = merged

                sys_constant_matrix = []
                for sys in range(len(sampled)):
                    nframs = sampled[sys]["type"].shape[0]

                    if sampled[sys]["find_atomic_polarizability"] > 0.0:
                        sys_atom_polar = compute_stats_from_atomic(
                            sampled[sys]["atomic_polarizability"].numpy(force=True),
                            sampled[sys]["type"].numpy(force=True),
                        )[0]
                    else:
                        if not sampled[sys]["find_polarizability"] > 0.0:
                            continue
                        sys_type_count = np.zeros(
                            (nframs, self.ntypes), dtype=env.GLOBAL_NP_FLOAT_PRECISION
                        )
                        for itype in range(self.ntypes):
                            type_mask = sampled[sys]["type"] == itype
                            sys_type_count[:, itype] = type_mask.sum(dim=1).numpy(
                                force=True
                            )

                        sys_bias_redu = sampled[sys]["polarizability"].numpy(force=True)

                        sys_atom_polar = compute_stats_from_redu(
                            sys_bias_redu, sys_type_count, rcond=self.rcond
                        )[0]
                    cur_constant_matrix = np.zeros(
                        self.ntypes, dtype=env.GLOBAL_NP_FLOAT_PRECISION
                    )

                    for itype in range(self.ntypes):
                        cur_constant_matrix[itype] = np.mean(
                            np.diagonal(sys_atom_polar[itype].reshape(3, 3))
                        )
                    sys_constant_matrix.append(cur_constant_matrix)
                constant_matrix = np.stack(sys_constant_matrix).mean(axis=0)

                # handle nan values.
                constant_matrix = np.nan_to_num(constant_matrix)
            if stat_file_path is not None:
                stat_file_path.save_numpy(constant_matrix)
            self.constant_matrix = torch.tensor(constant_matrix, device=env.DEVICE)

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: Optional[torch.Tensor] = None,
        g2: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ):
        nframes, nloc, _ = descriptor.shape
        assert (
            gr is not None
        ), "Must provide the rotation matrix for polarizability fitting."
        # (nframes, nloc, _net_out_dim)
        out = self._forward_common(descriptor, atype, gr, g2, h2, fparam, aparam)[
            self.var_name
        ]
        out = out * self.scale[atype]
        gr = gr.view(nframes * nloc, -1, 3)  # (nframes * nloc, m1, 3)

        if self.fit_diag:
            out = out.reshape(-1, self.embedding_width)
            out = torch.einsum("ij,ijk->ijk", out, gr)
        else:
            out = out.reshape(-1, self.embedding_width, self.embedding_width)
            out = (out + out.transpose(1, 2)) / 2
            out = torch.einsum("bim,bmj->bij", out, gr)  # (nframes * nloc, m1, 3)
        out = torch.einsum(
            "bim,bmj->bij", gr.transpose(1, 2), out
        )  # (nframes * nloc, 3, 3)
        out = out.view(nframes, nloc, 3, 3)
        if self.shift_diag:
            bias = self.constant_matrix[atype]

            # (nframes, nloc, 1)
            bias = bias.unsqueeze(-1) * self.scale[atype]

            eye = torch.eye(3, device=env.DEVICE)
            eye = eye.repeat(nframes, nloc, 1, 1)
            # (nframes, nloc, 3, 3)
            bias = bias.unsqueeze(-1) * eye
            out = out + bias

        return {self.var_name: out.to(env.GLOBAL_PT_FLOAT_PRECISION)}

    # make jit happy with torch 2.0.0
    exclude_types: List[int]
