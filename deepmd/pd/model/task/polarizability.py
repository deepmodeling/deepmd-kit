# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
from typing import (
    Optional,
    Union,
)

import paddle

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pd.model.task.fitting import (
    GeneralFitting,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.pd.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
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
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    embedding_width : int
        The dimension of rotation matrix, m1.
    neuron : list[int]
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
    scale : list[float]
        The output of the fitting net (polarizability matrix) for type i atom will be scaled by scale[i]
    shift_diag : bool
        Whether to shift the diagonal part of the polarizability matrix. The shift operation is carried out after scale.
    type_map: list[str], Optional
        A list of strings. Give the name to each type of atoms.

    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        embedding_width: int,
        neuron: list[int] = [128, 128, 128],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        rcond: Optional[float] = None,
        seed: Optional[Union[int, list[int]]] = None,
        exclude_types: list[int] = [],
        fit_diag: bool = True,
        scale: Optional[Union[list[float], float]] = None,
        shift_diag: bool = True,
        type_map: Optional[list[str]] = None,
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
        super().__init__(
            var_name="polar",
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
            type_map=type_map,
            **kwargs,
        )
        self.scale = paddle.to_tensor(
            self.scale, dtype=env.GLOBAL_PD_FLOAT_PRECISION, place=env.DEVICE
        ).reshape([ntypes, 1])
        self.shift_diag = shift_diag
        self.constant_matrix = paddle.zeros(
            [ntypes], dtype=env.GLOBAL_PD_FLOAT_PRECISION
        ).to(device=env.DEVICE)

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

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert (
            self.type_map is not None
        ), "'type_map' must be defined when performing type changing!"
        assert self.mixed_types, "Only models in mixed types can perform type changing!"
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        super().change_type_map(type_map=type_map)
        if has_new_type:
            extend_shape = [len(type_map), *list(self.scale.shape[1:])]
            extend_scale = paddle.ones(
                extend_shape, dtype=self.scale.dtype, device=self.scale.place
            )
            self.scale = paddle.concat([self.scale, extend_scale], axis=0)
            extend_shape = [len(type_map), *list(self.constant_matrix.shape[1:])]
            extend_constant_matrix = paddle.zeros(
                extend_shape,
                dtype=self.constant_matrix.dtype,
            ).to(device=self.constant_matrix.place)
            self.constant_matrix = paddle.concat(
                [self.constant_matrix, extend_constant_matrix], axis=0
            )
        self.scale = self.scale[remap_index]
        self.constant_matrix = self.constant_matrix[remap_index]

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "polar"
        data["@version"] = 3
        data["embedding_width"] = self.embedding_width
        data["fit_diag"] = self.fit_diag
        data["shift_diag"] = self.shift_diag
        data["@variables"]["scale"] = to_numpy_array(self.scale)
        data["@variables"]["constant_matrix"] = to_numpy_array(self.constant_matrix)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 3, 1)
        data.pop("var_name", None)
        return super().deserialize(data)

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "polarizability",
                    [3, 3],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def forward(
        self,
        descriptor: paddle.Tensor,
        atype: paddle.Tensor,
        gr: Optional[paddle.Tensor] = None,
        g2: Optional[paddle.Tensor] = None,
        h2: Optional[paddle.Tensor] = None,
        fparam: Optional[paddle.Tensor] = None,
        aparam: Optional[paddle.Tensor] = None,
    ):
        nframes, nloc, _ = descriptor.shape
        assert (
            gr is not None
        ), "Must provide the rotation matrix for polarizability fitting."
        # (nframes, nloc, _net_out_dim)
        out = self._forward_common(descriptor, atype, gr, g2, h2, fparam, aparam)[
            self.var_name
        ]
        out = out * (self.scale.to(atype.place))[atype]
        gr = gr.reshape(
            [nframes * nloc, self.embedding_width, 3]
        )  # (nframes * nloc, m1, 3)

        if self.fit_diag:
            out = out.reshape([-1, self.embedding_width])
            out = paddle.einsum("ij,ijk->ijk", out, gr)
        else:
            out = out.reshape([-1, self.embedding_width, self.embedding_width])
            out = (out + out.transpose([0, 2, 1])) / 2
            out = paddle.einsum("bim,bmj->bij", out, gr)  # (nframes * nloc, m1, 3)
        out = paddle.einsum(
            "bim,bmj->bij", gr.transpose([0, 2, 1]), out
        )  # (nframes * nloc, 3, 3)
        out = out.reshape([nframes, nloc, 3, 3])
        return {"polarizability": out.to(env.GLOBAL_PD_FLOAT_PRECISION)}

    # make jit happy with paddle 2.0.0
    exclude_types: list[int]
