# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np

from deepmd.common import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.dpmodel import (
    DEFAULT_PRECISION,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .general_fitting import (
    GeneralFitting,
)


@BaseFitting.register("polar")
@fitting_check_output
class PolarFitting(GeneralFitting):
    r"""Fitting rotationally equivariant polarizability of the system.

    Parameters
    ----------
    ntypes
            The number of atom types.
    dim_descrpt
            The dimension of the input descriptor.
    embedding_width : int
        The dimension of rotation matrix, m1.
    neuron
            Number of neurons :math:`N` in each hidden layer of the fitting net
    resnet_dt
            Time-step `dt` in the resnet construction:
            :math:`y = x + dt * \phi (Wx + b)`
    numb_fparam
            Number of frame parameter
    numb_aparam
            Number of atomic parameter
    rcond
            The condition number for the regression of atomic energy.
    tot_ener_zero
            Force the total energy to zero. Useful for the charge fitting.
    trainable
            If the weights of fitting net are trainable.
            Suppose that we have :math:`N_l` hidden layers in the fitting net,
            this list is of length :math:`N_l + 1`, specifying if the hidden layers and the output layer are trainable.
    activation_function
            The activation function :math:`\boldsymbol{\phi}` in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    layer_name : list[Optional[str]], optional
            The name of the each layer. If two layers, either in the same fitting or different fittings,
            have the same name, they will share the same neural network parameters.
    use_aparam_as_mask: bool, optional
            If True, the atomic parameters will be used as a mask that determines the atom is real/virtual.
            And the aparam will not be used as the atomic parameters for embedding.
    mixed_types
            If true, use a uniform fitting net for all atom types, otherwise use
            different fitting nets for different atom types.
    fit_diag : bool
            Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to
            normal polarizability matrix by contracting with the rotation matrix.
    scale : List[float]
            The output of the fitting net (polarizability matrix) for type i atom will be scaled by scale[i]
    shift_diag : bool
            Whether to shift the diagonal part of the polarizability matrix. The shift operation is carried out after scale.
    type_map: List[str], Optional
            A list of strings. Give the name to each type of atoms.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        embedding_width: int,
        neuron: List[int] = [120, 120, 120],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        rcond: Optional[float] = None,
        tot_ener_zero: bool = False,
        trainable: Optional[List[bool]] = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        layer_name: Optional[List[Optional[str]]] = None,
        use_aparam_as_mask: bool = False,
        spin: Any = None,
        mixed_types: bool = False,
        exclude_types: List[int] = [],
        old_impl: bool = False,
        fit_diag: bool = True,
        scale: Optional[List[float]] = None,
        shift_diag: bool = True,
        type_map: Optional[List[str]] = None,
        seed: Optional[Union[int, List[int]]] = None,
    ):
        if tot_ener_zero:
            raise NotImplementedError("tot_ener_zero is not implemented")
        if spin is not None:
            raise NotImplementedError("spin is not implemented")
        if use_aparam_as_mask:
            raise NotImplementedError("use_aparam_as_mask is not implemented")
        if layer_name is not None:
            raise NotImplementedError("layer_name is not implemented")

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
        self.scale = np.array(self.scale, dtype=GLOBAL_NP_FLOAT_PRECISION).reshape(
            ntypes, 1
        )
        self.shift_diag = shift_diag
        self.constant_matrix = np.zeros(ntypes, dtype=GLOBAL_NP_FLOAT_PRECISION)
        super().__init__(
            var_name="polar",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            neuron=neuron,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            rcond=rcond,
            tot_ener_zero=tot_ener_zero,
            trainable=trainable,
            activation_function=activation_function,
            precision=precision,
            layer_name=layer_name,
            use_aparam_as_mask=use_aparam_as_mask,
            spin=spin,
            mixed_types=mixed_types,
            exclude_types=exclude_types,
            type_map=type_map,
            seed=seed,
        )
        self.old_impl = False

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
        data["@version"] = 3
        data["embedding_width"] = self.embedding_width
        data["old_impl"] = self.old_impl
        data["fit_diag"] = self.fit_diag
        data["shift_diag"] = self.shift_diag
        data["@variables"]["scale"] = self.scale
        data["@variables"]["constant_matrix"] = self.constant_matrix
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 3, 1)
        var_name = data.pop("var_name", None)
        assert var_name == "polar"
        return super().deserialize(data)

    def output_def(self):
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

    def change_type_map(
        self, type_map: List[str], model_with_new_type_stat=None
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
            extend_scale = np.ones(extend_shape, dtype=self.scale.dtype)
            self.scale = np.concatenate([self.scale, extend_scale], axis=0)
            extend_shape = [len(type_map), *list(self.constant_matrix.shape[1:])]
            extend_constant_matrix = np.zeros(
                extend_shape, dtype=self.constant_matrix.dtype
            )
            self.constant_matrix = np.concatenate(
                [self.constant_matrix, extend_constant_matrix], axis=0
            )
        self.scale = self.scale[remap_index]
        self.constant_matrix = self.constant_matrix[remap_index]

    def call(
        self,
        descriptor: np.ndarray,
        atype: np.ndarray,
        gr: Optional[np.ndarray] = None,
        g2: Optional[np.ndarray] = None,
        h2: Optional[np.ndarray] = None,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Calculate the fitting.

        Parameters
        ----------
        descriptor
            input descriptor. shape: nf x nloc x nd
        atype
            the atom type. shape: nf x nloc
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3
        g2
            The rotationally invariant pair-partical representation.
            shape: nf x nloc x nnei x ng
        h2
            The rotationally equivariant pair-partical representation.
            shape: nf x nloc x nnei x 3
        fparam
            The frame parameter. shape: nf x nfp. nfp being `numb_fparam`
        aparam
            The atomic parameter. shape: nf x nloc x nap. nap being `numb_aparam`

        """
        nframes, nloc, _ = descriptor.shape
        assert (
            gr is not None
        ), "Must provide the rotation matrix for polarizability fitting."
        # (nframes, nloc, _net_out_dim)
        out = self._call_common(descriptor, atype, gr, g2, h2, fparam, aparam)[
            self.var_name
        ]
        out = out * self.scale[atype]
        # (nframes * nloc, m1, 3)
        gr = gr.reshape(nframes * nloc, -1, 3)

        if self.fit_diag:
            out = out.reshape(-1, self.embedding_width)
            out = np.einsum("ij,ijk->ijk", out, gr)
        else:
            out = out.reshape(-1, self.embedding_width, self.embedding_width)
            out = (out + np.transpose(out, axes=(0, 2, 1))) / 2
            out = np.einsum("bim,bmj->bij", out, gr)  # (nframes * nloc, m1, 3)
        out = np.einsum(
            "bim,bmj->bij", np.transpose(gr, axes=(0, 2, 1)), out
        )  # (nframes * nloc, 3, 3)
        out = out.reshape(nframes, nloc, 3, 3)
        if self.shift_diag:
            bias = self.constant_matrix[atype]
            # (nframes, nloc, 1)
            bias = np.expand_dims(bias, axis=-1) * self.scale[atype]
            eye = np.eye(3)
            eye = np.tile(eye, (nframes, nloc, 1, 1))
            # (nframes, nloc, 3, 3)
            bias = np.expand_dims(bias, axis=-1) * eye
            out = out + bias
        return {"polarizability": out}
