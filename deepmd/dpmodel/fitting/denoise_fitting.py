# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.common import (
    cast_precision,
    get_xp_precision,
    to_numpy_array,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.dpmodel.utils import (
    AtomExcludeMask,
    FittingNet,
    NetworkCollection,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_atom_exclude_types,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_fitting import (
    BaseFitting,
)


class DenoiseFitting(NativeOP, BaseFitting):
    r"""Deoise fitting class.

    Parameters
    ----------
    var_name
            The name of the output variable.
    ntypes
            The number of atom types.
    dim_descrpt
            The dimension of the input descriptor.
    neuron
            Number of neurons :math:`N` in each hidden layer of the fitting net
    bias_atom_e
            Average energy per atom for each element.
    resnet_dt
            Time-step `dt` in the resnet construction:
            :math:`y = x + dt * \phi (Wx + b)`
    numb_fparam
            Number of frame parameter
    numb_aparam
            Number of atomic parameter
    trainable
            If the weights of fitting net are trainable.
            Suppose that we have :math:`N_l` hidden layers in the fitting net,
            this list is of length :math:`N_l + 1`, specifying if the hidden layers and the output layer are trainable.
    activation_function
            The activation function :math:`\boldsymbol{\phi}` in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    use_aparam_as_mask: bool, optional
            If True, the atomic parameters will be used as a mask that determines the atom is real/virtual.
            And the aparam will not be used as the atomic parameters for embedding.
    mixed_types
            If true, use a uniform fitting net for all atom types, otherwise use
            different fitting nets for different atom types.
    exclude_types: list[int]
            Atomic contributions of the excluded atom types are set zero.
    type_map: list[str], Optional
            A list of strings. Give the name to each type of atoms.
    seed: Optional[Union[int, list[int]]]
        Random seed for initializing the network parameters.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        embedding_width: int,
        neuron: list[int] = [120, 120, 120],
        bias_atom_e: Optional[np.ndarray] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        exclude_types: list[int] = [],
        trainable: Optional[list[bool]] = None,
        type_map: Optional[list[str]] = None,
        use_aparam_as_mask: bool = False,
        coord_noise: Optional[float] = None,
        cell_pert_fraction: Optional[float] = None,
        noise_type: Optional[str] = None,
    ) -> None:
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.neuron = neuron
        self.embedding_width = embedding_width
        self.resnet_dt = resnet_dt
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.dim_case_embd = dim_case_embd
        self.trainable = trainable
        self.type_map = type_map
        self.seed = seed
        self.var_name = ["strain_components", "updated_coord", "logits"]
        if self.trainable is None:
            self.trainable = [True for ii in range(len(self.neuron) + 1)]
        if isinstance(self.trainable, bool):
            self.trainable = [self.trainable] * (len(self.neuron) + 1)
        self.activation_function = activation_function
        self.precision = precision
        if self.precision.lower() not in PRECISION_DICT:
            raise ValueError(
                f"Unsupported precision '{self.precision}'. Supported options are: {list(PRECISION_DICT.keys())}"
            )
        self.prec = PRECISION_DICT[self.precision.lower()]
        self.use_aparam_as_mask = use_aparam_as_mask
        self.mixed_types = mixed_types
        # order matters, should be place after the assignment of ntypes
        self.reinit_exclude(exclude_types)

        # init constants
        if bias_atom_e is None:
            self.bias_atom_e = np.zeros(
                [self.ntypes, self.embedding_width], dtype=GLOBAL_NP_FLOAT_PRECISION
            )
        else:
            assert bias_atom_e.shape == (self.ntypes, self.embedding_width)
            self.bias_atom_e = bias_atom_e.astype(GLOBAL_NP_FLOAT_PRECISION)
        if self.numb_fparam > 0:
            self.fparam_avg = np.zeros(self.numb_fparam, dtype=self.prec)
            self.fparam_inv_std = np.ones(self.numb_fparam, dtype=self.prec)
        else:
            self.fparam_avg, self.fparam_inv_std = None, None
        if self.numb_aparam > 0:
            self.aparam_avg = np.zeros(self.numb_aparam, dtype=self.prec)
            self.aparam_inv_std = np.ones(self.numb_aparam, dtype=self.prec)
        else:
            self.aparam_avg, self.aparam_inv_std = None, None
        if self.dim_case_embd > 0:
            self.case_embd = np.zeros(self.dim_case_embd, dtype=self.prec)
        else:
            self.case_embd = None
        # init networks
        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + self.dim_case_embd
        )
        self.coord_nets = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim,
                    self.embedding_width,
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                    seed=child_seed(seed, ii),
                )
                for ii in range(self.ntypes if not self.mixed_types else 1)
            ],
        )
        self.cell_nets = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim,
                    6,
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                    seed=child_seed(self.seed, ii),
                )
                for ii in range(self.ntypes if not self.mixed_types else 1)
            ],
        )
        self.token_nets = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim,
                    self.ntypes - 1,
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                    seed=child_seed(self.seed, ii),
                )
                for ii in range(self.ntypes if not self.mixed_types else 1)
            ],
        )

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.numb_fparam

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.numb_aparam

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return [ii for ii in range(self.ntypes) if ii not in self.exclude_types]

    def get_type_map(self) -> list[str]:
        """Get the name to each type of atoms."""
        return self.type_map

    def set_case_embd(self, case_idx: int):
        """
        Set the case embedding of this fitting net by the given case_idx,
        typically concatenated with the output of the descriptor and fed into the fitting net.
        """
        self.case_embd = np.eye(self.dim_case_embd, dtype=self.prec)[case_idx]

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert self.type_map is not None, (
            "'type_map' must be defined when performing type changing!"
        )
        assert self.mixed_types, "Only models in mixed types can perform type changing!"
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.reinit_exclude(map_atom_exclude_types(self.exclude_types, remap_index))
        if has_new_type:
            extend_shape = [len(type_map), *list(self.bias_atom_e.shape[1:])]
            extend_bias_atom_e = np.zeros(extend_shape, dtype=self.bias_atom_e.dtype)
            self.bias_atom_e = np.concatenate(
                [self.bias_atom_e, extend_bias_atom_e], axis=0
            )
        self.bias_atom_e = self.bias_atom_e[remap_index]

    def __setitem__(self, key, value) -> None:
        if key in ["bias_atom_e"]:
            self.bias_atom_e = value
        elif key in ["fparam_avg"]:
            self.fparam_avg = value
        elif key in ["fparam_inv_std"]:
            self.fparam_inv_std = value
        elif key in ["aparam_avg"]:
            self.aparam_avg = value
        elif key in ["aparam_inv_std"]:
            self.aparam_inv_std = value
        elif key in ["case_embd"]:
            self.case_embd = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ["bias_atom_e"]:
            return self.bias_atom_e
        elif key in ["fparam_avg"]:
            return self.fparam_avg
        elif key in ["fparam_inv_std"]:
            return self.fparam_inv_std
        elif key in ["aparam_avg"]:
            return self.aparam_avg
        elif key in ["aparam_inv_std"]:
            return self.aparam_inv_std
        elif key in ["case_embd"]:
            return self.case_embd
        else:
            raise KeyError(key)

    def reinit_exclude(
        self,
        exclude_types: list[int] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = AtomExcludeMask(self.ntypes, self.exclude_types)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            "@class": "Fitting",
            "@version": 3,
            "type": "denoise",
            "ntypes": self.ntypes,
            "embedding_width": self.embedding_width,
            "dim_descrpt": self.dim_descrpt,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "dim_case_embd": self.dim_case_embd,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "mixed_types": self.mixed_types,
            "cell_nets": self.cell_nets.serialize(),
            "coord_nets": self.coord_nets.serialize(),
            "token_nets": self.token_nets.serialize(),
            "exclude_types": self.exclude_types,
            "@variables": {
                "bias_atom_e": to_numpy_array(self.bias_atom_e),
                "case_embd": to_numpy_array(self.case_embd),
                "fparam_avg": to_numpy_array(self.fparam_avg),
                "fparam_inv_std": to_numpy_array(self.fparam_inv_std),
                "aparam_avg": to_numpy_array(self.aparam_avg),
                "aparam_inv_std": to_numpy_array(self.aparam_inv_std),
            },
            "type_map": self.type_map,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DenoiseFitting":
        data = data.copy()
        data.pop("@class")
        data.pop("type")
        check_version_compatibility(data.pop("@version"), 3, 1)
        variables = data.pop("@variables")
        cell_nets = data.pop("cell_nets")
        coord_nets = data.pop("coord_nets")
        token_nets = data.pop("token_nets")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = variables[kk]
        obj.cell_nets = NetworkCollection.deserialize(cell_nets)
        obj.coord_nets = NetworkCollection.deserialize(coord_nets)
        obj.token_nets = NetworkCollection.deserialize(token_nets)
        return obj

    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "strain_components",
                    [6],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                    intensive=True,
                ),
                OutputVariableDef(
                    "updated_coord",
                    [3],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
                OutputVariableDef(
                    "logits",
                    [self.ntypes - 1],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    @cast_precision
    def call(
        self,
        descriptor: np.ndarray,
        atype: np.ndarray,
        gr: Optional[np.ndarray] = None,
        g2: Optional[np.ndarray] = None,
        h2: Optional[np.ndarray] = None,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
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
        xp = array_api_compat.array_namespace(descriptor, atype)
        nf, nloc, nd = descriptor.shape
        # check input dim
        if nd != self.dim_descrpt:
            raise ValueError(
                "get an input descriptor of dim {nd},"
                "which is not consistent with {self.dim_descrpt}."
            )
        xx = descriptor

        # check fparam dim, concate to input descriptor
        if self.numb_fparam > 0:
            assert fparam is not None, "fparam should not be None"
            if fparam.shape[-1] != self.numb_fparam:
                raise ValueError(
                    f"get an input fparam of dim {fparam.shape[-1]}, "
                    f"which is not consistent with {self.numb_fparam}."
                )
            fparam = (fparam - self.fparam_avg) * self.fparam_inv_std
            fparam = xp.tile(
                xp.reshape(fparam, [nf, 1, self.numb_fparam]), (1, nloc, 1)
            )
            xx = xp.concat(
                [xx, fparam],
                axis=-1,
            )
        # check aparam dim, concate to input descriptor
        if self.numb_aparam > 0 and not self.use_aparam_as_mask:
            assert aparam is not None, "aparam should not be None"
            if aparam.shape[-1] != self.numb_aparam:
                raise ValueError(
                    f"get an input aparam of dim {aparam.shape[-1]}, "
                    f"which is not consistent with {self.numb_aparam}."
                )
            aparam = xp.reshape(aparam, [nf, nloc, self.numb_aparam])
            aparam = (aparam - self.aparam_avg) * self.aparam_inv_std
            xx = xp.concat(
                [xx, aparam],
                axis=-1,
            )

        if self.dim_case_embd > 0:
            assert self.case_embd is not None
            case_embd = xp.tile(xp.reshape(self.case_embd, [1, 1, -1]), [nf, nloc, 1])
            xx = xp.concat(
                [xx, case_embd],
                axis=-1,
            )

        # calculate the prediction
        if not self.mixed_types:
            strain_components = xp.zeros(
                [nf, nloc, 6], dtype=get_xp_precision(xp, self.precision)
            )
            updated_coord = xp.zeros(
                [nf, nloc, 3], dtype=get_xp_precision(xp, self.precision)
            )
            logits = xp.zeros(
                [nf, nloc, self.ntypes - 1], dtype=get_xp_precision(xp, self.precision)
            )
            # coord fitting
            for type_i in range(self.ntypes):
                mask = xp.tile(xp.reshape((atype == type_i), [nf, nloc, 1]), (1, 1, 3))
                updated_coord_type = self.coord_nets[(type_i,)](xx)
                assert list(updated_coord_type.shape) == [
                    nf,
                    nloc,
                    self.embedding_width,
                ]
                updated_coord_type = xp.reshape(
                    updated_coord_type, (-1, 1, self.embedding_width)
                )  # (nf * nloc, 1, embedding_width)
                gr = xp.reshape(
                    gr, (nf * nloc, -1, 3)
                )  # (nf * nloc, embedding_width, 3)
                updated_coord_type = updated_coord_type @ gr  # (nf, nloc, 3)
                updated_coord_type = xp.reshape(updated_coord_type, (nf, nloc, 3))
                updated_coord_type = xp.where(
                    mask, updated_coord_type, xp.zeros_like(updated_coord_type)
                )
                updated_coord = (
                    updated_coord + updated_coord_type
                )  # Shape is [nf, nloc, 3]
            # cell fitting
            for type_i in range(self.ntypes):
                mask = xp.tile(xp.reshape((atype == type_i), [nf, nloc, 1]), (1, 1, 6))
                strain_components_type = self.cell_nets[(type_i,)](xx)
                strain_components_type = xp.where(
                    mask, strain_components_type, xp.zeros_like(strain_components_type)
                )
                strain_components = strain_components + strain_components_type
            # token fitting
            for type_i in range(self.ntypes):
                mask = xp.tile(
                    xp.reshape((atype == type_i), [nf, nloc, 1]),
                    (1, 1, self.ntypes - 1),
                )
                logits_type = self.token_nets[(type_i,)](xx)
                logits_type = xp.where(mask, logits_type, xp.zeros_like(logits_type))
                logits = logits + logits_type
        else:
            # coord fitting
            updated_coord = self.coord_nets[()](xx)
            assert list(updated_coord.shape) == [nf, nloc, self.embedding_width]
            updated_coord = xp.reshape(
                updated_coord, (-1, 1, self.embedding_width)
            )  # (nf * nloc, 1, embedding_width)
            gr = xp.reshape(gr, (nf * nloc, -1, 3))  # (nf * nloc, embedding_width, 3)
            updated_coord = updated_coord @ gr  # (nf, nloc, 3)
            updated_coord = xp.reshape(updated_coord, (nf, nloc, 3))
            # cell fitting
            strain_components = self.cell_nets[()](xx)  # [nf, nloc, 6]
            # token fitting
            logits = self.token_nets[()](xx)  # [nf, natoms[0], ntypes-1]
        # nf x nloc
        exclude_mask = self.emask.build_type_exclude_mask(atype)
        exclude_mask = xp.astype(exclude_mask, xp.bool)
        # nf x nloc x od
        strain_components = xp.where(
            exclude_mask[:, :, None],
            strain_components,
            xp.zeros_like(strain_components),
        )
        updated_coord = xp.where(
            exclude_mask[:, :, None], updated_coord, xp.zeros_like(updated_coord)
        )
        logits = xp.where(exclude_mask[:, :, None], logits, xp.zeros_like(logits))

        return {
            "strain_components": strain_components,
            "updated_coord": updated_coord,
            "logits": logits,
        }
