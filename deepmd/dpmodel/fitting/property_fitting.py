# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import numpy as np

from deepmd.dpmodel.common import (
    DEFAULT_PRECISION,
)
from deepmd.dpmodel.fitting.invar_fitting import (
    InvarFitting,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@InvarFitting.register("property")
class PropertyFittingNet(InvarFitting):
    r"""Fitting the rotationally invariant properties of `task_dim` of the system.

    Parameters
    ----------
    ntypes
            The number of atom types.
    dim_descrpt
            The dimension of the input descriptor.
    task_dim
            The dimension of outputs of fitting net.
    neuron
            Number of neurons :math:`N` in each hidden layer of the fitting net
    bias_atom_p
            Average property per atom for each element.
    rcond
            The condition number for the regression of atomic energy.
    trainable
            If the weights of fitting net are trainable.
            Suppose that we have :math:`N_l` hidden layers in the fitting net,
            this list is of length :math:`N_l + 1`, specifying if the hidden layers and the output layer are trainable.
    intensive
            Whether the fitting property is intensive.
    property_name:
            The name of fitting property, which should be consistent with the property name in the dataset.
            If the data file is named `humo.npy`, this parameter should be "humo".
    resnet_dt
            Time-step `dt` in the resnet construction:
            :math:`y = x + dt * \phi (Wx + b)`
    numb_fparam
            Number of frame parameter
    numb_aparam
            Number of atomic parameter
    activation_function
            The activation function :math:`\boldsymbol{\phi}` in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    mixed_types
            If false, different atomic types uses different fitting net, otherwise different atom types share the same fitting net.
    exclude_types: list[int]
            Atomic contributions of the excluded atom types are set zero.
    type_map: list[str], Optional
            A list of strings. Give the name to each type of atoms.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        task_dim: int = 1,
        neuron: list[int] = [128, 128, 128],
        bias_atom_p: Optional[np.ndarray] = None,
        rcond: Optional[float] = None,
        trainable: Union[bool, list[bool]] = True,
        intensive: bool = False,
        property_name: str = "property",
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        exclude_types: list[int] = [],
        type_map: Optional[list[str]] = None,
        # not used
        seed: Optional[int] = None,
    ) -> None:
        self.task_dim = task_dim
        self.intensive = intensive
        super().__init__(
            var_name=property_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=task_dim,
            neuron=neuron,
            bias_atom=bias_atom_p,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            rcond=rcond,
            trainable=trainable,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            exclude_types=exclude_types,
            type_map=type_map,
        )

    @classmethod
    def deserialize(cls, data: dict) -> "PropertyFittingNet":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 4, 1)
        data.pop("dim_out")
        data["property_name"] = data.pop("var_name")
        data.pop("tot_ener_zero")
        data.pop("layer_name")
        data.pop("use_aparam_as_mask", None)
        data.pop("spin", None)
        data.pop("atom_ener", None)
        obj = super().deserialize(data)

        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        dd = {
            **InvarFitting.serialize(self),
            "type": "property",
            "task_dim": self.task_dim,
            "intensive": self.intensive,
        }
        dd["@version"] = 4

        return dd
