# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from abc import (
    abstractmethod,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    NativeOP,
)
from deepmd.dpmodel.utils import (
    AtomExcludeMask,
    FittingNet,
    NetworkCollection,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_fitting import (
    BaseFitting,
)


class GeneralFitting(NativeOP, BaseFitting):
    r"""General fitting class.

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
    exclude_types: List[int]
            Atomic contributions of the excluded atom types are set zero.
    remove_vaccum_contribution: List[bool], optional
        Remove vaccum contribution before the bias is added. The list assigned each
        type. For `mixed_types` provide `[True]`, otherwise it should be a list of the same
        length as `ntypes` signaling if or not removing the vaccum contribution for the atom types in the list.
    """

    def __init__(
        self,
        var_name: str,
        ntypes: int,
        dim_descrpt: int,
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
        mixed_types: bool = True,
        exclude_types: List[int] = [],
        remove_vaccum_contribution: Optional[List[bool]] = None,
    ):
        self.var_name = var_name
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.neuron = neuron
        self.resnet_dt = resnet_dt
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.rcond = rcond
        self.tot_ener_zero = tot_ener_zero
        self.trainable = trainable
        if self.trainable is None:
            self.trainable = [True for ii in range(len(self.neuron) + 1)]
        if isinstance(self.trainable, bool):
            self.trainable = [self.trainable] * (len(self.neuron) + 1)
        self.activation_function = activation_function
        self.precision = precision
        self.layer_name = layer_name
        self.use_aparam_as_mask = use_aparam_as_mask
        self.spin = spin
        self.mixed_types = mixed_types
        # order matters, should be place after the assignment of ntypes
        self.reinit_exclude(exclude_types)
        if self.spin is not None:
            raise NotImplementedError("spin is not supported")
        self.remove_vaccum_contribution = remove_vaccum_contribution

        net_dim_out = self._net_out_dim()
        # init constants
        self.bias_atom_e = np.zeros([self.ntypes, net_dim_out])
        if self.numb_fparam > 0:
            self.fparam_avg = np.zeros(self.numb_fparam)
            self.fparam_inv_std = np.ones(self.numb_fparam)
        else:
            self.fparam_avg, self.fparam_inv_std = None, None
        if self.numb_aparam > 0:
            self.aparam_avg = np.zeros(self.numb_aparam)
            self.aparam_inv_std = np.ones(self.numb_aparam)
        else:
            self.aparam_avg, self.aparam_inv_std = None, None
        # init networks
        in_dim = self.dim_descrpt + self.numb_fparam + self.numb_aparam
        self.nets = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim,
                    net_dim_out,
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                )
                for ii in range(self.ntypes if not self.mixed_types else 1)
            ],
        )

    @abstractmethod
    def _net_out_dim(self):
        """Set the FittingNet output dim."""
        pass

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.numb_fparam

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.numb_aparam

    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return []

    def __setitem__(self, key, value):
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
        elif key in ["scale"]:
            self.scale = value
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
        elif key in ["scale"]:
            return self.scale
        else:
            raise KeyError(key)

    def reinit_exclude(
        self,
        exclude_types: List[int] = [],
    ):
        self.exclude_types = exclude_types
        self.emask = AtomExcludeMask(self.ntypes, self.exclude_types)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            "@class": "Fitting",
            "@version": 1,
            "var_name": self.var_name,
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "rcond": self.rcond,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "mixed_types": self.mixed_types,
            "exclude_types": self.exclude_types,
            "nets": self.nets.serialize(),
            "@variables": {
                "bias_atom_e": self.bias_atom_e,
                "fparam_avg": self.fparam_avg,
                "fparam_inv_std": self.fparam_inv_std,
                "aparam_avg": self.aparam_avg,
                "aparam_inv_std": self.aparam_inv_std,
            },
            # not supported
            "tot_ener_zero": self.tot_ener_zero,
            "trainable": self.trainable,
            "layer_name": self.layer_name,
            "use_aparam_as_mask": self.use_aparam_as_mask,
            "spin": self.spin,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class")
        data.pop("type")
        variables = data.pop("@variables")
        nets = data.pop("nets")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = variables[kk]
        obj.nets = NetworkCollection.deserialize(nets)
        return obj

    def _call_common(
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
        nf, nloc, nd = descriptor.shape
        net_dim_out = self._net_out_dim()
        # check input dim
        if nd != self.dim_descrpt:
            raise ValueError(
                "get an input descriptor of dim {nd},"
                "which is not consistent with {self.dim_descrpt}."
            )
        xx = descriptor
        if self.remove_vaccum_contribution is not None:
            # TODO: Idealy, the input for vaccum should be computed;
            # we consider it as always zero for convenience.
            # Needs a compute_input_stats for vaccum passed from the
            # descriptor.
            xx_zeros = np.zeros_like(xx)
        else:
            xx_zeros = None
        # check fparam dim, concate to input descriptor
        if self.numb_fparam > 0:
            assert fparam is not None, "fparam should not be None"
            if fparam.shape[-1] != self.numb_fparam:
                raise ValueError(
                    "get an input fparam of dim {fparam.shape[-1]}, ",
                    "which is not consistent with {self.numb_fparam}.",
                )
            fparam = (fparam - self.fparam_avg) * self.fparam_inv_std
            fparam = np.tile(fparam.reshape([nf, 1, self.numb_fparam]), [1, nloc, 1])
            xx = np.concatenate(
                [xx, fparam],
                axis=-1,
            )
            if xx_zeros is not None:
                xx_zeros = np.concatenate(
                    [xx_zeros, fparam],
                    axis=-1,
                )
        # check aparam dim, concate to input descriptor
        if self.numb_aparam > 0:
            assert aparam is not None, "aparam should not be None"
            if aparam.shape[-1] != self.numb_aparam:
                raise ValueError(
                    "get an input aparam of dim {aparam.shape[-1]}, ",
                    "which is not consistent with {self.numb_aparam}.",
                )
            aparam = aparam.reshape([nf, nloc, self.numb_aparam])
            aparam = (aparam - self.aparam_avg) * self.aparam_inv_std
            xx = np.concatenate(
                [xx, aparam],
                axis=-1,
            )
            if xx_zeros is not None:
                xx_zeros = np.concatenate(
                    [xx_zeros, aparam],
                    axis=-1,
                )

        # calcualte the prediction
        if not self.mixed_types:
            outs = np.zeros([nf, nloc, net_dim_out])
            for type_i in range(self.ntypes):
                mask = np.tile(
                    (atype == type_i).reshape([nf, nloc, 1]), [1, 1, net_dim_out]
                )
                atom_property = self.nets[(type_i,)](xx)
                if self.remove_vaccum_contribution is not None and not (
                    len(self.remove_vaccum_contribution) > type_i
                    and not self.remove_vaccum_contribution[type_i]
                ):
                    assert xx_zeros is not None
                    atom_property -= self.nets[(type_i,)](xx_zeros)
                atom_property = atom_property + self.bias_atom_e[type_i]
                atom_property = atom_property * mask
                outs = outs + atom_property  # Shape is [nframes, natoms[0], 1]
        else:
            outs = self.nets[()](xx) + self.bias_atom_e[atype]
            if xx_zeros is not None:
                outs -= self.nets[()](xx_zeros)
        # nf x nloc
        exclude_mask = self.emask.build_type_exclude_mask(atype)
        # nf x nloc x nod
        outs = outs * exclude_mask[:, :, None]
        return {self.var_name: outs}
