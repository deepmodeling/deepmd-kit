# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Any,
    List,
    Optional,
)

import numpy as np

from .common import (
    DEFAULT_PRECISION,
    NativeOP,
)
from .network import (
    FittingNet,
    NetworkCollection,
)
from .output_def import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)


@fitting_check_output
class InvarFitting(NativeOP):
    r"""Fitting the energy (or a porperty of `dim_out`) of the system. The force and the virial can also be trained.

    Lets take the energy fitting task as an example.
    The potential energy :math:`E` is a fitting network function of the descriptor :math:`\mathcal{D}`:

    .. math::
        E(\mathcal{D}) = \mathcal{L}^{(n)} \circ \mathcal{L}^{(n-1)}
        \circ \cdots \circ \mathcal{L}^{(1)} \circ \mathcal{L}^{(0)}

    The first :math:`n` hidden layers :math:`\mathcal{L}^{(0)}, \cdots, \mathcal{L}^{(n-1)}` are given by

    .. math::
        \mathbf{y}=\mathcal{L}(\mathbf{x};\mathbf{w},\mathbf{b})=
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b})

    where :math:`\mathbf{x} \in \mathbb{R}^{N_1}` is the input vector and :math:`\mathbf{y} \in \mathbb{R}^{N_2}`
    is the output vector. :math:`\mathbf{w} \in \mathbb{R}^{N_1 \times N_2}` and
    :math:`\mathbf{b} \in \mathbb{R}^{N_2}` are weights and biases, respectively,
    both of which are trainable if `trainable[i]` is `True`. :math:`\boldsymbol{\phi}`
    is the activation function.

    The output layer :math:`\mathcal{L}^{(n)}` is given by

    .. math::
        \mathbf{y}=\mathcal{L}^{(n)}(\mathbf{x};\mathbf{w},\mathbf{b})=
            \mathbf{x}^T\mathbf{w}+\mathbf{b}

    where :math:`\mathbf{x} \in \mathbb{R}^{N_{n-1}}` is the input vector and :math:`\mathbf{y} \in \mathbb{R}`
    is the output scalar. :math:`\mathbf{w} \in \mathbb{R}^{N_{n-1}}` and
    :math:`\mathbf{b} \in \mathbb{R}` are weights and bias, respectively,
    both of which are trainable if `trainable[n]` is `True`.

    Parameters
    ----------
    var_name
            The name of the output variable.
    ntypes
            The number of atom types.
    dim_descrpt
            The dimension of the input descriptor.
    dim_out
            The dimension of the output fit property.
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
    atom_ener
            Specifying atomic energy contribution in vacuum. The `set_davg_zero` key in the descrptor should be set.
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
    distinguish_types
            Different atomic types uses different fitting net.

    """

    def __init__(
        self,
        var_name: str,
        ntypes: int,
        dim_descrpt: int,
        dim_out: int,
        neuron: List[int] = [120, 120, 120],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        rcond: Optional[float] = None,
        tot_ener_zero: bool = False,
        trainable: Optional[List[bool]] = None,
        atom_ener: Optional[List[float]] = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        layer_name: Optional[List[Optional[str]]] = None,
        use_aparam_as_mask: bool = False,
        spin: Any = None,
        distinguish_types: bool = False,
    ):
        # seed, uniform_seed are not included
        if tot_ener_zero:
            raise NotImplementedError("tot_ener_zero is not implemented")
        if spin is not None:
            raise NotImplementedError("spin is not implemented")
        if use_aparam_as_mask:
            raise NotImplementedError("use_aparam_as_mask is not implemented")
        if use_aparam_as_mask:
            raise NotImplementedError("use_aparam_as_mask is not implemented")
        if layer_name is not None:
            raise NotImplementedError("layer_name is not implemented")
        if atom_ener is not None:
            raise NotImplementedError("atom_ener is not implemented")

        self.var_name = var_name
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.dim_out = dim_out
        self.neuron = neuron
        self.resnet_dt = resnet_dt
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.rcond = rcond
        self.tot_ener_zero = tot_ener_zero
        self.trainable = trainable
        self.atom_ener = atom_ener
        self.activation_function = activation_function
        self.precision = precision
        self.layer_name = layer_name
        self.use_aparam_as_mask = use_aparam_as_mask
        self.spin = spin
        self.distinguish_types = distinguish_types
        if self.spin is not None:
            raise NotImplementedError("spin is not supported")

        # init constants
        self.bias_atom_e = np.zeros([self.ntypes, self.dim_out])
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
        out_dim = self.dim_out
        self.nets = NetworkCollection(
            1 if self.distinguish_types else 0,
            self.ntypes,
            network_type="fitting_network",
            networks=[
                FittingNet(
                    in_dim,
                    out_dim,
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                    bias_out=True,
                )
                for ii in range(self.ntypes if self.distinguish_types else 1)
            ],
        )

    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name, [self.dim_out], reduciable=True, differentiable=True
                ),
            ]
        )

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
        else:
            raise KeyError(key)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            "var_name": self.var_name,
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "dim_out": self.dim_out,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "rcond": self.rcond,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "distinguish_types": self.distinguish_types,
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
            "atom_ener": self.atom_ener,
            "layer_name": self.layer_name,
            "use_aparam_as_mask": self.use_aparam_as_mask,
            "spin": self.spin,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "InvarFitting":
        data = copy.deepcopy(data)
        variables = data.pop("@variables")
        nets = data.pop("nets")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = variables[kk]
        obj.nets = NetworkCollection.deserialize(nets)
        return obj

    def call(
        self,
        descriptor: np.array,
        atype: np.array,
        gr: Optional[np.array] = None,
        g2: Optional[np.array] = None,
        h2: Optional[np.array] = None,
        fparam: Optional[np.array] = None,
        aparam: Optional[np.array] = None,
    ):
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
                    "get an input fparam of dim {fparam.shape[-1]}, ",
                    "which is not consistent with {self.numb_fparam}.",
                )
            fparam = (fparam - self.fparam_avg) * self.fparam_inv_std
            fparam = np.tile(fparam.reshape([nf, 1, -1]), [1, nloc, 1])
            xx = np.concatenate(
                [xx, fparam],
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
            aparam = (aparam - self.aparam_avg) * self.aparam_inv_std
            xx = np.concatenate(
                [xx, aparam],
                axis=-1,
            )

        # calcualte the prediction
        if self.distinguish_types:
            outs = np.zeros([nf, nloc, self.dim_out])
            for type_i in range(self.ntypes):
                mask = np.tile(
                    (atype == type_i).reshape([nf, nloc, 1]), [1, 1, self.dim_out]
                )
                atom_energy = self.nets[(type_i,)](xx)
                atom_energy = atom_energy + self.bias_atom_e[type_i]
                atom_energy = atom_energy * mask
                outs = outs + atom_energy  # Shape is [nframes, natoms[0], 1]
        else:
            outs = self.nets[()](xx) + self.bias_atom_e[atype]
        return {self.var_name: outs}
