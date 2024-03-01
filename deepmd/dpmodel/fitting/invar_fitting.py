# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)

from .general_fitting import (
    GeneralFitting,
)


@GeneralFitting.register("invar")
@fitting_check_output
class InvarFitting(GeneralFitting):
    r"""Fitting the energy (or a rotationally invariant porperty of `dim_out`) of the system. The force and the virial can also be trained.

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
    mixed_types
            If false, different atomic types uses different fitting net, otherwise different atom types share the same fitting net.
    exclude_types: List[int]
            Atomic contributions of the excluded atom types are set zero.

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
        atom_ener: Optional[List[float]] = [],
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        layer_name: Optional[List[Optional[str]]] = None,
        use_aparam_as_mask: bool = False,
        spin: Any = None,
        mixed_types: bool = True,
        exclude_types: List[int] = [],
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

        self.dim_out = dim_out
        self.atom_ener = atom_ener
        super().__init__(
            var_name=var_name,
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
            remove_vaccum_contribution=None
            if atom_ener is None or len([x for x in atom_ener if x is not None]) == 0
            else [x is not None for x in atom_ener],
        )

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "invar"
        data["dim_out"] = self.dim_out
        data["atom_ener"] = self.atom_ener
        return data

    def _net_out_dim(self):
        """Set the FittingNet output dim."""
        return self.dim_out

    def compute_output_stats(self, merged):
        """Update the output bias for fitting net."""
        raise NotImplementedError

    def init_fitting_stat(self, result_dict):
        """Initialize the model bias by the statistics."""
        raise NotImplementedError

    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reduciable=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
            ]
        )

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
        return self._call_common(descriptor, atype, gr, g2, h2, fparam, aparam)
