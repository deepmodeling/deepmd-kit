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
    fitting_check_output,
)

from .general_fitting import (
    GeneralFitting,
)


@fitting_check_output
class DipoleFitting(GeneralFitting):
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
    dim_rot_mat : int
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
        dim_rot_mat: int,
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
        exclude_types: List[int] = [],
        old_impl=False,
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

        self.dim_rot_mat = dim_rot_mat
        super().__init__(
            var_name=var_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=dim_out,
            neuron=neuron,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            rcond=rcond,
            tot_ener_zero=tot_ener_zero,
            trainable=trainable,
            atom_ener=atom_ener,
            activation_function=activation_function,
            precision=precision,
            layer_name=layer_name,
            use_aparam_as_mask=use_aparam_as_mask,
            spin=spin,
            distinguish_types=distinguish_types,
            exclude_types=exclude_types,
        )
        self.old_impl = False

    def _net_out_dim(self):
        """Set the FittingNet output dim."""
        return self.dim_rot_mat

    def serialize(self) -> dict:
        data = super().serialize()
        data["dim_rot_mat"] = self.dim_rot_mat
        data["old_impl"] = self.old_impl
        return data

    def call(
        self,
        descriptor: np.array,
        atype: np.array,
        gr: Optional[np.array] = None,
        g2: Optional[np.array] = None,
        h2: Optional[np.array] = None,
        fparam: Optional[np.array] = None,
        aparam: Optional[np.array] = None,
    ) -> Dict[str, np.array]:
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
        assert gr is not None, "Must provide the rotation matrix for dipole fitting."
        # (nframes, nloc, m1)
        out = self._call_common(descriptor, atype, gr, g2, h2, fparam, aparam)[
            self.var_name
        ]
        # (nframes * nloc, 1, m1)
        out = out.reshape(-1, 1, self.dim_rot_mat)
        # (nframes * nloc, m1, 3)
        gr = gr.reshape(nframes * nloc, -1, 3)
        # (nframes, nloc, 3)
        out = np.einsum("bim,bmj->bij", out, gr).squeeze(-2).reshape(nframes, nloc, 3)
        return {self.var_name: out}
