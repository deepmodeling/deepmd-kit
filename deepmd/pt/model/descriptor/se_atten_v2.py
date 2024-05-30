# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
    Union,
    Tuple
)

from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.descriptor.se_atten import (
    DescrptBlockSeAtten,
)


@DescriptorBlock.register("se_atten_v2")
class DescrptBlockSeAttenV2(DescrptBlockSeAtten):
    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: Union[List[int], int],
        ntypes: int,
        neuron: list = [25, 50, 100],
        axis_neuron: int = 16,
        tebd_dim: int = 8,
        set_davg_zero: bool = True,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        activation_function="tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        scaling_factor=1.0,
        normalize=True,
        temperature=None,
        type_one_side: bool = False,
        exclude_types: List[Tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        seed: Optional[int] = None,
        type: Optional[str] = None,
        old_impl: bool = False,
    ) -> None:
        r"""Construct an embedding net of type `se_atten`.

        Parameters
        ----------
        rcut : float
            The cut-off radius :math:`r_c`
        rcut_smth : float
            From where the environment matrix should be smoothed :math:`r_s`
        sel : list[int], int
            list[int]: sel[i] specifies the maxmum number of type i atoms in the cut-off radius
            int: the total maxmum number of atoms in the cut-off radius
        ntypes : int
            Number of element types
        neuron : list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
        axis_neuron : int
            Number of the axis neuron :math:`M_2` (number of columns of the sub-matrix of the embedding matrix)
        tebd_dim : int
            Dimension of the type embedding
        resnet_dt : bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
        trainable_ln : bool
            Whether to use trainable shift and scale weights in layer normalization.
        ln_eps : float, Optional
            The epsilon value for layer normalization.
        type_one_side : bool
            If 'False', type embeddings of both neighbor and central atoms are considered.
            If 'True', only type embeddings of neighbor atoms are considered.
            Default is 'False'.
        attn : int
            Hidden dimension of the attention vectors
        attn_layer : int
            Number of attention layers
        attn_dotr : bool
            If dot the angular gate to the attention weights
        attn_mask : bool
            (Only support False to keep consistent with other backend references.)
            (Not used in this version.)
            If mask the diagonal of attention weights
        exclude_types : List[List[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
        env_protection : float
            Protection parameter to prevent division by zero errors during environment matrix calculations.
        set_davg_zero : bool
            Set the shift of embedding net input to zero.
        activation_function : str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
        precision : str
            The precision of the embedding net parameters. Supported options are |PRECISION|
        scaling_factor : float
            The scaling factor of normalization in calculations of attention weights.
            If `temperature` is None, the scaling of attention weights is (N_dim * scaling_factor)**0.5
        normalize : bool
            Whether to normalize the hidden vectors in attention weights calculation.
        temperature : float
            If not None, the scaling of attention weights is `temperature` itself.
        seed : int, Optional
            Random seed for parameter initialization.
        """
        DescrptBlockSeAtten.__init__(
            self,
            rcut,
            rcut_smth,
            sel,
            ntypes,
            neuron=neuron,
            axis_neuron=axis_neuron,
            tebd_dim=tebd_dim,
            tebd_input_mode="strip",
            set_davg_zero=set_davg_zero,
            attn=attn,
            attn_layer=attn_layer,
            attn_dotr=attn_dotr,
            attn_mask=attn_mask,
            activation_function=activation_function,
            precision=precision,
            resnet_dt=resnet_dt,
            scaling_factor=scaling_factor,
            normalize=normalize,
            temperature=temperature,
            smooth=True,
            type_one_side=type_one_side,
            exclude_types=exclude_types,
            env_protection=env_protection,
            trainable_ln=trainable_ln,
            ln_eps=ln_eps,
            seed=seed,
            type=type,
            old_impl=old_impl,
        )
