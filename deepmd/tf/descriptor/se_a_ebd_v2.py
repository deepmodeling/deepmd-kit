# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    List,
    Optional,
)

from deepmd.tf.utils.spin import (
    Spin,
)

from .descriptor import (
    Descriptor,
)
from .se_a import (
    DescrptSeA,
)

log = logging.getLogger(__name__)


@Descriptor.register("se_a_tpe_v2")
@Descriptor.register("se_a_ebd_v2")
class DescrptSeAEbdV2(DescrptSeA):
    r"""A compressible se_a_ebd model.

    This model is a warpper for DescriptorSeA, which set stripped_type_embedding=True.
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: List[str],
        neuron: List[int] = [24, 48, 96],
        axis_neuron: int = 8,
        resnet_dt: bool = False,
        trainable: bool = True,
        seed: Optional[int] = None,
        type_one_side: bool = True,
        exclude_types: List[List[int]] = [],
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
        multi_task: bool = False,
        spin: Optional[Spin] = None,
        **kwargs,
    ) -> None:
        DescrptSeA.__init__(
            self,
            rcut,
            rcut_smth,
            sel,
            neuron=neuron,
            axis_neuron=axis_neuron,
            resnet_dt=resnet_dt,
            trainable=trainable,
            seed=seed,
            type_one_side=type_one_side,
            exclude_types=exclude_types,
            set_davg_zero=set_davg_zero,
            activation_function=activation_function,
            precision=precision,
            uniform_seed=uniform_seed,
            multi_task=multi_task,
            spin=spin,
            stripped_type_embedding=True,
            **kwargs,
        )
