import logging
from typing import List
from typing import Optional
import pdb
import numpy as np
from paddle import nn

from deepmd.common import add_data_requirement
from deepmd.common import cast_precision
from deepmd.common import get_activation_func
from deepmd.common import get_precision
from deepmd.env import GLOBAL_PD_FLOAT_PRECISION
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import global_cvt_2_pd_float
from deepmd.env import global_cvt_2_tf_float
from deepmd.env import paddle
from deepmd.env import tf
from deepmd.fit.fitting import Fitting
from deepmd.utils.graph import get_fitting_net_variables_from_graph_def
from deepmd.utils.network import one_layer
from deepmd.utils.network import one_layer_rand_seed_shift
# from deepmd.infer import DeepPotential
from deepmd.nvnmd.fit.ener import one_layer_nvnmd
from deepmd.nvnmd.utils.config import nvnmd_cfg
from deepmd.utils.errors import GraphWithoutTensorError
from deepmd.utils.graph import get_fitting_net_variables_from_graph_def
from deepmd.utils.graph import get_tensor_by_name_from_graph
from deepmd.utils.network import OneLayer as OneLayer_deepmd
from deepmd.utils.network import one_layer as one_layer_deepmd
from deepmd.utils.network import one_layer_rand_seed_shift
from deepmd.utils.spin import Spin

# @Fitting.register("dipole")
class DipoleFittingSeA(nn.Layer):
    r"""Fit the atomic dipole with descriptor se_a.

    Parameters
    ----------
    descrpt : tf.Tensor
            The descrptor
    neuron : List[int]
            Number of neurons in each hidden layer of the fitting net
    resnet_dt : bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    sel_type : List[int]
            The atom types selected to have an atomic dipole prediction. If is None, all atoms are selected.
    seed : int
            Random seed for initializing the network parameters.
    activation_function : str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision : str
            The precision of the embedding net parameters. Supported options are |PRECISION|
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    """

    def __init__(
        self,
        descrpt: paddle.Tensor,
        neuron: List[int] = [120, 120, 120],
        resnet_dt: bool = True,
        sel_type: Optional[List[int]] = None,
        seed: Optional[int] = None,
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
    ) -> None:
        super().__init__(name_scope="DipoleFittingSeA")
        """Constructor."""
        self.ntypes = descrpt.get_ntypes()#2
        self.dim_descrpt = descrpt.get_dim_out()
        self.n_neuron = neuron
        self.resnet_dt = resnet_dt
        self.sel_type = sel_type
        if self.sel_type is None:
            self.sel_type = [ii for ii in range(self.ntypes)]
        self.sel_mask = np.array(
            [ii in self.sel_type for ii in range(self.ntypes)], dtype=bool
        )
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.ntypes_spin = 0
        self.seed_shift = one_layer_rand_seed_shift()
        self.fitting_activation_fn = get_activation_func(activation_function)
        self.fitting_precision = get_precision(precision)
        self.dim_rot_mat_1 = descrpt.get_dim_rot_mat_1()
        self.dim_rot_mat = self.dim_rot_mat_1 * 3
        self.useBN = False
        self.fitting_net_variables = None
        self.mixed_prec = None

        type_suffix = ""
        suffix = ""
        self.one_layers = nn.LayerList()
        self.final_layers = nn.LayerList()
        ntypes_atom = self.ntypes - self.ntypes_spin
        for type_i in range(0, ntypes_atom):
            type_i_layers = nn.LayerList()
            for ii in range(0, len(self.n_neuron)):

                layer_suffix = "layer_" + str(ii) + type_suffix + suffix

                if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii - 1]:
                    type_i_layers.append(
                        OneLayer_deepmd(
                            self.n_neuron[ii - 1],
                            self.n_neuron[ii],
                            activation_fn=self.fitting_activation_fn,
                            precision=self.fitting_precision,
                            name=layer_suffix,
                            seed=self.seed,
                            use_timestep=self.resnet_dt,
                        )
                    )
                else:
                    type_i_layers.append(
                        OneLayer_deepmd(
                            self.dim_descrpt,
                            self.n_neuron[ii],
                            activation_fn=self.fitting_activation_fn,
                            precision=self.fitting_precision,
                            name=layer_suffix,
                            seed=self.seed,
                        )
                    )
                if (not self.uniform_seed) and (self.seed is not None):
                    self.seed += self.seed_shift

            self.one_layers.append(type_i_layers)
            self.final_layers.append(
                OneLayer_deepmd(
                    self.n_neuron[-1],
                    self.dim_rot_mat_1,
                    activation_fn=None,
                    precision=self.fitting_precision,
                    name=layer_suffix,
                    seed=self.seed,
                )
            )

    def get_sel_type(self) -> int:
        """Get selected type."""
        return self.sel_type

    def get_out_size(self) -> int:
        """Get the output size. Should be 3."""
        return 3
    
    def _build_lower(self, start_index, natoms, inputs, rot_mat, suffix="", reuse=None,
                     type_i=None,
    ):
        # cut-out inputs
        inputs_i = paddle.slice(
            inputs,
            [0, 1, 2],
            [0, start_index, 0],
            [inputs.shape[0], start_index + natoms, inputs.shape[2]],
        )   
        inputs_i = paddle.reshape(inputs_i, [-1, self.dim_descrpt])
        rot_mat_i = paddle.slice(
            rot_mat,
            [0, 1, 2],
            [0, start_index, 0],
            [rot_mat.shape[0], start_index + natoms, rot_mat.shape[2]],
        )   
        # paddle.slice(rot_mat, [0, start_index, 0], [-1, natoms, -1])
        rot_mat_i = paddle.reshape(rot_mat_i, [-1, self.dim_rot_mat_1, 3])
        layer = inputs_i
        for ii in range(0, len(self.n_neuron)):
            if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii - 1]:
                layer += self.one_layers[type_i][ii](layer)
            else:
                layer = self.one_layers[type_i][ii](layer)            
            # if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii - 1]:
            #     layer += one_layer(
            #         layer,
            #         self.n_neuron[ii],
            #         name="layer_" + str(ii) + suffix,
            #         reuse=reuse,
            #         seed=self.seed,
            #         use_timestep=self.resnet_dt,
            #         activation_fn=self.fitting_activation_fn,
            #         precision=self.fitting_precision,
            #         uniform_seed=self.uniform_seed,
            #         initial_variables=self.fitting_net_variables,
            #         mixed_prec=self.mixed_prec,
            #     )
            # else:
            #     layer = one_layer(
            #         layer,
            #         self.n_neuron[ii],
            #         name="layer_" + str(ii) + suffix,
            #         reuse=reuse,
            #         seed=self.seed,
            #         activation_fn=self.fitting_activation_fn,
            #         precision=self.fitting_precision,
            #         uniform_seed=self.uniform_seed,
            #         initial_variables=self.fitting_net_variables,
            #         mixed_prec=self.mixed_prec,
            #     )
            if (not self.uniform_seed) and (self.seed is not None):
                self.seed += self.seed_shift

        final_layer = self.final_layers[type_i](
            layer,
        )
        # # (nframes x natoms) x naxis
        # final_layer = one_layer(
        #     layer,
        #     self.dim_rot_mat_1,
        #     activation_fn=None,
        #     name="final_layer" + suffix,
        #     reuse=reuse,
        #     seed=self.seed,
        #     precision=self.fitting_precision,
        #     uniform_seed=self.uniform_seed,
        #     initial_variables=self.fitting_net_variables,
        #     mixed_prec=self.mixed_prec,
        #     final_layer=True,
        # )
        if (not self.uniform_seed) and (self.seed is not None):
            self.seed += self.seed_shift
        # (nframes x natoms) x 1 * naxis
        final_layer = paddle.reshape(
            final_layer, [paddle.shape(inputs)[0] * natoms, 1, self.dim_rot_mat_1]
        )#natoms=64, self.dim_rot_mat_1=100
        # (nframes x natoms) x 1 x 3(coord)
        final_layer = paddle.matmul(final_layer, rot_mat_i)
        # nframes x natoms x 3
        final_layer = paddle.reshape(final_layer, [paddle.shape(inputs)[0], natoms, 3])
        # pdb.set_trace()        
        return final_layer # [1, 64, 3]
    

    def forward(
        self,
        input_d: paddle.Tensor,
        rot_mat: paddle.Tensor,
        natoms: paddle.Tensor,
        input_dict: Optional[dict] = None,
        reuse: Optional[bool] = None,
        suffix: str = "",
    ) -> paddle.Tensor:
        """Build the computational graph for fitting net.

        Parameters
        ----------
        input_d
            The input descriptor
        rot_mat
            The rotation matrix from the descriptor.
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        input_dict
            Additional dict for inputs.
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        dipole
            The atomic dipole.
        """
        if input_dict is None:
            input_dict = {}
        type_embedding = input_dict.get("type_embedding", None)
        atype = input_dict.get("atype", None)
        nframes = input_dict.get("nframes")
        start_index = 0
        inputs = paddle.reshape(input_d, [-1, natoms[0], self.dim_descrpt])
        rot_mat = paddle.reshape(rot_mat, [-1, natoms[0], self.dim_rot_mat])

        if type_embedding is not None:
            nloc_mask = paddle.reshape(
                paddle.tile(paddle.repeat(self.sel_mask, natoms[2:]), [nframes]), [nframes, -1]
            )
            atype_nall = paddle.reshape(atype, [-1, natoms[1]])
            # (nframes x nloc_masked)
            self.atype_nloc_masked = paddle.reshape(
                paddle.slice(atype_nall, [0, 0], [-1, natoms[0]])[nloc_mask], [-1]
            )  ## lammps will make error
            self.nloc_masked = paddle.shape(
                paddle.reshape(self.atype_nloc_masked, [nframes, -1])
            )[1]
            atype_embed = paddle.nn.embedding_lookup(type_embedding, self.atype_nloc_masked)
        else:
            atype_embed = None

        self.atype_embed = atype_embed

        if atype_embed is None:
            count = 0
            outs_list = []
            # pdb.set_trace()
            for type_i in range(self.ntypes):#2
                if type_i not in self.sel_type:
                    start_index += natoms[2 + type_i]
                    continue #sel_type是0，所以就循环了一次
                final_layer = self._build_lower(
                    start_index,
                    natoms[2 + type_i],
                    inputs,
                    rot_mat,
                    suffix="_type_" + str(type_i) + suffix,
                    reuse=reuse,
                    type_i=type_i,
                )
                start_index += natoms[2 + type_i]
                # concat the results
                outs_list.append(final_layer)
                count += 1
                # pdb.set_trace()

            outs = paddle.concat(outs_list, axis=1) # [1, 64, 3]
        else:
            inputs = paddle.reshape(
                paddle.reshape(inputs, [nframes, natoms[0], self.dim_descrpt])[nloc_mask],
                [-1, self.dim_descrpt],
            )
            rot_mat = paddle.reshape(
                paddle.reshape(rot_mat, [nframes, natoms[0], self.dim_rot_mat_1 * 3])[
                    nloc_mask
                ],
                [-1, self.dim_rot_mat_1, 3],
            )
            atype_embed = paddle.cast(atype_embed, self.fitting_precision)
            type_shape = atype_embed.get_shape().as_list()
            inputs = paddle.concat([inputs, atype_embed], axis=1)
            self.dim_descrpt = self.dim_descrpt + type_shape[1]
            inputs = paddle.reshape(inputs, [nframes, self.nloc_masked, self.dim_descrpt])
            rot_mat = paddle.reshape(
                rot_mat, [nframes, self.nloc_masked, self.dim_rot_mat_1 * 3]
            )
            final_layer = self._build_lower(
                0, self.nloc_masked, inputs, rot_mat, suffix=suffix, reuse=reuse
            )
            pdb.set_trace()

            # nframes x natoms x 3
            outs = paddle.reshape(final_layer, [nframes, self.nloc_masked, 3])

        # paddle.summary.histogram("fitting_net_output", outs)
        return paddle.reshape(outs, [-1])
        # return tf.reshape(outs, [tf.shape(inputs)[0] * natoms[0] * 3 // 3])

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        suffix: str = "",
    ) -> None:
        """Init the fitting net variables with the given dict.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str
            suffix to name scope
        """
        self.fitting_net_variables = get_fitting_net_variables_from_graph_def(
            graph_def, suffix=suffix
        )

    def enable_mixed_precision(self, mixed_prec: Optional[dict] = None) -> None:
        """Reveive the mixed precision setting.

        Parameters
        ----------
        mixed_prec
            The mixed precision setting used in the embedding net
        """
        self.mixed_prec = mixed_prec
        self.fitting_precision = get_precision(mixed_prec["output_prec"])
