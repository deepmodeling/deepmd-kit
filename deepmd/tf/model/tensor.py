# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
    Union,
)

from deepmd.tf.env import (
    MODEL_VERSION,
    tf,
)
from deepmd.tf.utils.type_embed import (
    TypeEmbedNet,
)

from .model import (
    StandardModel,
)
from .model_stat import (
    make_stat_input,
    merge_sys_stat,
)


class TensorModel(StandardModel):
    """Tensor model.

    Parameters
    ----------
    tensor_name
            Name of the tensor.
    descriptor
            Descriptor
    fitting_net
            Fitting net
    type_embedding
            Type embedding net
    type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
    data_stat_nbatch
            Number of frames used for data statistic
    data_stat_protect
            Protect parameter for atomic energy regression
    """

    def __init__(
        self,
        tensor_name: str,
        descriptor: dict,
        fitting_net: dict,
        type_embedding: Optional[Union[dict, TypeEmbedNet]] = None,
        type_map: Optional[List[str]] = None,
        data_stat_nbatch: int = 10,
        data_stat_protect: float = 1e-2,
        **kwargs,
    ) -> None:
        """Constructor."""
        super().__init__(
            descriptor=descriptor,
            fitting_net=fitting_net,
            type_embedding=type_embedding,
            type_map=type_map,
            data_stat_nbatch=data_stat_nbatch,
            data_stat_protect=data_stat_protect,
            **kwargs,
        )
        self.model_type = tensor_name

    def get_rcut(self):
        return self.rcut

    def get_ntypes(self):
        return self.ntypes

    def get_type_map(self):
        return self.type_map

    def get_sel_type(self):
        return self.fitting.get_sel_type()

    def get_out_size(self):
        return self.fitting.get_out_size()

    def data_stat(self, data):
        all_stat = make_stat_input(data, self.data_stat_nbatch, merge_sys=False)
        m_all_stat = merge_sys_stat(all_stat)
        self._compute_input_stat(m_all_stat, protection=self.data_stat_protect)
        self._compute_output_stat(all_stat)

    def _compute_input_stat(self, all_stat, protection=1e-2):
        self.descrpt.compute_input_stats(
            all_stat["coord"],
            all_stat["box"],
            all_stat["type"],
            all_stat["natoms_vec"],
            all_stat["default_mesh"],
            all_stat,
        )
        if hasattr(self.fitting, "compute_input_stats"):
            self.fitting.compute_input_stats(all_stat, protection=protection)

    def _compute_output_stat(self, all_stat):
        if hasattr(self.fitting, "compute_output_stats"):
            self.fitting.compute_output_stats(all_stat)

    def build(
        self,
        coord_,
        atype_,
        natoms,
        box,
        mesh,
        input_dict,
        frz_model=None,
        ckpt_meta: Optional[str] = None,
        suffix="",
        reuse=None,
    ):
        if input_dict is None:
            input_dict = {}
        with tf.variable_scope("model_attr" + suffix, reuse=reuse):
            t_tmap = tf.constant(" ".join(self.type_map), name="tmap", dtype=tf.string)
            t_st = tf.constant(self.get_sel_type(), name="sel_type", dtype=tf.int32)
            t_mt = tf.constant(self.model_type, name="model_type", dtype=tf.string)
            t_ver = tf.constant(MODEL_VERSION, name="model_version", dtype=tf.string)
            t_od = tf.constant(self.get_out_size(), name="output_dim", dtype=tf.int32)

        natomsel = sum(natoms[2 + type_i] for type_i in self.get_sel_type())
        nout = self.get_out_size()

        coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        atype = tf.reshape(atype_, [-1, natoms[1]])
        input_dict["nframes"] = tf.shape(coord)[0]

        # type embedding if any
        if self.typeebd is not None:
            type_embedding = self.build_type_embedding(
                self.ntypes,
                reuse=reuse,
                suffix=suffix,
                ckpt_meta=ckpt_meta,
                frz_model=frz_model,
            )
            input_dict["type_embedding"] = type_embedding
            input_dict["atype"] = atype_

        dout = self.build_descrpt(
            coord,
            atype,
            natoms,
            box,
            mesh,
            input_dict,
            frz_model=frz_model,
            ckpt_meta=ckpt_meta,
            suffix=suffix,
            reuse=reuse,
        )

        rot_mat = self.descrpt.get_rot_mat()
        rot_mat = tf.identity(rot_mat, name="o_rot_mat" + suffix)

        output = self.fitting.build(
            dout, rot_mat, natoms, input_dict, reuse=reuse, suffix=suffix
        )
        framesize = nout if "global" in self.model_type else natomsel * nout
        output = tf.reshape(
            output, [-1, framesize], name="o_" + self.model_type + suffix
        )

        model_dict = {self.model_type: output}

        if "global" not in self.model_type:
            gname = "global_" + self.model_type
            atom_out = tf.reshape(output, [-1, natomsel, nout])
            global_out = tf.reduce_sum(atom_out, axis=1)
            global_out = tf.reshape(global_out, [-1, nout], name="o_" + gname + suffix)

            out_cpnts = tf.split(atom_out, nout, axis=-1)
            force_cpnts = []
            virial_cpnts = []
            atom_virial_cpnts = []

            for out_i in out_cpnts:
                force_i, virial_i, atom_virial_i = self.descrpt.prod_force_virial(
                    out_i, natoms
                )
                force_cpnts.append(tf.reshape(force_i, [-1, 3 * natoms[1]]))
                virial_cpnts.append(tf.reshape(virial_i, [-1, 9]))
                atom_virial_cpnts.append(tf.reshape(atom_virial_i, [-1, 9 * natoms[1]]))

            # [nframe x nout x (natom x 3)]
            force = tf.concat(force_cpnts, axis=1, name="o_force" + suffix)
            # [nframe x nout x 9]
            virial = tf.concat(virial_cpnts, axis=1, name="o_virial" + suffix)
            # [nframe x nout x (natom x 9)]
            atom_virial = tf.concat(
                atom_virial_cpnts, axis=1, name="o_atom_virial" + suffix
            )

            model_dict[gname] = global_out
            model_dict["force"] = force
            model_dict["virial"] = virial
            model_dict["atom_virial"] = atom_virial

        return model_dict

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        model_type: str = "original_model",
        suffix: str = "",
    ) -> None:
        """Init the embedding net variables with the given frozen model.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        model_type : str
            the type of the model
        suffix : str
            suffix to name scope
        """
        if model_type == "original_model":
            self.descrpt.init_variables(graph, graph_def, suffix=suffix)
            self.fitting.init_variables(graph, graph_def, suffix=suffix)
            tf.constant("original_model", name="model_type", dtype=tf.string)
        elif model_type == "compressed_model":
            self.fitting.init_variables(graph, graph_def, suffix=suffix)
            tf.constant("compressed_model", name="model_type", dtype=tf.string)
        else:
            raise RuntimeError("Unknown model type %s" % model_type)


class WFCModel(TensorModel):
    def __init__(self, *args, **kwargs) -> None:
        TensorModel.__init__(self, "wfc", *args, **kwargs)


class DipoleModel(TensorModel):
    def __init__(self, *args, **kwargs) -> None:
        TensorModel.__init__(self, "dipole", *args, **kwargs)


class PolarModel(TensorModel):
    def __init__(self, *args, **kwargs) -> None:
        TensorModel.__init__(self, "polar", *args, **kwargs)


class GlobalPolarModel(TensorModel):
    def __init__(self, *args, **kwargs) -> None:
        TensorModel.__init__(self, "global_polar", *args, **kwargs)
