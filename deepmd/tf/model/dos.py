# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
    Union,
)

from deepmd.tf.env import (
    MODEL_VERSION,
    global_cvt_2_ener_float,
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


class DOSModel(StandardModel):
    """DOS model.

    Parameters
    ----------
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

    model_type = "dos"

    def __init__(
        self,
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
        # fitting
        self.numb_dos = self.fitting.get_numb_dos()
        self.numb_fparam = self.fitting.get_numb_fparam()
        self.numb_aparam = self.fitting.get_numb_aparam()

    def get_numb_dos(self):
        return self.numb_dos

    def get_rcut(self):
        return self.rcut

    def get_ntypes(self):
        return self.ntypes

    def get_type_map(self):
        return self.type_map

    def get_numb_fparam(self) -> int:
        """Get the number of frame parameters."""
        return self.numb_fparam

    def get_numb_aparam(self) -> int:
        """Get the number of atomic parameters."""
        return self.numb_aparam

    def data_stat(self, data):
        all_stat = make_stat_input(data, self.data_stat_nbatch, merge_sys=False)
        m_all_stat = merge_sys_stat(all_stat)
        self._compute_input_stat(
            m_all_stat, protection=self.data_stat_protect, mixed_type=data.mixed_type
        )
        # self._compute_output_stat(all_stat, mixed_type=data.mixed_type)
        # self.bias_atom_e = data.compute_energy_shift(self.rcond)

    def _compute_input_stat(self, all_stat, protection=1e-2, mixed_type=False):
        if mixed_type:
            self.descrpt.compute_input_stats(
                all_stat["coord"],
                all_stat["box"],
                all_stat["type"],
                all_stat["natoms_vec"],
                all_stat["default_mesh"],
                all_stat,
                mixed_type,
                all_stat["real_natoms_vec"],
            )
        else:
            self.descrpt.compute_input_stats(
                all_stat["coord"],
                all_stat["box"],
                all_stat["type"],
                all_stat["natoms_vec"],
                all_stat["default_mesh"],
                all_stat,
            )
        self.fitting.compute_input_stats(all_stat, protection=protection)

    def _compute_output_stat(self, all_stat, mixed_type=False):
        if mixed_type:
            self.fitting.compute_output_stats(all_stat, mixed_type=mixed_type)
        else:
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
            t_mt = tf.constant(self.model_type, name="model_type", dtype=tf.string)
            t_ver = tf.constant(MODEL_VERSION, name="model_version", dtype=tf.string)
            t_od = tf.constant(self.numb_dos, name="output_dim", dtype=tf.int32)

        coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        atype = tf.reshape(atype_, [-1, natoms[1]])
        input_dict["nframes"] = tf.shape(coord)[0]

        # type embedding if any
        if self.typeebd is not None:
            type_embedding = self.build_type_embedding(
                self.ntypes,
                reuse=reuse,
                suffix=suffix,
                frz_model=frz_model,
                ckpt_meta=ckpt_meta,
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

        atom_dos = self.fitting.build(
            dout, natoms, input_dict, reuse=reuse, suffix=suffix
        )
        self.atom_dos = atom_dos

        dos_raw = atom_dos

        dos_raw = tf.reshape(dos_raw, [natoms[0], -1], name="o_atom_dos" + suffix)
        dos = tf.reduce_sum(
            global_cvt_2_ener_float(dos_raw), axis=0, name="o_dos" + suffix
        )

        model_dict = {}
        model_dict["dos"] = dos
        model_dict["atom_dos"] = dos_raw
        model_dict["coord"] = coord
        model_dict["atype"] = atype

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
        # self.frz_model will control the self.model to import the descriptor from the given frozen model instead of building from scratch...
        # initialize fitting net with the given compressed frozen model
        if model_type == "original_model":
            self.descrpt.init_variables(graph, graph_def, suffix=suffix)
            self.fitting.init_variables(graph, graph_def, suffix=suffix)
            tf.constant("original_model", name="model_type", dtype=tf.string)
        elif model_type == "compressed_model":
            self.fitting.init_variables(graph, graph_def, suffix=suffix)
            tf.constant("compressed_model", name="model_type", dtype=tf.string)
        else:
            raise RuntimeError("Unknown model type %s" % model_type)
        if self.typeebd is not None:
            self.typeebd.init_variables(graph, graph_def, suffix=suffix)
