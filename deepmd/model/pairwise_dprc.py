# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

from deepmd.common import (
    add_data_requirement,
    make_default_mesh,
)
from deepmd.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    MODEL_VERSION,
    op_module,
    tf,
)
from deepmd.loss.loss import (
    Loss,
)
from deepmd.model.model import (
    Model,
)
from deepmd.utils.graph import (
    load_graph_def,
)
from deepmd.utils.spin import (
    Spin,
)
from deepmd.utils.type_embed import (
    TypeEmbedNet,
)

from .ener import (
    EnerModel,
)


class PairwiseDPRc(Model):
    """Pairwise Deep Potential - Range Correction."""

    model_type = "ener"

    def __init__(
        self,
        qm_model: dict,
        qmmm_model: dict,
        type_embedding: Union[dict, TypeEmbedNet],
        type_map: List[str],
        data_stat_nbatch: int = 10,
        data_stat_nsample: int = 10,
        data_stat_protect: float = 1e-2,
        use_srtab: Optional[str] = None,
        smin_alpha: Optional[float] = None,
        sw_rmin: Optional[float] = None,
        sw_rmax: Optional[float] = None,
        spin: Optional[Spin] = None,
        compress: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            type_embedding=type_embedding,
            type_map=type_map,
            data_stat_nbatch=data_stat_nbatch,
            data_stat_nsample=data_stat_nsample,
            data_stat_protect=data_stat_protect,
            use_srtab=use_srtab,
            smin_alpha=smin_alpha,
            sw_rmin=sw_rmin,
            sw_rmax=sw_rmax,
            spin=spin,
            compress=compress,
            **kwargs,
        )
        # type embedding
        if isinstance(type_embedding, TypeEmbedNet):
            self.typeebd = type_embedding
        else:
            self.typeebd = TypeEmbedNet(
                **type_embedding,
                # must use se_atten, so it must be True
                padding=True,
            )

        self.qm_model = EnerModel(
            **qm_model,
            type_map=type_map,
            type_embedding=self.typeebd,
            compress=compress,
        )
        self.qmmm_model = EnerModel(
            **qmmm_model,
            type_map=type_map,
            type_embedding=self.typeebd,
            compress=compress,
        )
        add_data_requirement("aparam", 1, atomic=True, must=True, high_prec=False)
        self.ntypes = len(type_map)
        self.rcut = max(self.qm_model.get_rcut(), self.qmmm_model.get_rcut())

    def build(
        self,
        coord_: tf.Tensor,
        atype_: tf.Tensor,
        natoms: tf.Tensor,
        box_: tf.Tensor,
        mesh: tf.Tensor,
        input_dict: dict,
        frz_model=None,
        ckpt_meta: Optional[str] = None,
        suffix: str = "",
        reuse: Optional[bool] = None,
    ):
        feed_dict = self.get_feed_dict(
            coord_, atype_, natoms, box_, mesh, aparam=input_dict["aparam"]
        )
        input_dict_qm = {"global_feed_dict": feed_dict}
        input_dict_qmmm = {"global_feed_dict": feed_dict}
        with tf.variable_scope("model_attr" + suffix, reuse=reuse):
            t_tmap = tf.constant(" ".join(self.type_map), name="tmap", dtype=tf.string)
            t_mt = tf.constant(self.model_type, name="model_type", dtype=tf.string)
            t_ver = tf.constant(MODEL_VERSION, name="model_version", dtype=tf.string)

        with tf.variable_scope("fitting_attr" + suffix, reuse=reuse):
            t_dfparam = tf.constant(0, name="dfparam", dtype=tf.int32)
            t_daparam = tf.constant(1, name="daparam", dtype=tf.int32)
        with tf.variable_scope("descrpt_attr" + suffix, reuse=reuse):
            t_ntypes = tf.constant(self.ntypes, name="ntypes", dtype=tf.int32)
            t_rcut = tf.constant(
                self.rcut, name="rcut", dtype=GLOBAL_TF_FLOAT_PRECISION
            )
        # convert X-frame to X-Y-frame coordinates
        box = tf.reshape(box_, [-1, 9])
        nframes = tf.shape(box)[0]
        idxs = tf.cast(input_dict["aparam"], tf.int32)
        idxs = tf.reshape(idxs, (nframes, natoms[1]))

        (
            forward_qm_map,
            backward_qm_map,
            forward_qmmm_map,
            backward_qmmm_map,
            natoms_qm,
            natoms_qmmm,
            qmmm_frame_idx,
        ) = op_module.dprc_pairwise_idx(idxs, natoms)

        coord = tf.reshape(coord_, [nframes, natoms[1], 3])
        atype = tf.reshape(atype_, [nframes, natoms[1], 1])
        nframes_qmmm = tf.shape(qmmm_frame_idx)[0]

        coord_qm = gather_placeholder(coord, forward_qm_map)
        atype_qm = gather_placeholder(atype, forward_qm_map, placeholder=-1)
        coord_qmmm = gather_placeholder(
            tf.gather(coord, qmmm_frame_idx), forward_qmmm_map
        )
        atype_qmmm = gather_placeholder(
            tf.gather(atype, qmmm_frame_idx), forward_qmmm_map, placeholder=-1
        )
        box_qm = box
        box_qmmm = tf.gather(box, qmmm_frame_idx)

        type_embedding = self.typeebd.build(
            self.ntypes,
            reuse=reuse,
            suffix=suffix,
        )
        input_dict_qm["type_embedding"] = type_embedding
        input_dict_qmmm["type_embedding"] = type_embedding

        mesh_mixed_type = make_default_mesh(False, True)

        qm_dict = self.qm_model.build(
            coord_qm,
            atype_qm,
            natoms_qm,
            box_qm,
            mesh_mixed_type,
            input_dict_qm,
            frz_model=frz_model,
            ckpt_meta=ckpt_meta,
            suffix="_qm" + suffix,
            reuse=reuse,
        )
        qmmm_dict = self.qmmm_model.build(
            coord_qmmm,
            atype_qmmm,
            natoms_qmmm,
            box_qmmm,
            mesh_mixed_type,
            input_dict_qmmm,
            frz_model=frz_model,
            ckpt_meta=ckpt_meta,
            suffix="_qmmm" + suffix,
            reuse=reuse,
        )

        energy_qm = qm_dict["energy"]
        energy_qmmm = tf.math.segment_sum(qmmm_dict["energy"], qmmm_frame_idx)
        energy = energy_qm + energy_qmmm
        energy = tf.identity(energy, name="o_energy" + suffix)

        force_qm = gather_placeholder(
            tf.reshape(qm_dict["force"], (nframes, natoms_qm[1], 3)),
            backward_qm_map,
            placeholder=0.0,
        )
        force_qmmm = tf.math.segment_sum(
            gather_placeholder(
                tf.reshape(qmmm_dict["force"], (nframes_qmmm, natoms_qmmm[1], 3)),
                backward_qmmm_map,
                placeholder=0.0,
            ),
            qmmm_frame_idx,
        )
        force = force_qm + force_qmmm
        force = tf.reshape(force, (nframes, 3 * natoms[1]), name="o_force" + suffix)

        virial_qm = qm_dict["virial"]
        virial_qmmm = tf.math.segment_sum(qmmm_dict["virial"], qmmm_frame_idx)
        virial = virial_qm + virial_qmmm
        virial = tf.identity(virial, name="o_virial" + suffix)

        atom_ener_qm = gather_placeholder(
            qm_dict["atom_ener"], backward_qm_map, placeholder=0.0
        )
        atom_ener_qmmm = tf.math.segment_sum(
            gather_placeholder(
                qmmm_dict["atom_ener"], backward_qmmm_map, placeholder=0.0
            ),
            qmmm_frame_idx,
        )
        atom_ener = atom_ener_qm + atom_ener_qmmm
        atom_ener = tf.identity(atom_ener, name="o_atom_energy" + suffix)

        atom_virial_qm = gather_placeholder(
            tf.reshape(qm_dict["atom_virial"], (nframes, natoms_qm[1], 9)),
            backward_qm_map,
            placeholder=0.0,
        )
        atom_virial_qmmm = tf.math.segment_sum(
            gather_placeholder(
                tf.reshape(qmmm_dict["atom_virial"], (nframes_qmmm, natoms_qmmm[1], 9)),
                backward_qmmm_map,
                placeholder=0.0,
            ),
            qmmm_frame_idx,
        )
        atom_virial = atom_virial_qm + atom_virial_qmmm
        atom_virial = tf.reshape(
            atom_virial, (nframes, 9 * natoms[1]), name="o_atom_virial" + suffix
        )

        model_dict = {}
        model_dict["energy"] = energy
        model_dict["force"] = force
        model_dict["virial"] = virial
        model_dict["atom_ener"] = atom_ener
        model_dict["atom_virial"] = atom_virial
        model_dict["coord"] = coord_
        model_dict["atype"] = atype_
        return model_dict

    def get_fitting(self) -> Union[str, dict]:
        """Get the fitting(s)."""
        return {
            "qm": self.qm_model.get_fitting(),
            "qmmm": self.qmmm_model.get_fitting(),
        }

    def get_loss(self, loss: dict, lr) -> Union[Loss, dict]:
        """Get the loss function(s)."""
        return self.qm_model.get_loss(loss, lr)

    def get_rcut(self):
        return max(self.qm_model.get_rcut(), self.qmmm_model.get_rcut())

    def get_ntypes(self) -> int:
        return self.qm_model.get_ntypes()

    def data_stat(self, data):
        self.qm_model.data_stat(data)
        self.qmmm_model.data_stat(data)

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
        self.typeebd.init_variables(graph, graph_def, model_type=model_type)
        self.qm_model.init_variables(
            graph, graph_def, model_type=model_type, suffix="_qm" + suffix
        )
        self.qmmm_model.init_variables(
            graph, graph_def, model_type=model_type, suffix="_qmmm" + suffix
        )

    def enable_compression(self, suffix: str = "") -> None:
        """Enable compression.

        Parameters
        ----------
        suffix : str
            suffix to name scope
        """
        graph, graph_def = load_graph_def(self.compress["model_file"])
        self.typeebd.init_variables(graph, graph_def)
        self.qm_model.enable_compression(suffix="_qm" + suffix)
        self.qmmm_model.enable_compression(suffix="_qmmm" + suffix)

    def get_feed_dict(
        self,
        coord_: tf.Tensor,
        atype_: tf.Tensor,
        natoms: tf.Tensor,
        box: tf.Tensor,
        mesh: tf.Tensor,
        **kwargs,
    ) -> Dict[str, tf.Tensor]:
        """Generate the feed_dict for current descriptor.

        Parameters
        ----------
        coord_ : tf.Tensor
            The coordinate of atoms
        atype_ : tf.Tensor
            The type of atoms
        natoms : tf.Tensor
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        box : tf.Tensor
            The box. Can be generated by deepmd.model.make_stat_input
        mesh : tf.Tensor
            For historical reasons, only the length of the Tensor matters.
            if size of mesh == 6, pbc is assumed.
            if size of mesh == 0, no-pbc is assumed.
        aparam : tf.Tensor
            The parameters of the descriptor
        **kwargs : dict
            The keyword arguments

        Returns
        -------
        feed_dict : dict[str, tf.Tensor]
            The output feed_dict of current descriptor
        """
        feed_dict = {
            "t_coord:0": coord_,
            "t_type:0": atype_,
            "t_natoms:0": natoms,
            "t_box:0": box,
            "t_aparam:0": kwargs["aparam"],
        }
        return feed_dict


def gather_placeholder(
    params: tf.Tensor, indices: tf.Tensor, placeholder: float = 0.0, **kwargs
) -> tf.Tensor:
    """Call tf.gather but allow indices to contain placeholders (-1)."""
    # (nframes, x, 2, 3) -> (nframes, 1, 2, 3)
    placeholder_shape = tf.concat(
        [[tf.shape(params)[0], 1], tf.shape(params)[2:]], axis=0
    )
    params = tf.concat(
        [tf.cast(tf.fill(placeholder_shape, placeholder), params.dtype), params], axis=1
    )
    return tf.gather(params, indices + 1, batch_dims=1, **kwargs)
