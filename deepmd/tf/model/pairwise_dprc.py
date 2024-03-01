# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

from deepmd.tf.common import (
    add_data_requirement,
    make_default_mesh,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    MODEL_VERSION,
    op_module,
    tf,
)
from deepmd.tf.loss.loss import (
    Loss,
)
from deepmd.tf.model.model import (
    Model,
)
from deepmd.tf.utils.graph import (
    load_graph_def,
)
from deepmd.tf.utils.spin import (
    Spin,
)
from deepmd.tf.utils.type_embed import (
    TypeEmbedNet,
)
from deepmd.tf.utils.update_sel import (
    UpdateSel,
)


@Model.register("pairwise_dprc")
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
        # internal variable to compare old and new behavior
        # expect they give the same results
        self.merge_frames = True

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

        self.qm_model = Model(
            **qm_model,
            type_map=type_map,
            type_embedding=self.typeebd,
            compress=compress,
        )
        self.qmmm_model = Model(
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
            t_aparam_nall = tf.constant(True, name="aparam_nall", dtype=tf.bool)
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

        if self.merge_frames:
            (
                forward_qmmm_map,
                backward_qmmm_map,
                natoms_qmmm,
                mesh_qmmm,
            ) = op_module.convert_forward_map(forward_qmmm_map, natoms_qmmm, natoms)
            coord_qmmm = tf.reshape(coord, [1, -1, 3])
            atype_qmmm = tf.reshape(atype, [1, -1, 1])
            box_qmmm = tf.reshape(box[0], [1, 9])
        else:
            mesh_qmmm = make_default_mesh(False, True)
            coord_qmmm = tf.gather(coord, qmmm_frame_idx)
            atype_qmmm = tf.gather(atype, qmmm_frame_idx)
            box_qmmm = tf.gather(box, qmmm_frame_idx)

        coord_qm = gather_placeholder(coord, forward_qm_map)
        atype_qm = gather_placeholder(atype, forward_qm_map, placeholder=-1)
        coord_qmmm = gather_placeholder(coord_qmmm, forward_qmmm_map)
        atype_qmmm = gather_placeholder(atype_qmmm, forward_qmmm_map, placeholder=-1)
        box_qm = box

        type_embedding = self.build_type_embedding(
            self.ntypes,
            reuse=reuse,
            suffix=suffix,
            frz_model=frz_model,
            ckpt_meta=ckpt_meta,
        )
        input_dict_qm["type_embedding"] = type_embedding
        input_dict_qmmm["type_embedding"] = type_embedding

        mesh_mixed_type = make_default_mesh(False, True)

        # allow loading a frozen QM model that has only QM types
        # Note: here we don't map the type between models, so
        #       the type of the frozen model must be the same as
        #       the first Ntypes of the current model
        if self.get_ntypes() > self.qm_model.get_ntypes():
            natoms_qm = tf.slice(natoms_qm, [0], [self.qm_model.get_ntypes() + 2])
        assert self.get_ntypes() == self.qmmm_model.get_ntypes()

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
            mesh_qmmm,
            input_dict_qmmm,
            frz_model=frz_model,
            ckpt_meta=ckpt_meta,
            suffix="_qmmm" + suffix,
            reuse=reuse,
        )

        if self.merge_frames:
            qmmm_dict = qmmm_dict.copy()
            sub_nframes = tf.shape(backward_qmmm_map)[0]
            qmmm_dict["force"] = tf.tile(qmmm_dict["force"], [sub_nframes, 1])
            qmmm_dict["atom_ener"] = tf.tile(qmmm_dict["atom_ener"], [sub_nframes, 1])
            qmmm_dict["atom_virial"] = tf.tile(
                qmmm_dict["atom_virial"], [sub_nframes, 1]
            )

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

        backward_qm_map_nloc = tf.slice(backward_qm_map, [0, 0], [-1, natoms[0]])
        backward_qmmm_map_nloc = tf.slice(backward_qmmm_map, [0, 0], [-1, natoms[0]])
        atom_ener_qm = gather_placeholder(
            qm_dict["atom_ener"], backward_qm_map_nloc, placeholder=0.0
        )
        atom_ener_qmmm = tf.math.segment_sum(
            gather_placeholder(
                qmmm_dict["atom_ener"], backward_qmmm_map_nloc, placeholder=0.0
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

        energy = tf.reduce_sum(atom_ener, axis=1, name="o_energy" + suffix)
        virial = tf.reduce_sum(
            tf.reshape(atom_virial, (nframes, natoms[1], 9)),
            axis=1,
            name="o_virial" + suffix,
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
        return self.ntypes

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
            The box. Can be generated by deepmd.tf.model.make_stat_input
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

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        # do not update sel; only find min distance
        # rcut is not important here
        UpdateSel().get_min_nbor_dist(global_jdata, 6.0)
        return local_jdata


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
