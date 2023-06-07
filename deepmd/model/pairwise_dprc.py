from typing import (
    List,
    Optional,
    Union,
)

from deepmd.common import (
    add_data_requirement,
)
from deepmd.env import (
    op_module,
    tf,
)
from deepmd.loss.loss import (
    Loss,
)
from deepmd.model.model import (
    Model,
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

    model_type = "pairwise_dprc"

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
            **qm_model, type_map=type_map, type_embedding=self.typeebd
        )
        self.qmmm_model = EnerModel(
            **qmmm_model, type_map=type_map, type_embedding=self.typeebd
        )
        add_data_requirement("aparam", 1, atomic=True, must=True, high_prec=False)

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
        with tf.variable_scope("fitting_attr" + suffix, reuse=reuse):
            t_dfparam = tf.constant(0, name="dfparam", dtype=tf.int32)
            t_daparam = tf.constant(1, name="daparam", dtype=tf.int32)
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

        # TODO: after #2481 is merged, change the mesh to mixed_type specific

        qm_dict = self.qm_model.build(
            coord_qm,
            atype_qm,
            natoms_qm,
            box_qm,
            mesh,
            input_dict,
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
            mesh,
            input_dict,
            frz_model=frz_model,
            ckpt_meta=ckpt_meta,
            suffix="_qmmm" + suffix,
            reuse=reuse,
        )

        energy_qm = qm_dict["energy"]
        energy_qmmm = tf.math.segment_sum(qmmm_dict["energy"], qmmm_frame_idx)
        energy = energy_qm + energy_qmmm

        force_qm = gather_placeholder(
            qm_dict["force"], backward_qm_map, placeholder=0.0
        )
        force_qmmm = tf.math.segment_sum(
            gather_placeholder(qmmm_dict["force"], backward_qmmm_map, placeholder=0.0),
            qmmm_frame_idx,
        )
        force = force_qm + force_qmmm

        virial_qm = qm_dict["virial"]
        virial_qmmm = tf.math.segment_sum(qmmm_dict["virial"], qmmm_frame_idx)
        virial = virial_qm + virial_qmmm

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

        atom_virial_qm = gather_placeholder(
            qm_dict["atom_virial"], backward_qm_map, placeholder=0.0
        )
        atom_virial_qmmm = tf.math.segment_sum(
            gather_placeholder(
                qmmm_dict["atom_virial"], backward_qmmm_map, placeholder=0.0
            ),
            qmmm_frame_idx,
        )
        atom_virial = atom_virial_qm + atom_virial_qmmm

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
