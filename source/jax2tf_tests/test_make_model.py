# SPDX-License-Identifier: LGPL-3.0-or-later
import tensorflow as tf

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.jax.jax2tf.make_model import (
    model_call_from_call_lower,
)

DTYPE = tf.float64


class TestMakeModel(tf.test.TestCase):
    def setUp(self) -> None:
        self.output_def = ModelOutputDef(
            FittingOutputDef([OutputVariableDef("coord_x", [1])])
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None, 3], DTYPE),
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([None, None, None], DTYPE),
        ]
    )
    def call_model(
        self,
        coord: tf.Tensor,
        atype: tf.Tensor,
        box: tf.Tensor,
    ) -> tf.Tensor:
        def call_lower(
            extended_coord: tf.Tensor,
            extended_atype: tf.Tensor,
            nlist: tf.Tensor,
            mapping: tf.Tensor,
            fparam: tf.Tensor,
            aparam: tf.Tensor,
        ) -> dict[str, tf.Tensor]:
            del extended_atype, nlist, mapping, fparam, aparam
            return {"coord_x": extended_coord[..., :1]}

        nframes = tf.shape(coord)[0]
        nloc = tf.shape(atype)[1]
        ret = model_call_from_call_lower(
            call_lower=call_lower,
            rcut=0.4,
            sel=[1],
            mixed_types=True,
            model_output_def=self.output_def,
            coord=coord,
            atype=atype,
            box=box,
            fparam=tf.zeros([nframes, 0], dtype=DTYPE),
            aparam=tf.zeros([nframes, nloc, 0], dtype=DTYPE),
        )
        return ret["coord_x"]

    def test_model_call_without_box(self) -> None:
        coord = tf.constant([[[0.2, 0.0, 0.0], [0.8, 0.0, 0.0]]], dtype=DTYPE)
        atype = tf.constant([[0, 1]], dtype=tf.int32)
        box = tf.zeros([1, 0, 0], dtype=DTYPE)

        coord_x = self.call_model(coord, atype, box)

        self.assertAllClose(coord_x, coord[..., :1])

    def test_model_call_with_box_normalizes_coord(self) -> None:
        coord = tf.constant([[[1.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]], dtype=DTYPE)
        atype = tf.constant([[0, 1]], dtype=tf.int32)
        box = tf.eye(3, batch_shape=[1], dtype=DTYPE)

        coord_x = self.call_model(coord, atype, box)

        self.assertAllClose(coord_x[:, :2], [[[0.2], [0.8]]])
