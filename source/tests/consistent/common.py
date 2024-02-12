# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    ClassVar,
    List,
    Optional,
    Tuple,
)
from uuid import (
    uuid4,
)

import numpy as np
from dargs import (
    Argument,
)

from deepmd.tf.common import (
    clear_session,
)
from deepmd.tf.env import (
    default_tf_session_config,
    tf,
)
from deepmd.tf.utils.sess import (
    run_sess,
)


class CommonTest(ABC):
    data: ClassVar[dict]
    """Arguments data."""
    tf_class: ClassVar[type]
    """TensorFlow model class."""
    dp_class: ClassVar[type]
    """Native DP model class."""
    pt_class: ClassVar[type]
    """PyTorch model class."""
    args: ClassVar[Optional[List[Argument]]]
    """Arguments that maps to the `data`."""
    skip_dp: ClassVar[bool] = False
    """Whether to skip the native DP model."""
    skip_pt: ClassVar[bool] = False
    """Whether to skip the PyTorch model."""

    def setUp(self):
        self.unique_id1 = uuid4().hex
        self.unique_id2 = uuid4().hex

    def init_tf(self) -> Any:
        """Initialize the TF model."""
        assert self.data is not None
        if self.args is None:
            data = self.data
        else:
            base = Argument("arg", dict, sub_fields=self.args)
            data = base.normalize_value(self.data, trim_pattern="_*")
            base.check_value(data, strict=True)
        return self.tf_class(**data)

    @abstractmethod
    def build_tf(self, obj: Any, suffix: str) -> Tuple[list, dict]:
        """Build the TF graph.

        Parameters
        ----------
        obj : Any
            The object of TF
        suffix : str
            The suffix of the scope

        Returns
        -------
        list of tf.Tensor
            The list of tensors
        dict
            The feed_dict
        """

    @abstractmethod
    def eval_dp(self, dp_obj: Any) -> Any:
        """Evaluate the return value of DP.

        Parameters
        ----------
        dp_obj : Any
            The object of DP
        """

    @abstractmethod
    def eval_pt(self, pt_obj: Any) -> Any:
        """Evaluate the return value of PT.

        Parameters
        ----------
        pt_obj : Any
            The object of PT
        """

    @abstractmethod
    def compare_tf_dp_ret(self, tf_ret: Any, dp_ret: Any) -> None:
        """Compare the return value of TF and DP.

        Parameters
        ----------
        tf_ret : Any
            The return value of TF
        dp_ret : Any
            The return value of DP
        """

    @abstractmethod
    def compare_tf_pt_ret(self, tf_ret: Any, pt_ret: Any) -> None:
        """Compare the return value of TF and PT.

        Parameters
        ----------
        tf_ret : Any
            The return value of TF
        pt_ret : Any
            The return value of PT
        """

    def build_eval_tf(
        self, sess: tf.Session, obj: Any, suffix: str
    ) -> List[np.ndarray]:
        """Build and evaluate the TF graph."""
        t_out, feed_dict = self.build_tf(obj, suffix)

        t_out_indentity = [
            tf.identity(tt, name=f"o_{ii}_{suffix}") for ii, tt in enumerate(t_out)
        ]
        run_sess(sess, tf.global_variables_initializer())
        return run_sess(
            sess,
            t_out_indentity,
            feed_dict=feed_dict,
        )

    def test_tf_dp_consistent(self):
        """Test whether TF and DP are consistent."""
        if self.skip_dp:
            self.skipTest("Unsupported")
        tf_obj1 = self.init_tf()
        with tf.Session(config=default_tf_session_config) as sess:
            graph = tf.get_default_graph()
            ret1 = self.build_eval_tf(sess, tf_obj1, suffix=self.unique_id1)
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                graph.as_graph_def(),
                [f"o_{ii}_{self.unique_id1}" for ii, _ in enumerate(ret1)],
            )
            with tf.Graph().as_default() as new_graph:
                tf.import_graph_def(output_graph_def, name="")
            tf_obj1.init_variables(new_graph, output_graph_def, suffix=self.unique_id1)

            data = tf_obj1.serialize(suffix=self.unique_id1)
        dp_obj = self.dp_class.deserialize(data)
        ret2 = self.eval_dp(dp_obj)
        self.compare_tf_dp_ret(ret1, ret2)

    def test_tf_self_consistent(self):
        """Test whether TF is self consistent."""
        tf_obj1 = self.init_tf()
        with tf.Session(config=default_tf_session_config) as sess:
            graph = tf.get_default_graph()
            ret1 = self.build_eval_tf(sess, tf_obj1, suffix=self.unique_id1)
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                graph.as_graph_def(),
                [f"o_{ii}_{self.unique_id1}" for ii, _ in enumerate(ret1)],
            )
            with tf.Graph().as_default() as new_graph:
                tf.import_graph_def(output_graph_def, name="")
            tf_obj1.init_variables(new_graph, output_graph_def, suffix=self.unique_id1)

            data = tf_obj1.serialize(suffix=self.unique_id1)
            tf_obj2 = self.tf_class.deserialize(data, suffix=self.unique_id2)
            ret2 = self.build_eval_tf(sess, tf_obj2, suffix=self.unique_id2)
        for rr1, rr2 in zip(ret1, ret2):
            np.testing.assert_allclose(rr1, rr2)

    def test_dp_self_consistent(self):
        """Test whether DP is self consistent."""
        if self.skip_dp:
            self.skipTest("Unsupported")
        tf_obj1 = self.init_tf()
        with tf.Session(config=default_tf_session_config) as sess:
            graph = tf.get_default_graph()
            ret1 = self.build_eval_tf(sess, tf_obj1, suffix=self.unique_id1)
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                graph.as_graph_def(),
                [f"o_{ii}_{self.unique_id1}" for ii, _ in enumerate(ret1)],
            )
            with tf.Graph().as_default() as new_graph:
                tf.import_graph_def(output_graph_def, name="")
            tf_obj1.init_variables(new_graph, output_graph_def, suffix=self.unique_id1)

            data = tf_obj1.serialize(suffix=self.unique_id1)
        dp_obj = self.dp_class.deserialize(data)
        ret2 = self.eval_dp(dp_obj)
        data2 = dp_obj.serialize()
        dp_obj2 = self.dp_class.deserialize(data2)
        ret3 = self.eval_dp(dp_obj2)
        for rr2, rr3 in zip(ret2, ret3):
            if isinstance(rr2, np.ndarray) and isinstance(rr3, np.ndarray):
                np.testing.assert_allclose(rr2, rr3)
            else:
                self.assertEqual(rr2, rr3)

    def test_tf_pt_consistent(self):
        """Test whether TF and PT are consistent."""
        if self.skip_pt:
            self.skipTest("Unsupported")
        tf_obj1 = self.init_tf()
        with tf.Session(config=default_tf_session_config) as sess:
            graph = tf.get_default_graph()
            ret1 = self.build_eval_tf(sess, tf_obj1, suffix=self.unique_id1)
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                graph.as_graph_def(),
                [f"o_{ii}_{self.unique_id1}" for ii, _ in enumerate(ret1)],
            )
            with tf.Graph().as_default() as new_graph:
                tf.import_graph_def(output_graph_def, name="")
            tf_obj1.init_variables(new_graph, output_graph_def, suffix=self.unique_id1)

            data = tf_obj1.serialize(suffix=self.unique_id1)
        pt_obj = self.pt_class.deserialize(data)
        ret2 = self.eval_pt(pt_obj)
        self.compare_tf_pt_ret(ret1, ret2)

    def test_pt_self_consistent(self):
        """Test whether PT is self consistent."""
        if self.skip_pt:
            self.skipTest("Unsupported")
        tf_obj1 = self.init_tf()
        with tf.Session(config=default_tf_session_config) as sess:
            graph = tf.get_default_graph()
            ret1 = self.build_eval_tf(sess, tf_obj1, suffix=self.unique_id1)
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                graph.as_graph_def(),
                [f"o_{ii}_{self.unique_id1}" for ii, _ in enumerate(ret1)],
            )
            with tf.Graph().as_default() as new_graph:
                tf.import_graph_def(output_graph_def, name="")
            tf_obj1.init_variables(new_graph, output_graph_def, suffix=self.unique_id1)

            data = tf_obj1.serialize(suffix=self.unique_id1)
        pt_obj = self.pt_class.deserialize(data)
        ret2 = self.eval_pt(pt_obj)
        data2 = pt_obj.serialize()
        pt_obj2 = self.pt_class.deserialize(data2)
        ret3 = self.eval_pt(pt_obj2)
        for rr2, rr3 in zip(ret2, ret3):
            if isinstance(rr2, np.ndarray) and isinstance(rr3, np.ndarray):
                np.testing.assert_allclose(rr2, rr3)
            else:
                self.assertEqual(rr2, rr3)

    def tearDown(self) -> None:
        """Clear the TF session."""
        clear_session()
