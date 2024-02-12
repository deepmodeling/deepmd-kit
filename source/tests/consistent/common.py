# SPDX-License-Identifier: LGPL-3.0-or-later
import os
from abc import (
    ABC,
    abstractmethod,
)
from enum import (
    Enum,
)
from importlib.util import (
    find_spec,
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

from deepmd.common import (
    make_default_mesh,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)

if find_spec("tensorflow") is None:
    if os.environ.get("CI"):
        raise ImportError("TensorFlow should be tested in the CI")
    INSTALLED_TF = False
else:
    INSTALLED_TF = True
try:
    import torch
except ImportError as e:
    if os.environ.get("CI"):
        raise ImportError("PyTorch should be tested in the CI") from e
    INSTALLED_PT = False
else:
    INSTALLED_PT = True

if INSTALLED_PT:
    from deepmd.pt.utils.env import DEVICE as PT_DEVICE
    from deepmd.pt.utils.nlist import build_neighbor_list as build_neighbor_list_pt
    from deepmd.pt.utils.nlist import (
        extend_coord_with_ghosts as extend_coord_with_ghosts_pt,
    )
if INSTALLED_TF:
    from deepmd.tf.common import (
        clear_session,
    )
    from deepmd.tf.env import (
        GLOBAL_TF_FLOAT_PRECISION,
        default_tf_session_config,
        tf,
    )
    from deepmd.tf.utils.sess import (
        run_sess,
    )


__all__ = [
    "CommonTest",
    "DescriptorTest",
    "INSTALLED_TF",
    "INSTALLED_PT",
]


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
    skip_tf: ClassVar[bool] = not INSTALLED_TF
    """Whether to skip the TensorFlow model."""
    skip_pt: ClassVar[bool] = not INSTALLED_PT
    """Whether to skip the PyTorch model."""

    def setUp(self):
        self.unique_id = uuid4().hex

    def reset_unique_id(self):
        self.unique_id = uuid4().hex

    def init_backend_cls(self, cls) -> Any:
        """Initialize a backend model."""
        assert self.data is not None
        if self.args is None:
            data = self.data
        else:
            base = Argument("arg", dict, sub_fields=self.args)
            data = base.normalize_value(self.data, trim_pattern="_*")
            base.check_value(data, strict=True)
        return cls(**data)

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

    class RefBackend(Enum):
        """Reference backend."""

        TF = 1
        DP = 2
        PT = 3

    @abstractmethod
    def extract_ret(self, ret: Any, backend: RefBackend) -> Any:
        """Extract the return value when comparing with other backends.

        Parameters
        ----------
        ret : Any
            The return value
        backend : RefBackend
            The backend

        Returns
        -------
        Any
            The extracted return value
        """

    def build_eval_tf(
        self, sess: "tf.Session", obj: Any, suffix: str
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

    def get_tf_ret_serialization_from_cls(self, obj):
        with tf.Session(config=default_tf_session_config) as sess:
            graph = tf.get_default_graph()
            ret = self.build_eval_tf(sess, obj, suffix=self.unique_id)
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                graph.as_graph_def(),
                [f"o_{ii}_{self.unique_id}" for ii, _ in enumerate(ret)],
            )
            with tf.Graph().as_default() as new_graph:
                tf.import_graph_def(output_graph_def, name="")
            obj.init_variables(new_graph, output_graph_def, suffix=self.unique_id)

            data = obj.serialize(suffix=self.unique_id)
        return ret, data

    def get_pt_ret_serialization_from_cls(self, obj):
        ret = self.eval_pt(obj)
        data = obj.serialize()
        return ret, data

    def get_dp_ret_serialization_from_cls(self, obj):
        ret = self.eval_dp(obj)
        data = obj.serialize()
        return ret, data

    def get_reference_backend(self):
        if not self.skip_dp:
            return self.RefBackend.DP
        if not self.skip_tf:
            return self.RefBackend.TF
        if not self.skip_pt:
            return self.RefBackend.PT
        raise ValueError("No available reference")

    def get_reference_ret_serialization(self, ref: RefBackend):
        if ref == self.RefBackend.DP:
            obj = self.init_backend_cls(self.dp_class)
            return self.get_dp_ret_serialization_from_cls(obj)
        if ref == self.RefBackend.TF:
            obj = self.init_backend_cls(self.tf_class)
            self.reset_unique_id()
            return self.get_tf_ret_serialization_from_cls(obj)
        if ref == self.RefBackend.PT:
            obj = self.init_backend_cls(self.pt_class)
            return self.get_pt_ret_serialization_from_cls(obj)
        raise ValueError("No available reference")

    def test_tf_consistent_with_ref(self):
        """Test whether TF and reference are consistent."""
        if self.skip_tf:
            self.skipTest("Unsupported backend")
        ref_backend = self.get_reference_backend()
        if ref_backend == self.RefBackend.TF:
            self.skipTest("Reference is self")
        ret1, data1 = self.get_reference_ret_serialization(ref_backend)
        ret1 = self.extract_ret(ret1, ref_backend)
        self.reset_unique_id()
        tf_obj = self.tf_class.deserialize(data1, suffix=self.unique_id)
        ret2, data2 = self.get_tf_ret_serialization_from_cls(tf_obj)
        ret2 = self.extract_ret(ret2, self.RefBackend.TF)
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            np.testing.assert_allclose(rr1, rr2)

    def test_tf_self_consistent(self):
        """Test whether TF is self consistent."""
        if self.skip_tf:
            self.skipTest("Unsupported backend")
        obj1 = self.init_backend_cls(self.tf_class)
        self.reset_unique_id()
        ret1, data1 = self.get_tf_ret_serialization_from_cls(obj1)
        self.reset_unique_id()
        obj2 = self.tf_class.deserialize(data1, suffix=self.unique_id)
        ret2, data2 = self.get_tf_ret_serialization_from_cls(obj2)
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            np.testing.assert_allclose(rr1, rr2)

    def test_dp_consistent_with_ref(self):
        """Test whether DP and reference are consistent."""
        if self.skip_dp:
            self.skipTest("Unsupported backend")
        ref_backend = self.get_reference_backend()
        if ref_backend == self.RefBackend.DP:
            self.skipTest("Reference is self")
        ret1, data1 = self.get_reference_ret_serialization(ref_backend)
        ret1 = self.extract_ret(ret1, ref_backend)
        dp_obj = self.dp_class.deserialize(data1)
        ret2 = self.eval_dp(dp_obj)
        ret2 = self.extract_ret(ret2, self.RefBackend.DP)
        data2 = dp_obj.serialize()
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            np.testing.assert_allclose(rr1, rr2)

    def test_dp_self_consistent(self):
        """Test whether DP is self consistent."""
        if self.skip_dp:
            self.skipTest("Unsupported backend")
        obj1 = self.init_backend_cls(self.dp_class)
        ret1, data1 = self.get_dp_ret_serialization_from_cls(obj1)
        obj1 = self.dp_class.deserialize(data1)
        ret2, data2 = self.get_dp_ret_serialization_from_cls(obj1)
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            if isinstance(rr1, np.ndarray) and isinstance(rr2, np.ndarray):
                np.testing.assert_allclose(rr1, rr2)
            else:
                self.assertEqual(rr1, rr2)

    def test_pt_consistent_with_ref(self):
        """Test whether PT and reference are consistent."""
        if self.skip_pt:
            self.skipTest("Unsupported backend")
        ref_backend = self.get_reference_backend()
        if ref_backend == self.RefBackend.PT:
            self.skipTest("Reference is self")
        ret1, data1 = self.get_reference_ret_serialization(ref_backend)
        ret1 = self.extract_ret(ret1, ref_backend)
        obj = self.pt_class.deserialize(data1)
        ret2 = self.eval_pt(obj)
        ret2 = self.extract_ret(ret2, self.RefBackend.PT)
        data2 = obj.serialize()
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            np.testing.assert_allclose(rr1, rr2)

    def test_pt_self_consistent(self):
        """Test whether PT is self consistent."""
        if self.skip_pt:
            self.skipTest("Unsupported backend")
        obj1 = self.init_backend_cls(self.pt_class)
        ret1, data1 = self.get_pt_ret_serialization_from_cls(obj1)
        obj2 = self.pt_class.deserialize(data1)
        ret2, data2 = self.get_pt_ret_serialization_from_cls(obj2)
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            if isinstance(rr1, np.ndarray) and isinstance(rr2, np.ndarray):
                np.testing.assert_allclose(rr1, rr2)
            else:
                self.assertEqual(rr1, rr2)

    def tearDown(self) -> None:
        """Clear the TF session."""
        if not self.skip_tf:
            clear_session()


class DescriptorTest:
    """Useful utilities for descriptor tests."""

    def build_tf_descriptor(self, obj, natoms, coords, atype, box, suffix):
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, natoms.shape, name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        t_des = obj.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {},
            suffix=suffix,
        )
        return [t_des], {
            t_coord: coords,
            t_type: atype,
            t_natoms: natoms,
            t_box: box,
            t_mesh: make_default_mesh(True, False),
        }

    def eval_dp_descriptor(self, dp_obj: Any, natoms, coords, atype, box) -> Any:
        ext_coords, ext_atype, mapping = extend_coord_with_ghosts(
            coords.reshape(1, -1, 3),
            atype.reshape(1, -1),
            box.reshape(1, 3, 3),
            dp_obj.get_rcut(),
        )
        nlist = build_neighbor_list(
            ext_coords,
            ext_atype,
            natoms[0],
            dp_obj.get_rcut(),
            dp_obj.get_sel(),
            distinguish_types=True,
        )
        return dp_obj(ext_coords, ext_atype, nlist=nlist)

    def eval_pt_descriptor(self, pt_obj: Any, natoms, coords, atype, box) -> Any:
        ext_coords, ext_atype, mapping = extend_coord_with_ghosts_pt(
            torch.from_numpy(coords).to(PT_DEVICE).reshape(1, -1, 3),
            torch.from_numpy(atype).to(PT_DEVICE).reshape(1, -1),
            torch.from_numpy(box).to(PT_DEVICE).reshape(1, 3, 3),
            pt_obj.get_rcut(),
        )
        nlist = build_neighbor_list_pt(
            ext_coords,
            ext_atype,
            natoms[0],
            pt_obj.get_rcut(),
            pt_obj.get_sel(),
            distinguish_types=True,
        )
        return [
            x.detach().cpu().numpy() if torch.is_tensor(x) else x
            for x in pt_obj(ext_coords, ext_atype, nlist=nlist)
        ]
