# SPDX-License-Identifier: LGPL-3.0-or-later
import inspect
import itertools
import os
import sys
import unittest
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
    Callable,
    ClassVar,
    Optional,
    Union,
)
from uuid import (
    uuid4,
)

import numpy as np
from dargs import (
    Argument,
)

from deepmd.backend.tensorflow import (
    Backend,
)

from ..utils import (
    CI,
    TEST_DEVICE,
)

INSTALLED_TF = Backend.get_backend("tensorflow")().is_available()
INSTALLED_PT = Backend.get_backend("pytorch")().is_available()
INSTALLED_JAX = Backend.get_backend("jax")().is_available()
INSTALLED_PD = Backend.get_backend("paddle")().is_available()
INSTALLED_ARRAY_API_STRICT = find_spec("array_api_strict") is not None

if os.environ.get("CI") and not (INSTALLED_TF and INSTALLED_PT and INSTALLED_PD):
    raise ImportError("TensorFlow, PyTorch or Paddle should be tested in the CI")


if INSTALLED_TF:
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


__all__ = [
    "INSTALLED_ARRAY_API_STRICT",
    "INSTALLED_JAX",
    "INSTALLED_PD",
    "INSTALLED_PT",
    "INSTALLED_TF",
    "CommonTest",
    "CommonTest",
]

SKIP_FLAG = object()


class CommonTest(ABC):
    data: ClassVar[dict]
    """Arguments data."""
    additional_data: ClassVar[dict] = {}
    """Additional data that will not be checked."""
    tf_class: ClassVar[Optional[type]]
    """TensorFlow model class."""
    dp_class: ClassVar[Optional[type]]
    """Native DP model class."""
    pt_class: ClassVar[Optional[type]]
    """PyTorch model class."""
    jax_class: ClassVar[Optional[type]]
    """JAX model class."""
    pd_class: ClassVar[Optional[type]]
    """Paddle model class."""
    array_api_strict_class: ClassVar[Optional[type]]
    args: ClassVar[Optional[Union[Argument, list[Argument]]]]
    """Arguments that maps to the `data`."""
    skip_dp: ClassVar[bool] = False
    """Whether to skip the native DP model."""
    skip_tf: ClassVar[bool] = not INSTALLED_TF
    """Whether to skip the TensorFlow model."""
    skip_pt: ClassVar[bool] = not INSTALLED_PT
    """Whether to skip the PyTorch model."""
    # we may usually skip jax before jax is fully supported
    skip_jax: ClassVar[bool] = True
    """Whether to skip the JAX model."""
    skip_pd: ClassVar[bool] = True
    """Whether to skip the Paddle model."""
    skip_array_api_strict: ClassVar[bool] = True
    """Whether to skip the array_api_strict model."""
    rtol = 1e-10
    """Relative tolerance for comparing the return value. Override for float32."""
    atol = 1e-10
    """Absolute tolerance for comparing the return value. Override for float32."""

    def setUp(self) -> None:
        self.unique_id = uuid4().hex

    def reset_unique_id(self) -> None:
        self.unique_id = uuid4().hex

    def init_backend_cls(self, cls) -> Any:
        """Initialize a backend model."""
        assert self.data is not None
        if self.args is None:
            data = self.data
        else:
            if isinstance(self.args, list):
                base = Argument("arg", dict, sub_fields=self.args)
            elif isinstance(self.args, Argument):
                base = self.args
            else:
                raise ValueError("Invalid type for args")
            data = base.normalize_value(self.data, trim_pattern="_*")
            base.check_value(data, strict=True)
        return self.pass_data_to_cls(cls, data)

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        return cls(**data, **self.additional_data)

    @abstractmethod
    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
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

    def eval_jax(self, jax_obj: Any) -> Any:
        """Evaluate the return value of JAX.

        Parameters
        ----------
        jax_obj : Any
            The object of JAX
        """
        raise NotImplementedError("Not implemented")

    def eval_pd(self, pd_obj: Any) -> Any:
        """Evaluate the return value of PD.

        Parameters
        ----------
        pd_obj : Any
            The object of PD
        """
        raise NotImplementedError("Not implemented")

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        """Evaluate the return value of array_api_strict.

        Parameters
        ----------
        array_api_strict_obj : Any
            The object of array_api_strict
        """
        raise NotImplementedError("Not implemented")

    class RefBackend(Enum):
        """Reference backend."""

        TF = 1
        DP = 2
        PT = 3
        PD = 4
        JAX = 5
        ARRAY_API_STRICT = 6

    @abstractmethod
    def extract_ret(self, ret: Any, backend: RefBackend) -> tuple[np.ndarray, ...]:
        """Extract the return value when comparing with other backends.

        Parameters
        ----------
        ret : Any
            The return value
        backend : RefBackend
            The backend

        Returns
        -------
        tuple[np.ndarray, ...]
            The extracted return value
        """

    def build_eval_tf(
        self, sess: "tf.Session", obj: Any, suffix: str
    ) -> list[np.ndarray]:
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

    def get_jax_ret_serialization_from_cls(self, obj):
        ret = self.eval_jax(obj)
        data = obj.serialize()
        return ret, data

    def get_pd_ret_serialization_from_cls(self, obj):
        ret = self.eval_pd(obj)
        data = obj.serialize()
        return ret, data

    def get_array_api_strict_ret_serialization_from_cls(self, obj):
        ret = self.eval_array_api_strict(obj)
        data = obj.serialize()
        return ret, data

    def get_reference_backend(self):
        """Get the reference backend.

        Order of checking for ref: DP, TF, PT.
        """
        if not self.skip_dp:
            return self.RefBackend.DP
        if not self.skip_tf:
            return self.RefBackend.TF
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_jax:
            return self.RefBackend.JAX
        if not self.skip_pd:
            return self.RefBackend.PD
        if not self.skip_array_api_strict:
            return self.RefBackend.ARRAY_API_STRICT
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
        if ref == self.RefBackend.JAX:
            obj = self.init_backend_cls(self.jax_class)
            return self.get_jax_ret_serialization_from_cls(obj)
        if ref == self.RefBackend.PD:
            obj = self.init_backend_cls(self.pd_class)
            return self.get_pd_ret_serialization_from_cls(obj)
        if ref == self.RefBackend.ARRAY_API_STRICT:
            obj = self.init_backend_cls(self.array_api_strict_class)
            return self.get_array_api_strict_ret_serialization_from_cls(obj)
        raise ValueError("No available reference")

    def test_tf_consistent_with_ref(self) -> None:
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
        if tf_obj.__class__.__name__.startswith(("Polar", "Dipole", "DOS")):
            # tf, pt serialization mismatch
            common_keys = set(data1.keys()) & set(data2.keys())
            data1 = {k: data1[k] for k in common_keys}
            data2 = {k: data2[k] for k in common_keys}

        # not comparing version
        data1.pop("@version")
        data2.pop("@version")

        if tf_obj.__class__.__name__.startswith("Polar"):
            data1["@variables"].pop("bias_atom_e")

        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            np.testing.assert_allclose(
                rr1.ravel(), rr2.ravel(), rtol=self.rtol, atol=self.atol
            )
            assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"

    def test_tf_self_consistent(self) -> None:
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
            np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)
            assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"

    @unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
    def test_dp_consistent_with_ref(self) -> None:
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
            if rr1 is SKIP_FLAG or rr2 is SKIP_FLAG:
                continue
            np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)
            assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"

    @unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
    def test_dp_self_consistent(self) -> None:
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
                np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)
                assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"
            else:
                self.assertEqual(rr1, rr2)

    def test_pt_consistent_with_ref(self) -> None:
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
        if obj.__class__.__name__.startswith(("Polar", "Dipole", "DOS")):
            # tf, pt serialization mismatch
            common_keys = set(data1.keys()) & set(data2.keys())
            data1 = {k: data1[k] for k in common_keys}
            data2 = {k: data2[k] for k in common_keys}
        # drop @variables since they are not equal
        data1.pop("@variables", None)
        data2.pop("@variables", None)
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)
            assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"

    def test_pt_self_consistent(self) -> None:
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
                np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)
                assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"
            else:
                self.assertEqual(rr1, rr2)

    def test_jax_consistent_with_ref(self) -> None:
        """Test whether JAX and reference are consistent."""
        if self.skip_jax:
            self.skipTest("Unsupported backend")
        ref_backend = self.get_reference_backend()
        if ref_backend == self.RefBackend.JAX:
            self.skipTest("Reference is self")
        ret1, data1 = self.get_reference_ret_serialization(ref_backend)
        ret1 = self.extract_ret(ret1, ref_backend)
        jax_obj = self.jax_class.deserialize(data1)
        ret2 = self.eval_jax(jax_obj)
        ret2 = self.extract_ret(ret2, self.RefBackend.JAX)
        data2 = jax_obj.serialize()
        # drop @variables since they are not equal
        data1.pop("@variables", None)
        data2.pop("@variables", None)
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)
            assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"

    def test_jax_self_consistent(self) -> None:
        """Test whether JAX is self consistent."""
        if self.skip_jax:
            self.skipTest("Unsupported backend")
        obj1 = self.init_backend_cls(self.jax_class)
        ret1, data1 = self.get_jax_ret_serialization_from_cls(obj1)
        obj1 = self.jax_class.deserialize(data1)
        ret2, data2 = self.get_jax_ret_serialization_from_cls(obj1)
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            if isinstance(rr1, np.ndarray) and isinstance(rr2, np.ndarray):
                np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)
                assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"
            else:
                self.assertEqual(rr1, rr2)

    def test_pd_consistent_with_ref(self):
        """Test whether PD and reference are consistent."""
        if self.skip_pd:
            self.skipTest("Unsupported backend")
        ref_backend = self.get_reference_backend()
        if ref_backend == self.RefBackend.PD:
            self.skipTest("Reference is self")
        ret1, data1 = self.get_reference_ret_serialization(ref_backend)
        ret1 = self.extract_ret(ret1, ref_backend)
        obj = self.pd_class.deserialize(data1)
        ret2 = self.eval_pd(obj)
        ret2 = self.extract_ret(ret2, self.RefBackend.PD)
        data2 = obj.serialize()
        if obj.__class__.__name__.startswith(("Polar", "Dipole", "DOS")):
            # tf, pd serialization mismatch
            common_keys = set(data1.keys()) & set(data2.keys())
            data1 = {k: data1[k] for k in common_keys}
            data2 = {k: data2[k] for k in common_keys}
        # drop @variables since they are not equal
        data1.pop("@variables", None)
        data2.pop("@variables", None)
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)
            assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"

    def test_pd_self_consistent(self):
        """Test whether PD is self consistent."""
        if self.skip_pd:
            self.skipTest("Unsupported backend")
        obj1 = self.init_backend_cls(self.pd_class)
        ret1, data1 = self.get_pd_ret_serialization_from_cls(obj1)
        obj2 = self.pd_class.deserialize(data1)
        ret2, data2 = self.get_pd_ret_serialization_from_cls(obj2)
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            if isinstance(rr1, np.ndarray) and isinstance(rr2, np.ndarray):
                np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)
                assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"
            else:
                self.assertEqual(rr1, rr2)

    @unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
    def test_array_api_strict_consistent_with_ref(self) -> None:
        """Test whether array_api_strict and reference are consistent."""
        if self.skip_array_api_strict:
            self.skipTest("Unsupported backend")
        ref_backend = self.get_reference_backend()
        if ref_backend == self.RefBackend.ARRAY_API_STRICT:
            self.skipTest("Reference is self")
        ret1, data1 = self.get_reference_ret_serialization(ref_backend)
        ret1 = self.extract_ret(ret1, ref_backend)
        array_api_strict_obj = self.array_api_strict_class.deserialize(data1)
        ret2 = self.eval_array_api_strict(array_api_strict_obj)
        ret2 = self.extract_ret(ret2, self.RefBackend.ARRAY_API_STRICT)
        data2 = array_api_strict_obj.serialize()
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)
            assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"

    @unittest.skipIf(TEST_DEVICE != "cpu" and CI, "Only test on CPU.")
    def test_array_api_strict_self_consistent(self) -> None:
        """Test whether array_api_strict is self consistent."""
        if self.skip_array_api_strict:
            self.skipTest("Unsupported backend")
        obj1 = self.init_backend_cls(self.array_api_strict_class)
        ret1, data1 = self.get_array_api_strict_ret_serialization_from_cls(obj1)
        obj1 = self.array_api_strict_class.deserialize(data1)
        ret2, data2 = self.get_array_api_strict_ret_serialization_from_cls(obj1)
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            if isinstance(rr1, np.ndarray) and isinstance(rr2, np.ndarray):
                np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)
                assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"
            else:
                self.assertEqual(rr1, rr2)

    def tearDown(self) -> None:
        """Clear the TF session."""
        if not self.skip_tf:
            clear_session()


def parameterized(*attrs: tuple, **subblock_attrs: tuple) -> Callable:
    """Parameterized test.

    Original class will not be actually generated. Avoid inherbiting from it.
    New classes are generated with the name of the original class and the
    parameters.

    Parameters
    ----------
    *attrs : tuple
        The attributes to be parameterized.
    **subblock_attrs : tuple
        The sub-blocked attributes to be parameterized separately.

    Returns
    -------
    object
        The decorator.

    Examples
    --------
    >>> @parameterized(
    ...     (True, False),
    ...     (True, False),
    ... )
    ... class TestSeA(CommonTest, unittest.TestCase):
    ...     @property
    ...     def data(self) -> dict:
    ...         (
    ...             param1,
    ...             param2,
    ...         ) = self.param
    ...         return {
    ...             "param1": param1,
    ...             "param2": param2,
    ...         }
    """
    global_combine = list(itertools.product(*attrs)) if len(attrs) else []
    block_combine = []
    for kk in subblock_attrs:
        block_combine += (
            list(itertools.product(*subblock_attrs[kk]))
            if len(subblock_attrs[kk])
            else []
        )
    full_parameterized = global_combine + block_combine

    def decorator(base_class: type):
        class_module = sys.modules[base_class.__module__].__dict__
        for pp in full_parameterized:

            class TestClass(base_class):
                param: ClassVar = pp

            name = f"{base_class.__name__}_{'_'.join(str(x) for x in pp)}"

            class_module[name] = TestClass
        # make unittest module happy by ignoring the original one
        return object

    return decorator


def parameterize_func(
    func: Callable,
    param_dict_list: dict[str, tuple],
):
    """Parameterize functions with different default values.

    Parameters
    ----------
    func : Callable
        The base function.
    param_dict_list : dict[str, Tuple]
        Dictionary of parameters with default values to be changed in base function, each of which is a tuple of choices.

    Returns
    -------
    list_func
        List of parameterized functions with changed default values.

    """

    def create_wrapper(_func, _new_sig, _pp):
        def wrapper(*args, **kwargs):
            bound_args = _new_sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            return _func(*bound_args.args, **bound_args.kwargs)

        wrapper.__name__ = f"{_func.__name__}_{'_'.join(str(x) for x in _pp)}"
        wrapper.__qualname__ = wrapper.__name__
        return wrapper

    list_func = []
    param_keys = list(param_dict_list.keys())
    for pp in itertools.product(*[param_dict_list[kk] for kk in param_keys]):
        sig = inspect.signature(func)
        new_params = dict(sig.parameters)
        for ii, val in enumerate(pp):
            val_name = param_keys[ii]
            # only change the default value of func
            new_params[val_name] = sig.parameters[val_name].replace(default=val)
        new_sig = sig.replace(parameters=list(new_params.values()))
        list_func.append(create_wrapper(func, new_sig, pp))

    return list_func
