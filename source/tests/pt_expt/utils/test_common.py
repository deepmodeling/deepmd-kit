# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import torch

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.pt_expt.common import (
    to_torch_array,
    torch_module,
)
from deepmd.pt_expt.utils import (
    env,
)


def test_to_torch_array_moves_device() -> None:
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = to_torch_array(arr)
    assert torch.is_tensor(tensor)
    assert tensor.device == env.DEVICE

    input_tensor = torch.as_tensor(arr, device=torch.device("cpu"))
    output_tensor = to_torch_array(input_tensor)
    assert torch.is_tensor(output_tensor)
    assert output_tensor.device == env.DEVICE


def test_torch_module_auto_generates_forward() -> None:
    """Test that torch_module auto-generates forward() from call()."""

    class MockNativeOP(NativeOP):
        def call(self, x: np.ndarray) -> np.ndarray:
            return x * 2

    @torch_module
    class MockModule(MockNativeOP):
        pass

    module = MockModule()
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device=torch.device("cpu"))
    output = module(input_tensor)
    expected = input_tensor * 2
    assert torch.allclose(output, expected)


def test_torch_module_auto_generates_forward_lower() -> None:
    """Test that torch_module auto-generates forward_lower() from call_lower()."""

    class MockNativeOP(NativeOP):
        def call(self, x: np.ndarray) -> np.ndarray:
            return x

        def call_lower(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return x + y

    @torch_module
    class MockModule(MockNativeOP):
        pass

    module = MockModule()
    input_x = torch.tensor([1.0, 2.0, 3.0], device=torch.device("cpu"))
    input_y = torch.tensor([4.0, 5.0, 6.0], device=torch.device("cpu"))
    output = module.forward_lower(input_x, input_y)
    expected = input_x + input_y
    assert torch.allclose(output, expected)


def test_torch_module_respects_explicit_forward() -> None:
    """Test that torch_module doesn't override an explicitly defined forward()."""

    class MockNativeOP(NativeOP):
        def call(self, x: np.ndarray) -> np.ndarray:
            return x * 2

    @torch_module
    class MockModule(MockNativeOP):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # This should override the auto-generated forward
            return x * 3

    module = MockModule()
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device=torch.device("cpu"))
    output = module(input_tensor)
    expected = input_tensor * 3  # Should use the explicit forward, not call()
    assert torch.allclose(output, expected)


def test_torch_module_respects_explicit_forward_lower() -> None:
    """Test that torch_module doesn't override an explicitly defined forward_lower()."""

    class MockNativeOP(NativeOP):
        def call(self, x: np.ndarray) -> np.ndarray:
            return x

        def call_lower(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return x + y

    @torch_module
    class MockModule(MockNativeOP):
        def forward_lower(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # This should override the auto-generated forward_lower
            return x - y

    module = MockModule()
    input_x = torch.tensor([1.0, 2.0, 3.0], device=torch.device("cpu"))
    input_y = torch.tensor([4.0, 5.0, 6.0], device=torch.device("cpu"))
    output = module.forward_lower(input_x, input_y)
    expected = input_x - input_y  # Should use the explicit forward_lower
    assert torch.allclose(output, expected)


def test_torch_module_handles_no_call_lower_method() -> None:
    """Test that torch_module works even when there's no call_lower() method."""

    class MockNativeOP(NativeOP):
        def call(self, x: np.ndarray) -> np.ndarray:
            return x * 2

    @torch_module
    class MockModule(MockNativeOP):
        pass

    module = MockModule()
    # Should have forward method since there's a call method
    assert hasattr(module, "forward")
    # Should not have forward_lower method since there's no call_lower method
    assert not hasattr(module, "forward_lower")
