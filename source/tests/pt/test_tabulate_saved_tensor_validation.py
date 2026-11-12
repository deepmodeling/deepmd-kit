# SPDX-License-Identifier: LGPL-3.0-or-later
"""Validate tensors restored by autograd before native tabulation dispatch."""

import pytest
import torch

from deepmd.pt.cxx_op import (
    ENABLE_CUSTOMIZED_OP,
)
from deepmd.pt.utils import (
    env,
)

pytestmark = pytest.mark.skipif(
    not ENABLE_CUSTOMIZED_OP, reason="PyTorch customized OPs are not built"
)

OP_NAMES = ("se_a", "se_atten", "se_t", "se_t_tebd", "se_r")


def _inputs(op_name: str) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Use distinct buffers so a hook can identify every saved tensor."""
    tensors = {
        "table": torch.linspace(
            0.1, 0.9, 96, dtype=torch.float64, device=env.DEVICE
        ).reshape(8, 12),
        "table_info": torch.tensor(
            [0.0, 1.0, 2.0, 0.5, 0.5], dtype=torch.float64, device="cpu"
        ),
    }

    def data(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.full(
            shape, 0.3, dtype=torch.float64, device=env.DEVICE, requires_grad=True
        )

    if op_name == "se_r":
        tensors["em"] = data((2, 2))
        return tensors, (tensors["em"],)

    se_a = op_name in ("se_a", "se_atten")
    tensors["em_x"] = data((2, 2 if se_a else 4))
    tensors["em"] = data((2, 2, 4 if se_a else 2))
    inputs = (tensors["em_x"],)
    if op_name != "se_t_tebd":
        inputs += (tensors["em"],)
    if op_name == "se_atten":
        tensors["two_embed"] = data((4, 2))
        inputs += (tensors["two_embed"],)
    return tensors, inputs


def _forward(op_name: str, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    args = [tensors["table"], tensors["table_info"]]
    if op_name != "se_r":
        args.append(tensors["em_x"])
    args.append(tensors["em"])
    if op_name == "se_atten":
        args.append(tensors["two_embed"])
    args.append(2)
    if op_name == "se_atten":
        args.append(False)
    return getattr(torch.ops.deepmd, f"tabulate_fusion_{op_name}")(*args)[0]


@pytest.mark.parametrize("op_name", OP_NAMES)
@pytest.mark.parametrize("derivative_order", (1, 2))
@pytest.mark.parametrize(
    "invalid_tensor", ("table_info", "table", "embedding", "descriptor")
)
def test_rejects_invalid_saved_tensor(
    op_name: str, derivative_order: int, invalid_tensor: str
) -> None:
    """First and second derivatives must recheck restored buffer contracts."""
    tensors, inputs = _inputs(op_name)
    target_name = invalid_tensor
    if invalid_tensor == "embedding":
        target_name = "em" if op_name == "se_r" else "em_x"
    messages = {
        "table_info": "table_info must contain at least 5 values",
        "table": "table does not contain enough coefficients",
        "embedding": f"{target_name} must be rank 2",
        "descriptor": "descriptor must have shape",
    }
    unpack_count = 0

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal unpack_count
        target = tensors[target_name]
        if tensor.device != target.device or tensor.data_ptr() != target.data_ptr():
            return tensor
        unpack_count += 1
        if invalid_tensor == "table_info":
            # Keep the complete backing allocation: the pre-fix implementation
            # can demonstrate the missing guard without an out-of-allocation read.
            return tensor[:4]
        if invalid_tensor == "embedding":
            if op_name == "se_r":
                return tensor.reshape(-1)
            return tensor.reshape(-1)[:-1].reshape(1, -1)
        return tensor[:1]

    hooks = torch.autograd.graph.saved_tensors_hooks(torch.Tensor.detach, unpack)
    if derivative_order == 1:
        with hooks:
            descriptor = _forward(op_name, tensors)
        tensors["descriptor"] = descriptor
        with pytest.raises(RuntimeError, match=messages[invalid_tensor]):
            torch.autograd.grad(descriptor.sum(), inputs)
    else:
        descriptor = _forward(op_name, tensors)
        tensors["descriptor"] = descriptor
        dy = torch.ones_like(descriptor, requires_grad=True)
        # Only the gradient op saves through the hook. Its inputs remain valid
        # for the first backward and become invalid when the second unpacks them.
        with hooks:
            gradients = torch.autograd.grad(descriptor, inputs, dy, create_graph=True)
        with pytest.raises(RuntimeError, match=messages[invalid_tensor]):
            torch.autograd.grad(sum(gradient.sum() for gradient in gradients), dy)
    assert unpack_count > 0
