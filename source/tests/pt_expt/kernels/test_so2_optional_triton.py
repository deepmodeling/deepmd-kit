# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU stack differentiation must not import the CUDA-only cohort launcher."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("without_triton", [False, True])
def test_cpu_stack_backward_keeps_optional_triton(without_triton):
    """Exercise registered autograd in a fresh interpreter, not a mocked backward."""
    code = textwrap.dedent(
        """
        import sys
        import torch

        without_triton = sys.argv[1] == "True"
        if without_triton:
            # A subprocess prevents imports cached by other tests from hiding
            # an unconditional Triton dependency in the registered backward.
            for name in list(sys.modules):
                if name == "triton" or name.startswith("triton."):
                    del sys.modules[name]

            class BlockTriton:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "triton" or fullname.startswith("triton."):
                        raise ModuleNotFoundError(
                            "Triton intentionally unavailable", name=fullname
                        )
                    return None

            sys.meta_path.insert(0, BlockTriton())

        from deepmd.pt_expt.kernels.triton.sezm import so2_value_path as sv

        if without_triton:
            assert not sv.SO2_VALUE_PATH_TRITON_AVAILABLE
        torch.set_num_threads(1)
        torch.manual_seed(719)
        for edges in (1, 37):
            for apply_alpha in (False, True):
                u = (torch.randn(2, edges, 320) * 0.1).requires_grad_()
                alpha = (torch.rand(edges, 2) * 0.8 + 0.1).requires_grad_()
                w0 = torch.randn(3, 2, 128, 128) * 0.01
                w1 = torch.randn(3, 2, 192, 192) * 0.01
                gw = torch.randn(2, 2, 32, 96) * 0.01
                parameters = (w0, w1, gw, 3, 32, apply_alpha)
                actual = sv._mixing_stack_op(u, alpha, *parameters)[0]
                expected = sv._mixing_stack_reference(u, alpha, *parameters)[0]
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
                upstream = torch.randn_like(actual)
                gradients = []
                for output in (actual, expected):
                    gradients.append(torch.autograd.grad(
                        output, (u, alpha), grad_outputs=upstream, allow_unused=True
                    ))
                for got, ref in zip(*gradients, strict=True):
                    if ref is None:
                        assert got is None
                    else:
                        assert torch.isfinite(got).all()
                        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)
        assert "deepmd.pt_expt.kernels.triton.sezm.so2_edge_cohort" not in sys.modules
        print("CPU reference forward/backward passed without importing ECS")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(without_triton)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
