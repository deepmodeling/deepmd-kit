# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the dpmodel DPA4 (SeZM) LoRA adapters: the fine-tune freeze
policy and the ``LoRASO3`` contraction contract (torch imported lazily).
"""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.dpmodel.descriptor.dpa4_nn.lora import (
    _iter_named_modules,
    apply_lora_to_sezm,
    has_lora,
)


def make_descriptor(**overrides) -> DescrptDPA4:
    kwargs = {
        "ntypes": 2,
        "sel": 8,
        "rcut": 4.0,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "grid_branch": [1, 1, 1],
        "s2_activation": [False, True],
        "random_gamma": False,
        "precision": "float64",
        "seed": 7,
    }
    kwargs.update(overrides)
    return DescrptDPA4(**kwargs)


def test_apply_lora_marks_adapters_trainable() -> None:
    # apply_lora freezes the pre-trained backbone and injects LoRASO3 / LoRASO2
    # adapters.  The dpmodel tracks trainability per module, so every injected
    # adapter module must be marked trainable for its low-rank delta to receive
    # gradients.  Regression for the ``_UNFREEZE_LEAF_NAMES`` adapter entries:
    # without them the adapters inherit ``trainable=False`` from the frozen base
    # (the base is built frozen) and would stay frozen, so fine-tuning would be
    # a no-op.
    dd = make_descriptor()
    apply_lora_to_sezm(dd, rank=2)
    assert has_lora(dd)

    modules = list(_iter_named_modules(dd))
    adapters = [m for _name, m in modules if type(m).__name__ in ("LoRASO3", "LoRASO2")]
    assert adapters, "apply_lora injected no LoRA adapter modules"
    still_frozen = [m for m in adapters if not m.trainable]
    assert not still_frozen, f"{len(still_frozen)} LoRA adapter module(s) left frozen"

    # The pre-trained backbone is otherwise frozen: the type embedding carries a
    # converged ``adam_type_embedding`` that ``apply_lora`` override-freezes, so
    # the policy is a genuine freeze (not a trivial unfreeze-everything).
    type_embeddings = [
        m for _name, m in modules if type(m).__name__ == "SeZMTypeEmbedding"
    ]
    assert type_embeddings
    assert all(not m.trainable for m in type_embeddings)


@pytest.mark.parametrize(
    "n_focus",
    [
        1,  # the common single-focus configuration
        2,  # shipped spin/property DPA4 examples
    ],
)
def test_lora_so3_call_matches_einsum_contract(n_focus) -> None:
    """Direct regression for the dpmodel ``LoRASO3.call`` contraction.

    ``B_by_l`` is set nonzero so the adapter delta participates; the
    forward must equal ``einsum("ndfi,difo->ndfo")`` over the effective
    (base + scaled ``B @ A``) per-degree weight, on both the NumPy and
    the Torch array namespaces.
    """
    import torch

    from deepmd.dpmodel.descriptor.dpa4_nn.lora import (
        LoRASO3,
    )

    lmax, cin, cout, rank = 2, 3, 4, 2
    mod = LoRASO3(
        lmax=lmax,
        in_channels=cin,
        out_channels=cout,
        n_focus=n_focus,
        precision="float64",
        trainable=True,
        seed=3,
        lora_rank=rank,
    )
    rng = np.random.default_rng(11)
    # unlock the adapter: B is zero-initialised, which would hide a delta bug
    mod.B_by_l = rng.normal(size=mod.B_by_l.shape).astype(np.float64)

    coeff_dim = (lmax + 1) ** 2
    x = rng.normal(size=(5, coeff_dim, n_focus, cin))
    delta = np.matmul(mod.B_by_l, mod.A_by_l).transpose(0, 2, 1) * mod.scaling
    w_eff = np.reshape(mod.weight + delta, (lmax + 1, cin, n_focus, cout))
    w_deg = w_eff[np.asarray(mod.expand_index)]  # (D, Cin, F, Cout)
    ref = np.einsum("ndfi,difo->ndfo", x, w_deg)

    out_np = np.asarray(mod.call(x))
    np.testing.assert_allclose(out_np, ref, rtol=1e-12, atol=1e-12)
    out_torch = mod.call(torch.from_numpy(x))
    np.testing.assert_allclose(
        out_torch.detach().cpu().numpy(), ref, rtol=1e-12, atol=1e-12
    )
