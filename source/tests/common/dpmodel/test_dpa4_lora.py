# SPDX-License-Identifier: LGPL-3.0-or-later
"""Torch-free tests for the dpmodel DPA4 (SeZM) LoRA fine-tune freeze policy."""

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
