# SPDX-License-Identifier: LGPL-3.0-or-later
"""Guard that resolving unused DPA4 capacities cannot change graph training."""

import copy

import numpy as np
import torch

from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.graph_lower import (
    model_config_skips_neighbor_stat,
    resolve_stat_free_selections,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...dpa4_fixtures import (
    jitter_zero_arrays,
)


def test_stat_free_dpa4_graph_training_equivalence(monkeypatch):
    """A binding sel, a large sel and resolved auto have identical training."""
    config = {
        "type": "dpa4",
        "type_map": ["H", "O"],
        "descriptor": {
            "type": "dpa4",
            "sel": "auto",
            "rcut": 4.0,
            "channels": 8,
            "n_radial": 4,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": 1,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [8],
            "precision": "float64",
            "seed": 1,
        },
    }
    assert model_config_skips_neighbor_stat(config)
    resolved = resolve_stat_free_selections(config)
    assert config["descriptor"]["sel"] == "auto"
    configs = [copy.deepcopy(config), copy.deepcopy(config), resolved]
    configs[0]["descriptor"]["sel"] = 1
    configs[1]["descriptor"]["sel"] = 256
    models = [get_model(c).to(env.DEVICE) for c in configs]
    # Compare independently initialized weights/buffers, not loaded clones.
    states = [m.state_dict() for m in models]
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        for key in state:
            torch.testing.assert_close(state[key], states[0][key], rtol=0, atol=0)
    # Fresh DPA4 residuals are zero: make the model sensitive to geometry.
    for model in models:
        data = model.atomic_model.descriptor.serialize()
        data = jitter_zero_arrays(data, np.random.default_rng(99))
        model.atomic_model.descriptor = DescrptDPA4.deserialize(data).to(env.DEVICE)
        model.train()
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 1.1, 0.3], [1.0, 1.0, 0.7]]],
        dtype=torch.float64,
        device=env.DEVICE,
    )
    atype = torch.tensor([[0, 1, 0, 1]], device=env.DEVICE)
    box = (torch.eye(3, dtype=torch.float64, device=env.DEVICE) * 12).reshape(1, 9)
    sample = [{"coord": coord, "box": box, "atype": atype}]
    for model in models:
        model.atomic_model.descriptor.compute_input_stats(sample)
    stat_states = [m.state_dict() for m in models]
    for state in stat_states[1:]:
        assert state.keys() == stat_states[0].keys()
        for key in state:
            torch.testing.assert_close(state[key], stat_states[0][key], rtol=0, atol=0)
    calls = []
    original = DescrptDPA4.call_graph

    def spy(self, graph, *args, **kwargs):
        calls.append(int(graph.edge_mask.sum().item()))
        return original(self, graph, *args, **kwargs)

    monkeypatch.setattr(DescrptDPA4, "call_graph", spy)
    outputs, grads, updated = [], [], []
    for model in models:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        out = model.forward_common(coord.clone().requires_grad_(True), atype, box)
        outputs.append(
            {
                k: out[k].detach().clone()
                for k in ("energy_redu", "energy_derv_r", "energy_derv_c_redu")
            }
        )
        assert out["energy_derv_r"].abs().max().item() > 1e-8
        loss = out["energy_redu"].square().sum() + out["energy_derv_r"].square().sum()
        loss.backward()
        grads.append(
            {
                n: None if p.grad is None else p.grad.detach().clone()
                for n, p in model.named_parameters()
            }
        )
        optimizer.step()
        updated.append(copy.deepcopy(model.state_dict()))
    # Four mutually neighbouring atoms have 12 directed edges; sel=1 must
    # not truncate this to four edges on the actual model forward route.
    assert len(calls) >= 3
    assert all(count == 12 for count in calls)
    for records in (outputs, grads, updated):
        for record in records[1:]:
            assert record.keys() == records[0].keys()
            for key, value in record.items():
                reference = records[0][key]
                if reference is None:
                    assert value is None
                else:
                    torch.testing.assert_close(value, reference, rtol=1e-10, atol=1e-12)
    perturbed = coord.clone()
    perturbed[0, 0, 0] += 0.1
    model = models[0]
    baseline = model.forward_common(coord.clone(), atype, box)["energy_redu"]
    changed = model.forward_common(perturbed, atype, box)["energy_redu"]
    assert (changed - baseline).abs().max().item() > 1e-8
