# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parameter-gradient parity: pt SeZM (reference) vs pt_expt DPA4 wrappers.

The pt_expt wrappers register dpmodel numpy attributes as torch buffers by
default; ``_promote_trainable_tree`` (deepmd/pt_expt/descriptor/dpa4.py) must
re-register every weight that is a trainable ``nn.Parameter`` in pt as a
Parameter so the optimizer sees it and autograd populates its grad.  This file
proves that promotion is complete and correct:

- weights are transferred pt -> pt_expt via serialize()/deserialize(),
- the forward outputs must match (guard assertion),
- a quadratic loss is backpropagated on both sides,
- every gradient is compared 1:1 through the shared serialization contract:
  both sides serialize to the same ``@variables`` key names (pt state_dict key
  contract), so swapping each Parameter's data with its grad and re-serializing
  yields name-aligned gradient trees.  A weight that is a Parameter in pt but
  was left a buffer in pt_expt shows up as grad-vs-weight mismatch (no silent
  drops); parameter counts are asserted equal as well.
"""

import numpy as np
import pytest
import torch

from deepmd.pt.utils import env as pt_env

from .test_dpa4_dpmodel_parity import (
    _build_descriptor_inputs,
)

PT_DEVICE = pt_env.DEVICE
_ON_CPU = PT_DEVICE.type == "cpu"
# device-conditional gates (see test_dpa4_dpmodel_parity.py header)
PT_RTOL, PT_ATOL = (1e-12, 1e-14) if _ON_CPU else (1e-10, 1e-12)


def to_pt(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(PT_DEVICE)


def _flatten_arrays(data, prefix="") -> dict[str, np.ndarray]:
    """Flatten a serialize() tree to {dotted-path: ndarray}."""
    out: dict[str, np.ndarray] = {}
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, (list, tuple)):
        items = ((str(i), v) for i, v in enumerate(data))
    else:
        return out
    for k, v in items:
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, np.ndarray):
            out[key] = v
        elif isinstance(v, (dict, list, tuple)):
            out.update(_flatten_arrays(v, key))
    return out


def _swap_data_with_grad(module: torch.nn.Module) -> int:
    """Replace each Parameter's data with its gradient, in place.

    Every requires-grad Parameter must have received a non-None grad
    (asserted); returns the number of swapped parameters.
    """
    missing = [
        n for n, p in module.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, f"parameters with no grad after backward: {missing}"
    n_swapped = 0
    with torch.no_grad():
        for _, p in module.named_parameters():
            if p.requires_grad:
                p.data = p.grad.detach().clone()
                n_swapped += 1
    return n_swapped


def _assert_grad_trees_match(pt_mod, expt_mod, rtol=PT_RTOL, atol=PT_ATOL) -> None:
    """Swap data<->grad on both sides, serialize, compare name-aligned."""
    n_pt = _swap_data_with_grad(pt_mod)
    n_expt = _swap_data_with_grad(expt_mod)
    # exact trainable-parameter count parity: a buffer wrongly promoted (or a
    # parameter left as buffer) changes the count
    assert n_pt == n_expt, f"trainable parameter count: pt {n_pt} vs pt_expt {n_expt}"
    ref = _flatten_arrays(pt_mod.serialize())
    res = _flatten_arrays(expt_mod.serialize())
    assert sorted(ref) == sorted(res)
    for key in sorted(ref):
        np.testing.assert_allclose(
            res[key],
            ref[key],
            rtol=rtol,
            atol=atol,
            err_msg=f"gradient mismatch for {key}",
        )


class TestDescriptorGradParity:
    nf = 2
    nloc = 6
    nall = 10
    nnei = 20
    ntypes = 2

    def _build_pair(self, **overrides):
        from deepmd.pt.model.descriptor.sezm import (
            DescrptSeZM,
        )
        from deepmd.pt_expt.descriptor.dpa4 import (
            DescrptDPA4,
        )

        kwargs = {
            "ntypes": self.ntypes,
            "sel": self.nnei,
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
        pt_mod = DescrptSeZM(**kwargs).double()
        # several projections are zero-initialized; perturb for nonzero
        # output and weight-dependent gradients everywhere
        rng = np.random.default_rng(2150)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += to_pt(0.05 * rng.normal(size=tuple(p.shape)))
        expt_mod = DescrptDPA4.deserialize(pt_mod.serialize())
        return pt_mod, expt_mod

    def _inputs(self, seed=2151):
        rng = np.random.default_rng(seed)
        return _build_descriptor_inputs(
            rng,
            nf=self.nf,
            nloc=self.nloc,
            nall=self.nall,
            nnei=self.nnei,
            ntypes=self.ntypes,
        )

    @pytest.mark.parametrize("use_env_seed", [False, True])  # env FiLM (film_* params)
    def test_descriptor_grad_parity(self, use_env_seed) -> None:
        pt_mod, expt_mod = self._build_pair(use_env_seed=use_env_seed)
        inp = self._inputs()
        coord = inp["coord"].reshape(self.nf, -1)
        atype_ext, nlist, mapping = inp["atype_ext"], inp["nlist"], inp["mapping"]

        out_pt = pt_mod(
            to_pt(inp["coord"]),
            to_pt(atype_ext),
            to_pt(nlist),
            mapping=to_pt(mapping),
        )[0]
        out_expt = expt_mod(
            to_pt(coord),
            to_pt(atype_ext.astype(np.int64)),
            to_pt(nlist.astype(np.int64)),
            mapping=to_pt(mapping.astype(np.int64)),
        )[0]
        # guard: forward outputs must match before comparing gradients
        np.testing.assert_allclose(
            out_expt.detach().cpu().numpy(),
            out_pt.detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-12,
        )
        # quadratic loss -> dL/dw depends on the weights, not just the inputs
        (out_pt**2).sum().backward()
        (out_expt**2).sum().backward()
        # descriptor-level gate (same as the forward parity gate in
        # test_dpa4_dpmodel_parity.py): grads chain the full descriptor
        # math, where fp64 accumulation-order drift reaches ~3e-11 rel
        _assert_grad_trees_match(pt_mod, expt_mod, rtol=1e-10, atol=1e-12)

    def test_descriptor_grad_parity_native_spin(self) -> None:
        # Native per-atom spin (``use_spin``) adds trainable Parameters that
        # pt_expt must promote from dpmodel numpy->buffer:
        # ``SpinEmbedding.{adam_spin_vec_weight, adam_spin_nbr_weight}``,
        # its ``mag_layer1/2`` weights (NativeLayer auto-promotes), and
        # ``EnvironmentInitialEmbedding.spin_scale``.  A missing promotion
        # surfaces as the trainable-parameter-count mismatch asserted inside
        # ``_assert_grad_trees_match`` (n_pt vs n_expt).  Type 0 carries spin;
        # the fixture's local types include type-0 atoms that are also edge
        # sources, so both the on-site (l=0 magnitude + l=1 direction) and the
        # neighbor-aggregation (edge l=1, gated by ``spin_scale``) paths fire.
        use_spin = [True, False]  # ntypes == 2; type 0 is spin-active
        pt_mod, expt_mod = self._build_pair(use_spin=use_spin)
        inp = self._inputs()
        coord = inp["coord"].reshape(self.nf, -1)
        atype_ext, nlist, mapping = inp["atype_ext"], inp["nlist"], inp["mapping"]
        # a spin-active type-0 atom must be local for the on-site spin path
        assert (atype_ext[:, : self.nloc] == 0).any()
        rng = np.random.default_rng(2170)
        spin = rng.normal(size=(self.nf, self.nloc, 3))

        out_pt = pt_mod(
            to_pt(inp["coord"]),
            to_pt(atype_ext),
            to_pt(nlist),
            mapping=to_pt(mapping),
            spin=to_pt(spin),
        )[0]
        out_expt = expt_mod(
            to_pt(coord),
            to_pt(atype_ext.astype(np.int64)),
            to_pt(nlist.astype(np.int64)),
            mapping=to_pt(mapping.astype(np.int64)),
            spin=to_pt(spin),
        )[0]
        # guard: forward outputs must match before comparing gradients
        np.testing.assert_allclose(
            out_expt.detach().cpu().numpy(),
            out_pt.detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-12,
        )
        # guard: the spin tensor must actually move the descriptor, otherwise
        # the spin-parameter gradients below would be a trivial (zero) match
        with torch.no_grad():
            out_pt_nospin = pt_mod(
                to_pt(inp["coord"]),
                to_pt(atype_ext),
                to_pt(nlist),
                mapping=to_pt(mapping),
                spin=None,
            )[0]
        assert (out_pt - out_pt_nospin).abs().max().item() > 1e-3
        # quadratic loss -> dL/dw depends on the weights, not just the inputs
        (out_pt**2).sum().backward()
        (out_expt**2).sum().backward()
        # count parity (validates spin-Parameter promotion) + name-aligned
        # gradient parity across the full descriptor, spin parameters included
        _assert_grad_trees_match(pt_mod, expt_mod, rtol=1e-10, atol=1e-12)


class TestFittingGradParity:
    nf = 2
    nloc = 6
    in_dim = 12
    ntypes = 2

    def _build_pair(self, **overrides):
        from deepmd.pt.model.task.sezm_ener import (
            SeZMEnergyFittingNet as SeZMEnergyFittingNetPT,
        )
        from deepmd.pt_expt.fitting.dpa4_ener import (
            SeZMEnergyFittingNet as SeZMEnergyFittingNetExpt,
        )

        kwargs = {
            "ntypes": self.ntypes,
            "dim_descrpt": self.in_dim,
            "neuron": [16, 16],
            "precision": "float64",
            "seed": 7,
        }
        kwargs.update(overrides)
        pt_mod = SeZMEnergyFittingNetPT(**kwargs)
        # bias_atom_e is zero-initialized; perturb for a nontrivial bias path
        rng = np.random.default_rng(2160)
        with torch.no_grad():
            pt_mod.bias_atom_e += to_pt(
                rng.normal(size=tuple(pt_mod.bias_atom_e.shape))
            )
        expt_mod = SeZMEnergyFittingNetExpt.deserialize(pt_mod.serialize())
        return pt_mod, expt_mod

    def _inputs(self, seed=2161):
        rng = np.random.default_rng(seed)
        descriptor = rng.normal(size=(self.nf, self.nloc, self.in_dim))
        atype = rng.integers(0, self.ntypes, size=(self.nf, self.nloc))
        atype[0, 0], atype[0, 1] = 0, 1
        return descriptor, atype

    @pytest.mark.parametrize("bias_out", [False, True])  # output-layer bias
    def test_fitting_grad_parity(self, bias_out) -> None:
        pt_mod, expt_mod = self._build_pair(bias_out=bias_out)
        descriptor, atype = self._inputs()

        out_pt = pt_mod(to_pt(descriptor), to_pt(atype))["energy"]
        out_expt = expt_mod(to_pt(descriptor), to_pt(atype.astype(np.int64)))["energy"]
        # guard: forward outputs must match before comparing gradients
        np.testing.assert_allclose(
            out_expt.detach().cpu().numpy(),
            out_pt.detach().cpu().numpy(),
            rtol=PT_RTOL,
            atol=PT_ATOL,
        )
        # quadratic loss: energy.sum() would make the grad of the last
        # linear layer independent of upstream weights
        (out_pt**2).sum().backward()
        (out_expt**2).sum().backward()
        _assert_grad_trees_match(pt_mod, expt_mod)
