# SPDX-License-Identifier: LGPL-3.0-or-later
"""Equivalence of the vesin O(N) ``NeighborList`` strategy with the default builder.

A ``NeighborList`` strategy is injected at ``forward_common``/``call_common``,
replacing the dense all-pairs ghost expansion (~27*N images + an O(N^2) distance
matrix) with vesin's O(N) cell list.  Both strategies hand the *same* extended
representation to the downstream model, so every model output (energy, force,
virial, atomic virial) must match the default builder to fp round-off.

Two layers are covered:

* :class:`TestNeighborListBuilder` -- the builder in isolation, asserting the
  per-atom neighbor *distance multisets* match the default, for the numpy
  (dpmodel) and torch (pt/pt_expt) namespaces, periodic and non-periodic, and
  that the returned tensors live on the input device.
* :class:`TestNeighborListModelEquivalence` -- full model equivalence across
  descriptor families (non-mixed, attention/mixed-types, message-passing with
  single and multiple cutoffs, repflows, hybrid), for dpmodel (energy/atomic
  energy) and pt_expt (energy/force/virial/atomic virial), periodic and
  non-periodic, including the ``neighbor_list=None`` default falling back to the
  dense builder byte-identically.
"""

import copy
import unittest

import numpy as np
import torch

from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.dpmodel.utils import (
    DefaultNeighborList,
    NeighborList,
)
from deepmd.pt_expt.model import (
    get_model,
)
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    VesinNeighborList,
    is_vesin_torch_available,
)

from ...seed import (
    GLOBAL_SEED,
)

# --- compact model configs (3-type type_map), reduced layers for test speed ---
TYPE_MAP = ["O", "H", "B"]

model_se_e2_a = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_e2_a",
        "sel": [20, 20, 8],
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12],
        "resnet_dt": False,
        "axis_neuron": 4,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_se_r = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_e2_r",
        "sel": [20, 20, 8],
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12],
        "resnet_dt": False,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_se_e3 = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_e3",
        "sel": [12, 12, 4],
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12],
        "resnet_dt": False,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_dpa1 = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_atten",
        "sel": 40,
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12, 24],
        "axis_neuron": 4,
        "attn": 16,
        "attn_layer": 2,
        "attn_dotr": True,
        "attn_mask": False,
        "activation_function": "tanh",
        "scaling_factor": 1.0,
        "normalize": False,
        "temperature": 1.0,
        "set_davg_zero": True,
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_se_atten_v2 = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_atten_v2",
        "sel": 40,
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12, 24],
        "axis_neuron": 4,
        "attn": 16,
        "attn_layer": 2,
        "attn_dotr": True,
        "attn_mask": False,
        "activation_function": "tanh",
        "set_davg_zero": False,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_dpa2 = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "dpa2",
        "repinit": {
            "rcut": 6.0,
            "rcut_smth": 2.0,
            "nsel": 30,
            "neuron": [2, 4, 8],
            "axis_neuron": 4,
            "activation_function": "tanh",
        },
        "repformer": {
            "rcut": 4.0,
            "rcut_smth": 0.5,
            "nsel": 20,
            "nlayers": 2,
            "g1_dim": 8,
            "g2_dim": 5,
            "attn2_hidden": 3,
            "attn2_nhead": 1,
            "attn1_hidden": 5,
            "attn1_nhead": 1,
            "axis_neuron": 4,
            "update_h2": False,
            "update_g1_has_conv": True,
            "update_g1_has_grrg": True,
            "update_g1_has_drrd": True,
            "update_g1_has_attn": True,
            "update_g2_has_g1g1": True,
            "update_g2_has_attn": True,
            "attn2_has_gate": True,
        },
        "seed": 1,
        "add_tebd_to_repinit_out": False,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_dpa3 = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "dpa3",
        "repflow": {
            "n_dim": 12,
            "e_dim": 8,
            "a_dim": 6,
            "nlayers": 2,
            "e_rcut": 6.0,
            "e_rcut_smth": 3.0,
            "e_sel": 20,
            "a_rcut": 4.0,
            "a_rcut_smth": 2.0,
            "a_sel": 10,
            "axis_neuron": 4,
            "update_angle": True,
            "update_style": "res_residual",
            "update_residual": 0.1,
            "update_residual_init": "const",
        },
        "activation_function": "tanh",
        "use_tebd_bias": False,
        "concat_output_tebd": False,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_hybrid = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "hybrid",
        "list": [
            {
                "type": "se_e2_a",
                "sel": [20, 20, 8],
                "rcut_smth": 0.5,
                "rcut": 4.0,
                "neuron": [6, 12],
                "resnet_dt": False,
                "axis_neuron": 4,
                "seed": 1,
            },
            {
                "type": "se_e2_r",
                "sel": [20, 20, 8],
                "rcut_smth": 0.5,
                "rcut": 4.0,
                "neuron": [6, 12],
                "resnet_dt": False,
                "seed": 1,
            },
        ],
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

ALL_MODELS = {
    "se_e2_a": model_se_e2_a,
    "se_r": model_se_r,
    "se_e3": model_se_e3,
    "dpa1": model_dpa1,
    "se_atten_v2": model_se_atten_v2,
    "dpa2": model_dpa2,
    "dpa3": model_dpa3,
    "hybrid": model_hybrid,
}


def _system(natoms: int = 6, box_len: float = 10.0, seed: int = GLOBAL_SEED):
    """A small 3-type periodic system; returns numpy (coord, atype, box)."""
    rng = np.random.default_rng(seed)
    coord = (rng.random((1, natoms, 3)) * box_len).astype(np.float64)
    atype = np.array([[0, 0, 1, 1, 2, 0]], dtype=np.int64)[:, :natoms]
    box = (np.eye(3) * box_len).reshape(1, 9).astype(np.float64)
    return coord, atype, box


def _per_atom_neighbor_dists(ext_coord, nlist, coord):
    """Sorted, rounded valid-neighbor distances for each local atom."""
    ext_coord = np.asarray(ext_coord).reshape(-1, 3)
    coord = np.asarray(coord).reshape(-1, 3)
    out = []
    for i in range(coord.shape[0]):
        ds = [
            round(float(np.linalg.norm(ext_coord[j] - coord[i])), 6)
            for j in np.asarray(nlist)[i]
            if j >= 0
        ]
        out.append(sorted(ds))
    return out


@unittest.skipIf(not is_vesin_torch_available(), "vesin.torch is not installed")
class TestNeighborListBuilder(unittest.TestCase):
    """The vesin builder must produce the same neighbor relationships as the
    default all-pairs builder, in both namespaces and on the input device.
    """

    def setUp(self) -> None:
        self.coord_np, self.atype_np, self.box_np = _system()
        self.rcut = 4.0
        self.sel = [20, 20, 8]

    def _compare(self, box_np) -> None:
        default = DefaultNeighborList()
        vesin = VesinNeighborList()
        # numpy (dpmodel) namespace
        ec_d, _, nl_d, _ = default.build(
            self.coord_np, self.atype_np, box_np, self.rcut, self.sel
        )
        ec_v, _, nl_v, _ = vesin.build(
            self.coord_np, self.atype_np, box_np, self.rcut, self.sel
        )
        self.assertEqual(
            _per_atom_neighbor_dists(ec_d, nl_d[0], self.coord_np[0]),
            _per_atom_neighbor_dists(ec_v, nl_v[0], self.coord_np[0]),
        )
        # torch namespace
        coord_t = torch.tensor(self.coord_np, dtype=torch.float64)
        atype_t = torch.tensor(self.atype_np, dtype=torch.int64)
        box_t = None if box_np is None else torch.tensor(box_np, dtype=torch.float64)
        ec_vt, _, nl_vt, _ = vesin.build(coord_t, atype_t, box_t, self.rcut, self.sel)
        self.assertTrue(torch.is_tensor(ec_vt))
        self.assertEqual(
            _per_atom_neighbor_dists(ec_d, nl_d[0], self.coord_np[0]),
            _per_atom_neighbor_dists(
                ec_vt[0].cpu().numpy(), nl_vt[0].cpu().numpy(), self.coord_np[0]
            ),
        )

    def test_pbc(self) -> None:
        self._compare(self.box_np)

    def test_nopbc(self) -> None:
        self._compare(None)

    def test_outputs_on_input_device(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        coord_t = torch.tensor(self.coord_np, dtype=torch.float64, device=device)
        atype_t = torch.tensor(self.atype_np, dtype=torch.int64, device=device)
        box_t = torch.tensor(self.box_np, dtype=torch.float64, device=device)
        outs = VesinNeighborList().build(coord_t, atype_t, box_t, self.rcut, self.sel)
        for t in outs:
            self.assertEqual(t.device.type, device.type)

    def test_isinstance(self) -> None:
        self.assertIsInstance(VesinNeighborList(), NeighborList)
        self.assertIsInstance(DefaultNeighborList(), NeighborList)


@unittest.skipIf(not is_vesin_torch_available(), "vesin.torch is not installed")
class TestNeighborListModelEquivalence(unittest.TestCase):
    """Full model outputs must be invariant to the neighbor-list strategy."""

    def setUp(self) -> None:
        self.coord_np, self.atype_np, self.box_np = _system()

    def _tol(self, model_dict) -> dict:
        prec = str(model_dict["descriptor"].get("precision", "float64"))
        if "float32" in prec or "32" in prec:
            return {"rtol": 1e-5, "atol": 1e-5}
        return {"rtol": 1e-9, "atol": 1e-9}

    def _run_dpmodel(self, name, model_dict, box_np) -> None:
        md = get_model_dp(copy.deepcopy(model_dict))
        tol = self._tol(model_dict)
        r0 = md.call(
            self.coord_np,
            self.atype_np,
            box=box_np,
            neighbor_list=DefaultNeighborList(),
        )
        r1 = md.call(
            self.coord_np, self.atype_np, box=box_np, neighbor_list=VesinNeighborList()
        )
        for k in ("energy", "atom_energy"):
            np.testing.assert_allclose(
                r0[k], r1[k], err_msg=f"dpmodel {name} {k}", **tol
            )

    def _run_pt_expt(self, name, model_dict, box_np) -> None:
        md = get_model(copy.deepcopy(model_dict))
        md.eval()
        tol = self._tol(model_dict)
        box_t = None if box_np is None else torch.tensor(box_np, dtype=torch.float64)
        atype_t = torch.tensor(self.atype_np, dtype=torch.int64)
        results = {}
        for tag, nl in (("def", DefaultNeighborList()), ("ves", VesinNeighborList())):
            coord_t = torch.tensor(self.coord_np, dtype=torch.float64).requires_grad_(
                True
            )
            results[tag] = md.forward(
                coord_t, atype_t, box=box_t, do_atomic_virial=True, neighbor_list=nl
            )
        for k in ("energy", "atom_energy", "force", "virial", "atom_virial"):
            a, b = results["def"].get(k), results["ves"].get(k)
            if a is None or b is None:
                continue
            np.testing.assert_allclose(
                a.detach().cpu().numpy(),
                b.detach().cpu().numpy(),
                err_msg=f"pt_expt {name} {k}",
                **tol,
            )

    def _run_default_fallback(self, name, model_dict, box_np) -> None:
        """``neighbor_list=None`` must equal an explicit DefaultNeighborList."""
        md = get_model(copy.deepcopy(model_dict))
        md.eval()
        box_t = None if box_np is None else torch.tensor(box_np, dtype=torch.float64)
        atype_t = torch.tensor(self.atype_np, dtype=torch.int64)
        outs = {}
        for tag, kw in (
            ("none", {}),
            ("explicit", {"neighbor_list": DefaultNeighborList()}),
        ):
            coord_t = torch.tensor(self.coord_np, dtype=torch.float64).requires_grad_(
                True
            )
            outs[tag] = md.forward(
                coord_t, atype_t, box=box_t, do_atomic_virial=True, **kw
            )
        for k in ("energy", "force", "virial"):
            np.testing.assert_array_equal(
                outs["none"][k].detach().cpu().numpy(),
                outs["explicit"][k].detach().cpu().numpy(),
                err_msg=f"default fallback {name} {k}",
            )


def _make_dpmodel_test(name, model_dict, periodic):
    def test(self) -> None:
        box = self.box_np if periodic else None
        self._run_dpmodel(name, model_dict, box)

    return test


def _make_pt_expt_test(name, model_dict, periodic):
    def test(self) -> None:
        box = self.box_np if periodic else None
        self._run_pt_expt(name, model_dict, box)

    return test


def _make_fallback_test(name, model_dict):
    def test(self) -> None:
        self._run_default_fallback(name, model_dict, self.box_np)

    return test


# generate one test per (family, pbc/nopbc) so failures pinpoint the family
for _name, _dict in ALL_MODELS.items():
    for _pbc, _suffix in ((True, "pbc"), (False, "nopbc")):
        setattr(
            TestNeighborListModelEquivalence,
            f"test_dpmodel_{_name}_{_suffix}",
            _make_dpmodel_test(_name, _dict, _pbc),
        )
        setattr(
            TestNeighborListModelEquivalence,
            f"test_pt_expt_{_name}_{_suffix}",
            _make_pt_expt_test(_name, _dict, _pbc),
        )
    setattr(
        TestNeighborListModelEquivalence,
        f"test_default_fallback_{_name}",
        _make_fallback_test(_name, _dict),
    )


if __name__ == "__main__":
    unittest.main()
