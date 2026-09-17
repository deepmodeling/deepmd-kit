# SPDX-License-Identifier: LGPL-3.0-or-later
"""Isolated-atom reference of a SeZM model on the PyTorch edge route.

The SeZM descriptor carries one reference atom per type through the same
forward as the real atoms, so an isolated atom contributes exactly its bias
and any cluster differs from the unreferenced model by the isolated-atom
network output of its atoms.
"""

import copy
import unittest

import numpy as np
import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)

TYPE_MAP = ["O", "H"]
BIAS = np.array([[-3.0], [0.5]])
PRESET = {"energy": {"O": float(BIAS[0, 0]), "H": float(BIAS[1, 0])}}
MODEL_PARAMS = {
    "type": "SeZM",
    "type_map": TYPE_MAP,
    "preset_out_bias": PRESET,
    "descriptor": {
        "type": "SeZM",
        "sel": [4, 4],
        "rcut": 4.0,
        "channels": 8,
        "n_focus": 1,
        "n_radial": 4,
        "radial_mlp": [8],
        "use_env_seed": True,
        "l_schedule": [1, 0],
        "mmax": 1,
        "so2_norm": False,
        "so2_layers": 1,
        "n_atten_head": 1,
        "sandwich_norm": [True, False, True, False],
        "ffn_neurons": 8,
        "ffn_blocks": 1,
        "s2_activation": [False, True],
        "mlp_bias": False,
        "layer_scale": False,
        "use_amp": False,
        "activation_function": "silu",
        "glu_activation": True,
        "precision": "float64",
        "seed": 7,
    },
    "fitting_net": {
        "neuron": [8],
        "activation_function": "silu",
        "precision": "float64",
        "seed": 7,
    },
}


class TestSeZMVacuumRef(unittest.TestCase):
    def make_model(self, vacuum_ref: bool, preset: bool = True) -> torch.nn.Module:
        params = copy.deepcopy(MODEL_PARAMS)
        if not preset:
            params.pop("preset_out_bias")
        params["fitting_net"]["vacuum_ref"] = vacuum_ref
        model = get_model(params).to(env.DEVICE)
        fitting = model.atomic_model.fitting_net
        with torch.no_grad():
            fitting.bias_atom_e.copy_(
                torch.as_tensor(
                    BIAS,
                    dtype=fitting.bias_atom_e.dtype,
                    device=fitting.bias_atom_e.device,
                )
            )
        return model.eval()

    def predict(
        self, model: torch.nn.Module, coord: np.ndarray, atype: np.ndarray
    ) -> dict[str, torch.Tensor]:
        return model(
            torch.as_tensor(coord, dtype=torch.float64, device=env.DEVICE),
            torch.as_tensor(atype, dtype=torch.long, device=env.DEVICE),
            None,
        )

    def atom_energies(
        self, model: torch.nn.Module, coord: np.ndarray, atype: np.ndarray
    ) -> np.ndarray:
        return (
            self.predict(model, coord, atype)["atom_energy"][..., 0]
            .detach()
            .cpu()
            .numpy()
        )

    def test_isolated_atoms_and_reference_subtraction(self) -> None:
        rng = np.random.default_rng(0)
        coord = rng.normal(size=(2, 6, 3)) * 1.2
        atype = np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]])
        iso_coord = np.zeros((2, 1, 3))
        iso_atype = np.array([[0], [1]])
        vac = self.make_model(True)
        ref = self.make_model(False)

        # an isolated atom of every type contributes exactly its bias
        np.testing.assert_allclose(
            self.atom_energies(vac, iso_coord, iso_atype),
            BIAS[:, 0][:, None],
            rtol=1e-10,
            atol=1e-10,
        )
        # the reference removes the isolated-atom network output of every atom
        e_vac = self.atom_energies(vac, coord, atype)
        e_ref = self.atom_energies(ref, coord, atype)
        iso_ref = self.atom_energies(ref, iso_coord, iso_atype)[:, 0]
        expected = e_ref - (iso_ref - BIAS[:, 0])[atype]
        np.testing.assert_allclose(e_vac, expected, rtol=1e-10, atol=1e-10)

    def test_without_preset_the_output_is_not_referenced(self) -> None:
        """A bias fitted from the data is no isolated-atom energy, so the output stays plain."""
        rng = np.random.default_rng(1)
        coord = rng.normal(size=(2, 6, 3)) * 1.2
        atype = np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]])
        plain = self.make_model(False)
        unreferenced = self.make_model(True, preset=False)
        fitting = unreferenced.atomic_model.fitting_net
        self.assertFalse(fitting.vacuum_ref)
        self.assertFalse(fitting.needs_vacuum_descriptor())
        np.testing.assert_allclose(
            self.atom_energies(unreferenced, coord, atype),
            self.atom_energies(plain, coord, atype),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_shared_fitting_references_only_the_branch_with_a_preset(self) -> None:
        """Branches sharing one fitting reference their output only where a preset fixes the bias."""
        from deepmd.pt.train.training import (
            get_model_for_wrapper,
            prepare_model_for_loss,
        )
        from deepmd.pt.train.wrapper import (
            ModelWrapper,
        )
        from deepmd.pt.utils.multi_task import (
            preprocess_shared_params,
        )

        branch = {
            "type": "SeZM",
            "type_map": "type_map",
            "descriptor": "descriptor",
            "fitting_net": "fitting",
        }
        config = {
            "shared_dict": {
                "type_map": TYPE_MAP,
                "descriptor": copy.deepcopy(MODEL_PARAMS["descriptor"]),
                "fitting": {
                    **MODEL_PARAMS["fitting_net"],
                    "vacuum_ref": True,
                    "dim_case_embd": 2,
                },
            },
            "model_dict": {
                "with_table": {**branch, "preset_out_bias": PRESET},
                "without_table": dict(branch),
            },
        }
        config, shared_links = preprocess_shared_params(config)
        models = get_model_for_wrapper(config)
        prepare_model_for_loss(models, {key: {"type": "ener"} for key in models})
        wrapper = ModelWrapper(models)
        wrapper.share_params(shared_links, dict.fromkeys(models, 0.5))
        referenced = wrapper.model["with_table"].to(env.DEVICE).eval()
        unreferenced = wrapper.model["without_table"].to(env.DEVICE).eval()
        fit_a = referenced.atomic_model.fitting_net
        fit_b = unreferenced.atomic_model.fitting_net
        # one network, one decision per branch
        self.assertIs(fit_a.filter_layers, fit_b.filter_layers)
        self.assertTrue(fit_a.vacuum_ref)
        self.assertFalse(fit_b.vacuum_ref)
        for fitting in (fit_a, fit_b):
            with torch.no_grad():
                fitting.bias_atom_e.copy_(
                    torch.as_tensor(
                        BIAS,
                        dtype=fitting.bias_atom_e.dtype,
                        device=fitting.bias_atom_e.device,
                    )
                )

        rng = np.random.default_rng(5)
        coord = rng.normal(size=(2, 6, 3)) * 1.2
        atype = np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]])
        iso_coord = np.zeros((2, 1, 3))
        iso_atype = np.array([[0], [1]])
        # the branch with a table pins its isolated atoms to the preset
        np.testing.assert_allclose(
            self.atom_energies(referenced, iso_coord, iso_atype),
            BIAS[:, 0][:, None],
            rtol=1e-10,
            atol=1e-10,
        )
        # the branch without a table is the plain model on the shared weights
        params = copy.deepcopy(MODEL_PARAMS)
        params.pop("preset_out_bias")
        params["fitting_net"]["dim_case_embd"] = 2
        plain = get_model(params).to(env.DEVICE)
        plain.load_state_dict(unreferenced.state_dict())
        plain.eval()
        np.testing.assert_allclose(
            self.atom_energies(unreferenced, coord, atype),
            self.atom_energies(plain, coord, atype),
            rtol=1e-12,
            atol=1e-12,
        )
        iso_energy = self.atom_energies(unreferenced, iso_coord, iso_atype)
        self.assertGreater(np.abs(iso_energy - BIAS[:, 0][:, None]).max(), 1e-6)

    def test_forces_are_unchanged(self) -> None:
        """The reference is independent of the coordinates, so forces do not move."""
        rng = np.random.default_rng(3)
        coord = rng.normal(size=(2, 6, 3)) * 1.2
        atype = np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]])
        force_vac = self.predict(self.make_model(True), coord, atype)["force"]
        force_ref = self.predict(self.make_model(False), coord, atype)["force"]
        np.testing.assert_allclose(
            force_vac.detach().cpu().numpy(),
            force_ref.detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_fold_reproduces_the_referenced_model(self) -> None:
        """Folding the reference into the bias keeps every energy and drops the rows."""
        rng = np.random.default_rng(1)
        coord = rng.normal(size=(2, 6, 3)) * 1.2
        atype = np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]])
        iso_coord = np.zeros((2, 1, 3))
        iso_atype = np.array([[0], [1]])
        model = self.make_model(True)
        e_cluster = self.atom_energies(model, coord, atype)
        e_iso = self.atom_energies(model, iso_coord, iso_atype)

        model.fold_vacuum_reference()
        self.assertFalse(model.atomic_model.fitting_net.vacuum_ref)
        np.testing.assert_allclose(
            self.atom_energies(model, coord, atype), e_cluster, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            self.atom_energies(model, iso_coord, iso_atype),
            e_iso,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_fold_drops_the_compiled_graphs(self) -> None:
        from deepmd.pt.model.model.sezm_model import (
            _sezm_structure_key,
        )

        model = self.make_model(True)
        key_live = _sezm_structure_key(model)
        model.compiled_core_compute_cache[(False, False)] = object()
        model.fold_vacuum_reference()
        self.assertEqual(model.compiled_core_compute_cache, {})
        # a folded model traces a graph without reference nodes
        self.assertNotEqual(_sezm_structure_key(model), key_live)


class TestSeZMNativeSpinVacuumRef(unittest.TestCase):
    """The reference atom carries the ground-state charge/spin condition and spin."""

    def make_model(self, vacuum_ref: bool) -> torch.nn.Module:
        params = copy.deepcopy(MODEL_PARAMS)
        params["descriptor"]["add_chg_spin_ebd"] = True
        params["descriptor"]["default_chg_spin"] = [0, 1]
        params["fitting_net"]["vacuum_ref"] = vacuum_ref
        params["spin"] = {"use_spin": [True, False], "scheme": "native"}
        model = get_model(params).to(env.DEVICE)
        fitting = model.atomic_model.fitting_net
        with torch.no_grad():
            fitting.bias_atom_e.copy_(
                torch.as_tensor(
                    BIAS,
                    dtype=fitting.bias_atom_e.dtype,
                    device=fitting.bias_atom_e.device,
                )
            )
        return model.eval()

    def atom_energies(
        self,
        model: torch.nn.Module,
        coord: np.ndarray,
        atype: np.ndarray,
        spin: np.ndarray,
        charge_spin: np.ndarray,
    ) -> np.ndarray:
        def tensor(
            array: np.ndarray, dtype: torch.dtype = torch.float64
        ) -> torch.Tensor:
            return torch.as_tensor(array, dtype=dtype, device=env.DEVICE)

        ret = model(
            tensor(coord),
            tensor(atype, torch.long),
            tensor(spin),
            None,
            charge_spin=tensor(charge_spin),
        )
        return ret["atom_energy"][..., 0].detach().cpu().numpy()

    def test_isolated_atoms_and_reference_subtraction(self) -> None:
        from deepmd.utils.vacuum_reference import (
            reference_charge_spin,
            reference_spin,
        )

        rng = np.random.default_rng(0)
        coord = rng.normal(size=(2, 6, 3)) * 1.2
        atype = np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]])
        spin = rng.normal(size=(2, 6, 3)) * (atype == 0)[..., None]
        charge_spin = np.array([[0.0, 1.0], [1.0, 2.0]])
        iso_coord = np.zeros((2, 1, 3))
        iso_atype = np.array([[0], [1]])
        iso_spin = reference_spin(TYPE_MAP)[:, None, :]
        iso_charge_spin = reference_charge_spin(TYPE_MAP)
        vac = self.make_model(True)
        ref = self.make_model(False)

        # an isolated neutral ground-state atom contributes exactly its bias
        np.testing.assert_allclose(
            self.atom_energies(vac, iso_coord, iso_atype, iso_spin, iso_charge_spin),
            BIAS[:, 0][:, None],
            rtol=1e-10,
            atol=1e-10,
        )
        # the reference removes the isolated-atom network output of every atom
        e_vac = self.atom_energies(vac, coord, atype, spin, charge_spin)
        e_ref = self.atom_energies(ref, coord, atype, spin, charge_spin)
        iso_ref = self.atom_energies(
            ref, iso_coord, iso_atype, iso_spin, iso_charge_spin
        )[:, 0]
        expected = e_ref - (iso_ref - BIAS[:, 0])[atype]
        np.testing.assert_allclose(e_vac, expected, rtol=1e-10, atol=1e-10)

        # the fold evaluates the reference under the ground-state conditions
        vac.fold_vacuum_reference()
        self.assertFalse(vac.atomic_model.fitting_net.vacuum_ref)
        np.testing.assert_allclose(
            self.atom_energies(vac, coord, atype, spin, charge_spin),
            e_vac,
            rtol=1e-10,
            atol=1e-10,
        )


class TestSeZMDeNSVacuumRef(unittest.TestCase):
    """The DeNS energy head references every atom to the isolated atom without force input."""

    def make_model(self, vacuum_ref: bool) -> torch.nn.Module:
        params = copy.deepcopy(MODEL_PARAMS)
        # the DeNS vector heads need an l=1 latent
        params["descriptor"]["l_schedule"] = [1, 1]
        params["fitting_net"]["vacuum_ref"] = vacuum_ref
        model = get_model(params).to(env.DEVICE)
        model.set_active_mode("dens")
        head = model.atomic_model.get_dens_fitting_net().energy_head
        self.assertEqual(head.vacuum_ref, vacuum_ref)
        with torch.no_grad():
            head.bias_atom_e.copy_(
                torch.as_tensor(
                    BIAS, dtype=head.bias_atom_e.dtype, device=head.bias_atom_e.device
                )
            )
        return model.eval()

    def test_without_preset_the_head_is_not_referenced(self) -> None:
        """The DeNS energy head follows the preset of the branch like the energy fitting."""
        params = copy.deepcopy(MODEL_PARAMS)
        params.pop("preset_out_bias")
        params["descriptor"]["l_schedule"] = [1, 1]
        params["fitting_net"]["vacuum_ref"] = True
        model = get_model(params)
        model.set_active_mode("dens")
        dens = model.atomic_model.get_dens_fitting_net()
        self.assertFalse(dens.vacuum_ref)
        self.assertFalse(dens.energy_head.vacuum_ref)
        self.assertFalse(dens.needs_vacuum_descriptor())
        self.assertFalse(model.atomic_model.fitting_net.vacuum_ref)

    def atom_energies(
        self,
        model: torch.nn.Module,
        coord: np.ndarray,
        atype: np.ndarray,
        force: np.ndarray,
        noise_mask: np.ndarray,
    ) -> np.ndarray:
        def tensor(
            array: np.ndarray, dtype: torch.dtype = torch.float64
        ) -> torch.Tensor:
            return torch.as_tensor(array, dtype=dtype, device=env.DEVICE)

        nf = coord.shape[0]
        box = np.tile(np.eye(3).reshape(1, 9) * 20.0, (nf, 1))
        ret = model(
            tensor(coord),
            tensor(atype, torch.long),
            box=tensor(box),
            force_input=tensor(force),
            noise_mask=tensor(noise_mask, torch.bool),
        )
        return ret["atom_energy"][..., 0].detach().cpu().numpy()

    def test_isolated_atoms_and_reference_subtraction(self) -> None:
        rng = np.random.default_rng(0)
        coord = rng.normal(size=(2, 6, 3)) * 1.2
        atype = np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]])
        force = rng.normal(size=(2, 6, 3)) * 0.3
        noise_mask = rng.integers(0, 2, size=(2, 6)).astype(bool)
        iso_coord = np.zeros((2, 1, 3))
        iso_atype = np.array([[0], [1]])
        iso_force = np.zeros((2, 1, 3))
        iso_mask = np.zeros((2, 1), dtype=bool)
        vac = self.make_model(True)
        ref = self.make_model(False)

        # an isolated atom without force input contributes exactly its bias
        np.testing.assert_allclose(
            self.atom_energies(vac, iso_coord, iso_atype, iso_force, iso_mask),
            BIAS[:, 0][:, None],
            rtol=1e-10,
            atol=1e-10,
        )
        # the reference removes the isolated-atom network output of every atom
        e_vac = self.atom_energies(vac, coord, atype, force, noise_mask)
        e_ref = self.atom_energies(ref, coord, atype, force, noise_mask)
        iso_ref = self.atom_energies(ref, iso_coord, iso_atype, iso_force, iso_mask)[
            :, 0
        ]
        expected = e_ref - (iso_ref - BIAS[:, 0])[atype]
        np.testing.assert_allclose(e_vac, expected, rtol=1e-10, atol=1e-10)


def test_isolated_atom_under_amp_and_fused_training_kernels(monkeypatch) -> None:
    """The identity holds in the mixed-precision fused training forward.

    The reference atom of every type is carried through the same fused
    kernels as the real atoms, so an isolated atom and its reference receive
    the same descriptor up to the round-off of the float32 fitting.
    """
    import pytest

    from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv_train import (
        op_available as cuda_value_available,
    )

    if not torch.cuda.is_available() or not cuda_value_available():
        pytest.skip("the DPA4 CUDA training operators are unavailable")
    monkeypatch.setenv("DP_CUDA_TRAIN", "1")
    monkeypatch.setenv("DP_TRITON_TRAIN", "1")
    params = {
        "type": "dpa4",
        "type_map": TYPE_MAP,
        "preset_out_bias": PRESET,
        "descriptor": {
            "type": "dpa4",
            "sel": 20,
            "rcut": 4.0,
            "channels": 32,
            "n_radial": 8,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": 2,
            "use_amp": True,
            "precision": "float32",
            "seed": 7,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [16],
            "precision": "float32",
            "vacuum_ref": True,
            "seed": 7,
        },
    }
    model = get_model(params).to(env.DEVICE)
    fitting = model.atomic_model.fitting_net
    with torch.no_grad():
        fitting.bias_atom_e.copy_(
            torch.as_tensor(
                BIAS, dtype=fitting.bias_atom_e.dtype, device=fitting.bias_atom_e.device
            )
        )
    model.train()
    coord = torch.zeros((2, 1, 3), dtype=torch.float64, device=env.DEVICE)
    atype = torch.tensor([[0], [1]], dtype=torch.long, device=env.DEVICE)
    energy = model(coord, atype, None)["atom_energy"][..., 0].detach().cpu().numpy()
    np.testing.assert_allclose(energy, BIAS[:, 0][:, None], rtol=0.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
