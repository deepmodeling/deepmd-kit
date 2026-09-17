# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fitting-side vacuum reference.

With ``vacuum_ref`` the output of an atom is
``bias(t_i) + f(x_i; c_i) - f(x_vac(t_i); c_i)``, where ``x_vac`` is the
descriptor of an isolated atom of every type and ``c_i`` the conditioning of
the atom itself. An atom whose descriptor equals its vacuum descriptor thus
contributes exactly the bias of its type.
"""

import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.fitting import InvarFitting as DPInvarFitting
from deepmd.pt.model.descriptor.sezm_nn.dens import (
    SeZMDeNSFittingNet,
)
from deepmd.pt.model.task.invar_fitting import (
    InvarFitting,
)
from deepmd.pt.model.task.sezm_ener import (
    SeZMEnergyFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from ...seed import (
    GLOBAL_SEED,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
NTYPES, ND, NF, NLOC = 3, 8, 2, 5
CONDITIONING = [(0, 0), (2, 0), (0, 1), (2, 1)]


class VacuumRefInputs(unittest.TestCase):
    """Random descriptors, a random vacuum table and random conditioning."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(GLOBAL_SEED)
        self.descriptor = self.tensor(self.rng.normal(size=(NF, NLOC, ND)))
        self.vacuum = self.tensor(self.rng.normal(size=(NTYPES, ND)))
        atype = self.rng.integers(0, NTYPES, size=(NF, NLOC))
        atype[0, :NTYPES] = np.arange(NTYPES)
        self.atype = torch.tensor(atype, dtype=torch.long, device=env.DEVICE)
        self.bias = self.rng.normal(size=(NTYPES, 1))

    def tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=dtype, device=env.DEVICE)

    def conditioning(
        self, nfp: int, nap: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        fparam = self.tensor(self.rng.normal(size=(NF, nfp))) if nfp else None
        aparam = self.tensor(self.rng.normal(size=(NF, NLOC, nap))) if nap else None
        return fparam, aparam

    def expected_bias(self) -> np.ndarray:
        return self.bias[to_numpy_array(self.atype)]

    def assert_reference_subtraction(
        self,
        ft_ref: torch.nn.Module,
        ft_vac: torch.nn.Module,
        fparam: torch.Tensor | None,
        aparam: torch.Tensor | None,
    ) -> None:
        """``ft_vac`` equals ``ft_ref`` minus ``ft_ref`` on the vacuum rows plus the bias."""
        out = ft_vac(
            self.descriptor,
            self.atype,
            fparam=fparam,
            aparam=aparam,
            vacuum_descriptor=self.vacuum,
        )["energy"]
        ref = ft_ref(self.descriptor, self.atype, fparam=fparam, aparam=aparam)
        ref_vac = ft_ref(
            self.vacuum[self.atype], self.atype, fparam=fparam, aparam=aparam
        )
        expected = (
            to_numpy_array(ref["energy"])
            - to_numpy_array(ref_vac["energy"])
            + self.expected_bias()
        )
        np.testing.assert_allclose(
            to_numpy_array(out), expected, rtol=1e-10, atol=1e-10
        )


class TestInvarFittingVacuumRef(VacuumRefInputs):
    def build(self, vacuum_ref: bool, **kwargs) -> InvarFitting:
        ft = InvarFitting(
            "energy",
            NTYPES,
            ND,
            1,
            neuron=[6, 6],
            bias_atom_e=self.bias,
            vacuum_ref=vacuum_ref,
            seed=GLOBAL_SEED,
            **kwargs,
        ).to(env.DEVICE)
        if ft.dim_case_embd > 0:
            ft.set_case_embd(1)
        return ft

    def test_isolated_atom_gives_bias(self) -> None:
        for mixed_types, (nfp, nap), ncase, mask in itertools.product(
            [True, False], CONDITIONING, [0, 2], [False, True]
        ):
            ft = self.build(
                True,
                mixed_types=mixed_types,
                numb_fparam=nfp,
                numb_aparam=nap,
                dim_case_embd=ncase,
                use_aparam_as_mask=mask,
            )
            fparam, aparam = self.conditioning(nfp, nap)
            out = ft(
                self.vacuum[self.atype],
                self.atype,
                fparam=fparam,
                aparam=aparam,
                vacuum_descriptor=self.vacuum,
            )["energy"]
            np.testing.assert_allclose(
                to_numpy_array(out), self.expected_bias(), rtol=1e-10, atol=1e-10
            )

    def test_matches_reference_subtraction(self) -> None:
        for mixed_types, (nfp, nap), ncase in itertools.product(
            [True, False], CONDITIONING, [0, 2]
        ):
            ft_ref = self.build(
                False,
                mixed_types=mixed_types,
                numb_fparam=nfp,
                numb_aparam=nap,
                dim_case_embd=ncase,
            )
            ft_vac = InvarFitting.deserialize(
                {**ft_ref.serialize(), "vacuum_ref": True}
            ).to(env.DEVICE)
            self.assertTrue(ft_vac.vacuum_ref)
            self.assert_reference_subtraction(
                ft_ref, ft_vac, *self.conditioning(nfp, nap)
            )

    def test_dpmodel_consistency(self) -> None:
        for mixed_types, (nfp, nap), ncase in itertools.product(
            [True, False], CONDITIONING, [0, 2]
        ):
            ft = self.build(
                True,
                mixed_types=mixed_types,
                numb_fparam=nfp,
                numb_aparam=nap,
                dim_case_embd=ncase,
            )
            ft_dp = DPInvarFitting.deserialize(ft.serialize())
            fparam, aparam = self.conditioning(nfp, nap)
            out = ft(
                self.descriptor,
                self.atype,
                fparam=fparam,
                aparam=aparam,
                vacuum_descriptor=self.vacuum,
            )["energy"]
            out_dp = ft_dp(
                to_numpy_array(self.descriptor),
                to_numpy_array(self.atype),
                fparam=to_numpy_array(fparam),
                aparam=to_numpy_array(aparam),
                vacuum_descriptor=to_numpy_array(self.vacuum),
            )["energy"]
            np.testing.assert_allclose(
                to_numpy_array(out), out_dp, rtol=1e-12, atol=1e-12
            )

    def test_fold_vacuum_reference(self) -> None:
        for mixed_types, ncase in itertools.product([True, False], [0, 2]):
            ft = self.build(True, mixed_types=mixed_types, dim_case_embd=ncase)
            expected = ft(self.descriptor, self.atype, vacuum_descriptor=self.vacuum)
            ft.fold_vacuum_reference(self.vacuum)
            self.assertFalse(ft.vacuum_ref)
            out = ft(self.descriptor, self.atype)["energy"]
            np.testing.assert_allclose(
                to_numpy_array(out),
                to_numpy_array(expected["energy"]),
                rtol=1e-10,
                atol=1e-10,
            )
        # a conditioned fitting stores the table and references from it; the
        # table is a deployment constant that checkpoints leave out and a
        # type-map change drops
        type_map = ["O", "H", "B"]
        ft = self.build(True, numb_fparam=2, type_map=type_map)
        fparam = self.tensor(self.rng.normal(size=(NF, 2)))
        expected = ft(
            self.descriptor, self.atype, fparam=fparam, vacuum_descriptor=self.vacuum
        )
        ft.fold_vacuum_reference(self.vacuum)
        self.assertTrue(ft.vacuum_ref)
        self.assertFalse(ft.needs_vacuum_descriptor())
        out = ft(self.descriptor, self.atype, fparam=fparam)["energy"]
        np.testing.assert_allclose(
            to_numpy_array(out),
            to_numpy_array(expected["energy"]),
            rtol=1e-10,
            atol=1e-10,
        )
        state = ft.state_dict()
        self.assertNotIn("vacuum_table", state)
        fresh = self.build(True, numb_fparam=2, type_map=type_map)
        fresh.load_state_dict(state)
        self.assertTrue(fresh.needs_vacuum_descriptor())
        ft.change_type_map(["B", "O", "H"])
        self.assertTrue(ft.needs_vacuum_descriptor())

    def test_default_fparam(self) -> None:
        ft = self.build(True, numb_fparam=1, default_fparam=[0.3])
        explicit = ft(
            self.descriptor,
            self.atype,
            fparam=self.tensor(np.full((NF, 1), 0.3)),
            vacuum_descriptor=self.vacuum,
        )
        default = ft(self.descriptor, self.atype, vacuum_descriptor=self.vacuum)
        np.testing.assert_allclose(
            to_numpy_array(default["energy"]),
            to_numpy_array(explicit["energy"]),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_jit(self) -> None:
        for mixed_types, (nfp, nap) in itertools.product([True, False], CONDITIONING):
            ft = self.build(
                True, mixed_types=mixed_types, numb_fparam=nfp, numb_aparam=nap
            )
            torch.jit.script(ft)

    def test_vacuum_descriptor_required(self) -> None:
        ft = self.build(True)
        with self.assertRaises(ValueError):
            ft(self.descriptor, self.atype)
        with self.assertRaises(ValueError):
            ft(self.descriptor, self.atype, vacuum_descriptor=self.vacuum[:, :-1])
        # the table is ignored without the option
        ft_ref = self.build(False)
        out = ft_ref(self.descriptor, self.atype, vacuum_descriptor=self.vacuum)
        np.testing.assert_allclose(
            to_numpy_array(out["energy"]),
            to_numpy_array(ft_ref(self.descriptor, self.atype)["energy"]),
        )

    def test_serialization(self) -> None:
        data = self.build(True).serialize()
        self.assertTrue(data["vacuum_ref"])
        self.assertTrue(InvarFitting.deserialize(data).vacuum_ref)
        self.assertFalse(
            InvarFitting.deserialize({**data, "vacuum_ref": False}).vacuum_ref
        )

    def test_atom_ener_is_exclusive(self) -> None:
        self.assertTrue(self.build(True, atom_ener=[None] * NTYPES).vacuum_ref)
        with self.assertRaises(ValueError):
            self.build(True, atom_ener=[1.0] + [None] * (NTYPES - 1))


class TestSeZMFittingVacuumRef(VacuumRefInputs):
    def build(self, vacuum_ref: bool, case_film_embd: bool) -> SeZMEnergyFittingNet:
        ft = SeZMEnergyFittingNet(
            NTYPES,
            ND,
            neuron=[16],
            bias_atom_e=self.bias,
            numb_fparam=2,
            dim_case_embd=2,
            case_film_embd=case_film_embd,
            precision="float64",
            vacuum_ref=vacuum_ref,
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        ft.set_case_embd(1)
        return ft

    def test_isolated_atom_gives_bias(self) -> None:
        for case_film_embd in [False, True]:
            ft = self.build(True, case_film_embd)
            fparam, _ = self.conditioning(2, 0)
            out = ft(
                self.vacuum[self.atype],
                self.atype,
                fparam=fparam,
                vacuum_descriptor=self.vacuum,
            )["energy"]
            np.testing.assert_allclose(
                to_numpy_array(out), self.expected_bias(), rtol=1e-10, atol=1e-10
            )

    def test_matches_reference_subtraction(self) -> None:
        for case_film_embd in [False, True]:
            ft_ref = self.build(False, case_film_embd)
            ft_vac = SeZMEnergyFittingNet.deserialize(
                {**ft_ref.serialize(), "vacuum_ref": True}
            ).to(env.DEVICE)
            self.assertTrue(ft_vac.vacuum_ref)
            self.assert_reference_subtraction(ft_ref, ft_vac, *self.conditioning(2, 0))

    def test_atom_ener_subtracts_the_zero_descriptor_output(self) -> None:
        """``atom_ener`` removes the output of a zero descriptor, also under case FiLM."""
        for case_film_embd in [False, True]:
            ft_ref = self.build(False, case_film_embd)
            ft = SeZMEnergyFittingNet.deserialize(
                {**ft_ref.serialize(), "atom_ener": [1.0] * NTYPES}
            ).to(env.DEVICE)
            fparam, _ = self.conditioning(2, 0)
            zeros = torch.zeros_like(self.descriptor)
            # an atom with a zero descriptor contributes exactly its bias
            out_zero = ft(zeros, self.atype, fparam=fparam)["energy"]
            np.testing.assert_allclose(
                to_numpy_array(out_zero), self.expected_bias(), rtol=1e-10, atol=1e-10
            )
            out = ft(self.descriptor, self.atype, fparam=fparam)["energy"]
            ref = ft_ref(self.descriptor, self.atype, fparam=fparam)["energy"]
            ref_zero = ft_ref(zeros, self.atype, fparam=fparam)["energy"]
            expected = (
                to_numpy_array(ref) - to_numpy_array(ref_zero) + self.expected_bias()
            )
            np.testing.assert_allclose(
                to_numpy_array(out), expected, rtol=1e-10, atol=1e-10
            )

    def test_fold_vacuum_reference(self) -> None:
        for case_film_embd in [False, True]:
            ft = SeZMEnergyFittingNet(
                NTYPES,
                ND,
                neuron=[16],
                bias_atom_e=self.bias,
                dim_case_embd=2,
                case_film_embd=case_film_embd,
                precision="float64",
                vacuum_ref=True,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)
            ft.set_case_embd(1)
            expected = ft(self.descriptor, self.atype, vacuum_descriptor=self.vacuum)
            ft.fold_vacuum_reference(self.vacuum)
            self.assertFalse(ft.vacuum_ref)
            out = ft(self.descriptor, self.atype)["energy"]
            np.testing.assert_allclose(
                to_numpy_array(out),
                to_numpy_array(expected["energy"]),
                rtol=1e-10,
                atol=1e-10,
            )


class TestDeNSFittingVacuumRef(VacuumRefInputs):
    def test_energy_head_is_referenced(self) -> None:
        channels, lmax = 4, 1
        ft = SeZMDeNSFittingNet(
            ntypes=NTYPES,
            dim_descrpt=ND,
            condition_lmax=lmax,
            latent_lmax=lmax,
            channels=channels,
            neuron=[8],
            bias_atom_e=self.bias,
            precision="float64",
            vacuum_ref=True,
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        self.assertTrue(ft.energy_head.vacuum_ref)
        latent = self.tensor(
            self.rng.normal(size=(NF * NLOC, (lmax + 1) ** 2, 1, channels))
        )
        out = ft(
            self.vacuum[self.atype],
            latent,
            self.atype,
            vacuum_descriptor=self.vacuum,
        )["energy"]
        np.testing.assert_allclose(
            to_numpy_array(out), self.expected_bias(), rtol=1e-10, atol=1e-10
        )
        data = ft.serialize()
        self.assertTrue(data["config"]["vacuum_ref"])
        self.assertTrue(SeZMDeNSFittingNet.deserialize(data).vacuum_ref)


if __name__ == "__main__":
    unittest.main()
