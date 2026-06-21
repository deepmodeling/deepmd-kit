# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the SeZM / DPA-4 ``nvalchemi-toolkit`` wrapper.

The wrapper (:class:`deepmd.pt.nvalchemi.DPA4Wrapper`) drives a SeZM model
through its sparse-edge lower interface from an ``nvalchemi`` graph batch. The
gold-standard correctness check is parity with the model's own neighbour-list
``forward``: feeding an identical structure through both paths must yield the
same energy, forces, and virial. These tests pin that parity (periodic,
non-periodic, and heterogeneous batched) plus the embedding and type-mapping
paths.

The whole suite is skipped when ``nvalchemi-toolkit`` is not installed.
"""

from __future__ import (
    annotations,
)

import contextlib
import unittest
from typing import (
    TYPE_CHECKING,
)

import torch

from deepmd.pt.model.model import (
    get_sezm_model,
)
from deepmd.pt.model.model.sezm_model import (
    ELEMENT_TO_Z,
)
from deepmd.pt.utils import (
    env,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )


@contextlib.contextmanager
def _clear_default_device() -> Iterator[None]:
    """Disable the pt-test ``cuda:9999999`` sentinel default device.

    ``source/tests/pt/__init__.py`` sets an invalid default device so tests
    that rely on implicit placement fail loudly. ``nvalchemi`` / ``tensordict``
    allocate unnamed tensors without an explicit device (both at import and at
    runtime), so this guard temporarily restores the real default. Matches the
    pattern in ``test_sezm_export.py``.
    """
    saved = torch.get_default_device()
    torch.set_default_device(None)
    try:
        yield
    finally:
        torch.set_default_device(saved)


try:
    with _clear_default_device():
        from nvalchemi.data import (
            AtomicData,
            Batch,
        )
        from nvalchemi.neighbors import (
            compute_neighbors,
        )

        from deepmd.pt.nvalchemi import (
            DPA4Wrapper,
        )

    NVALCHEMI_AVAILABLE = True
except ImportError:
    NVALCHEMI_AVAILABLE = False

TYPE_MAP = ["O", "H"]
RCUT = 4.0


class _ClearDefaultDeviceTestCase(unittest.TestCase):
    """Run a test class while the pt default-device sentinel is disabled."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._default_device_ctx = _clear_default_device()
        cls._default_device_ctx.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            super().tearDownClass()
        finally:
            ctx = getattr(cls, "_default_device_ctx", None)
            if ctx is not None:
                ctx.__exit__(None, None, None)
                delattr(cls, "_default_device_ctx")


@unittest.skipUnless(NVALCHEMI_AVAILABLE, "nvalchemi-toolkit is not installed")
class TestSeZMNVAlchemiWrapper(_ClearDefaultDeviceTestCase):
    """Parity of the nvalchemi wrapper against the native SeZM forward."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        self.model = self._build_model()

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    def _build_model(self) -> torch.nn.Module:
        """A tiny float64 SeZM model with randomized (non-trivial) weights."""
        params = {
            "type": "SeZM",
            "type_map": TYPE_MAP,
            "descriptor": {
                "type": "SeZM",
                "sel": [80, 80],
                "rcut": RCUT,
                "channels": 8,
                "n_focus": 1,
                "n_radial": 4,
                "radial_mlp": [8],
                "use_env_seed": True,
                "l_schedule": [2, 1],
                "mmax": 1,
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
            "use_compile": False,
        }
        model = get_sezm_model(params).to(self.device)
        torch.manual_seed(1234)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
        model.eval()
        return model

    def _system(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A periodic system with a tight cluster and a cross-boundary pair.

        Distances are kept either well below ``RCUT`` or well above it, so the
        native ghost-atom neighbour list and the nvalchemi COO list select the
        same edge set (the C3 envelope makes any near-cutoff edge negligible).
        """
        coord = torch.tensor(
            [
                [4.0, 4.0, 4.0],
                [4.85, 4.20, 4.10],
                [4.10, 4.90, 3.85],
                [3.80, 4.15, 4.80],
                [0.40, 1.20, 1.20],
                [8.75, 1.25, 1.15],
            ],
            dtype=torch.float64,
            device=self.device,
        )
        atype = torch.tensor([0, 1, 1, 1, 0, 1], dtype=torch.int64, device=self.device)
        box = torch.eye(3, dtype=torch.float64, device=self.device) * 9.0
        return coord, atype, box

    def _second_system(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A differently sized periodic system with a different cell."""
        coord = torch.tensor(
            [
                [2.0, 2.0, 2.0],
                [2.9, 2.1, 1.9],
                [2.1, 2.8, 2.2],
                [1.85, 1.95, 2.85],
            ],
            dtype=torch.float64,
            device=self.device,
        )
        atype = torch.tensor([0, 1, 1, 0], dtype=torch.int64, device=self.device)
        box = torch.eye(3, dtype=torch.float64, device=self.device) * 8.0
        return coord, atype, box

    def _atype_to_z(self, atype: torch.Tensor) -> torch.Tensor:
        z_of_type = torch.tensor(
            [ELEMENT_TO_Z[s] for s in TYPE_MAP],
            dtype=torch.long,
            device=atype.device,
        )
        return z_of_type.index_select(0, atype.long())

    def _native(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Energy / forces / virial from the native neighbour-list forward."""
        nloc = coord.shape[0]
        out = self.model(
            coord.view(1, nloc, 3),
            atype.view(1, nloc),
            box=None if box is None else box.reshape(1, 9),
            do_atomic_virial=True,
        )
        return {
            "energy": out["energy"].reshape(1).detach(),
            "forces": out["force"].reshape(nloc, 3).detach(),
            "virial": out["virial"].reshape(3, 3).detach(),
        }

    def _data(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None,
    ) -> AtomicData:
        fields = {
            "atomic_numbers": self._atype_to_z(atype),
            "positions": coord.clone(),
        }
        if box is not None:
            fields["cell"] = box.reshape(1, 3, 3)
            fields["pbc"] = torch.ones(1, 3, dtype=torch.bool, device=self.device)
        return AtomicData(**fields)

    def _wrapper_batch_out(
        self,
        wrapper: DPA4Wrapper,
        batch: Batch,
    ) -> dict[str, torch.Tensor]:
        compute_neighbors(batch, config=wrapper.model_config.neighbor_config)
        return wrapper(batch)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_parity_periodic(self) -> None:
        """Energy / forces / virial match native forward for a periodic cell."""
        coord, atype, box = self._system()
        wrapper = DPA4Wrapper(self.model, compute_stress=True)
        ref = self._native(coord, atype, box)

        batch = Batch.from_data_list(
            [self._data(coord, atype, box)], device=self.device
        )
        out = self._wrapper_batch_out(wrapper, batch)
        volume = torch.det(box).abs()

        torch.testing.assert_close(
            out["energy"].reshape(1), ref["energy"], atol=1e-7, rtol=1e-6
        )
        torch.testing.assert_close(
            out["forces"].reshape(-1, 3), ref["forces"], atol=1e-7, rtol=1e-6
        )
        torch.testing.assert_close(
            out["stress"].reshape(3, 3) * volume, ref["virial"], atol=1e-7, rtol=1e-6
        )

    def test_parity_nonperiodic(self) -> None:
        """Energy / forces match native forward for an open-boundary cluster."""
        coord, atype, _ = self._system()
        wrapper = DPA4Wrapper(self.model, compute_stress=False)
        ref = self._native(coord, atype, None)

        batch = Batch.from_data_list(
            [self._data(coord, atype, None)], device=self.device
        )
        out = self._wrapper_batch_out(wrapper, batch)

        torch.testing.assert_close(
            out["energy"].reshape(1), ref["energy"], atol=1e-7, rtol=1e-6
        )
        torch.testing.assert_close(
            out["forces"].reshape(-1, 3), ref["forces"], atol=1e-7, rtol=1e-6
        )
        self.assertNotIn("stress", out)

    def test_parity_batched_heterogeneous(self) -> None:
        """Per-graph outputs match native runs for a two-graph batch.

        The two graphs differ in size *and* cell, exercising the ``batch_idx``
        segment reduction and the global ``neighbor_list`` node offsets.
        """
        coord_a, atype_a, box_a = self._system()
        coord_b, atype_b, box_b = self._second_system()
        n_a = coord_a.shape[0]
        wrapper = DPA4Wrapper(self.model, compute_stress=True)

        ref_a = self._native(coord_a, atype_a, box_a)
        ref_b = self._native(coord_b, atype_b, box_b)

        batch = Batch.from_data_list(
            [self._data(coord_a, atype_a, box_a), self._data(coord_b, atype_b, box_b)],
            device=self.device,
        )
        out = self._wrapper_batch_out(wrapper, batch)
        vol_a = torch.det(box_a).abs()
        vol_b = torch.det(box_b).abs()

        torch.testing.assert_close(
            out["energy"][0], ref_a["energy"], atol=1e-7, rtol=1e-6
        )
        torch.testing.assert_close(
            out["energy"][1], ref_b["energy"], atol=1e-7, rtol=1e-6
        )
        torch.testing.assert_close(
            out["forces"][:n_a], ref_a["forces"], atol=1e-7, rtol=1e-6
        )
        torch.testing.assert_close(
            out["forces"][n_a:], ref_b["forces"], atol=1e-7, rtol=1e-6
        )
        torch.testing.assert_close(
            out["stress"][0] * vol_a, ref_a["virial"], atol=1e-7, rtol=1e-6
        )
        torch.testing.assert_close(
            out["stress"][1] * vol_b, ref_b["virial"], atol=1e-7, rtol=1e-6
        )

    def test_compute_embeddings_shapes(self) -> None:
        """Embeddings have the advertised per-atom / per-graph descriptor width."""
        coord, atype, box = self._system()
        wrapper = DPA4Wrapper(self.model)
        dim = wrapper.embedding_shapes["node_embeddings"][0]

        batch = Batch.from_data_list(
            [self._data(coord, atype, box)], device=self.device
        )
        compute_neighbors(batch, config=wrapper.model_config.neighbor_config)
        out = wrapper.compute_embeddings(batch)

        self.assertEqual(tuple(out.node_embeddings.shape), (coord.shape[0], dim))
        self.assertEqual(tuple(out.graph_embeddings.shape), (1, dim))
        self.assertTrue(torch.isfinite(out.node_embeddings).all())

    def test_custom_type_mapping(self) -> None:
        """An explicit atomic-number map reproduces the type-map default."""
        coord, atype, box = self._system()
        ref = DPA4Wrapper(self.model)
        override = DPA4Wrapper(
            self.model,
            atomic_number_to_type={ELEMENT_TO_Z["O"]: 0, ELEMENT_TO_Z["H"]: 1},
        )

        batch_ref = Batch.from_data_list(
            [self._data(coord, atype, box)], device=self.device
        )
        batch_ovr = Batch.from_data_list(
            [self._data(coord, atype, box)], device=self.device
        )
        out_ref = self._wrapper_batch_out(ref, batch_ref)
        out_ovr = self._wrapper_batch_out(override, batch_ovr)
        torch.testing.assert_close(out_ref["energy"], out_ovr["energy"])

    def test_unknown_atomic_number_raises(self) -> None:
        """An atomic number outside the type map raises a clear error."""
        coord, atype, box = self._system()
        wrapper = DPA4Wrapper(self.model)
        data = AtomicData(
            atomic_numbers=torch.full(
                (coord.shape[0],), 6, dtype=torch.long, device=self.device
            ),  # carbon: not in the O/H type map
            positions=coord.clone(),
            cell=box.reshape(1, 3, 3),
            pbc=torch.ones(1, 3, dtype=torch.bool, device=self.device),
        )
        batch = Batch.from_data_list([data], device=self.device)
        compute_neighbors(batch, config=wrapper.model_config.neighbor_config)
        with self.assertRaises(ValueError):
            wrapper(batch)


if __name__ == "__main__":
    unittest.main()
