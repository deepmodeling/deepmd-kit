# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for SeZM LAMMPS multi-rank (edge-based) inference.

The parallel path expands the descriptor node set to the extended region and
refreshes ghost-node features through ``deepmd_export::border_op`` between
interaction blocks. A single process can emulate one MPI rank by driving
``border_op`` with a self-send swap whose send-list maps each ghost slot to its
local owner; the exchange then reduces to an exact owner->ghost copy, so the
parallel path must reproduce the single-domain (folded) path bit-for-bit on the
owned atoms. These tests pin that equivalence end-to-end (descriptor and model)
and guard the export-capability predicate used to gate the with-comm artifact.
"""

from __future__ import (
    annotations,
)

import ctypes
import unittest

import numpy as np
import torch

from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.pt.model.descriptor.sezm_nn import block as sezm_block
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt_expt.utils.comm import (
    ensure_comm_registered,
)

# Self-send send-lists embed the address of a numpy array; the arrays must
# outlive the eager ``border_op`` calls that dereference them.
_SENDLIST_KEEPALIVE: list[np.ndarray] = []


def _tiny_parallel_model_params(**overrides) -> dict:
    """Minimal fp64 SeZM config exercising message passing and FiLM/GIE seeds."""
    descriptor = {
        "type": "SeZM",
        "sel": [8, 8],
        "rcut": 4.0,
        "channels": 8,
        "n_focus": 1,
        "n_radial": 4,
        "radial_mlp": [8],
        "use_env_seed": True,
        "l_schedule": [1, 1, 0],
        "mmax": 1,
        "so2_layers": 2,
        "n_atten_head": 1,
        "ffn_neurons": 16,
        "ffn_blocks": 1,
        "mlp_bias": False,
        "use_amp": False,
        "random_gamma": False,
        "precision": "float64",
        "seed": 11,
    }
    descriptor.update(overrides.pop("descriptor", {}))
    params = {
        "type": "SeZM",
        "type_map": ["A", "B"],
        "descriptor": descriptor,
        "fitting_net": {
            "neuron": [16],
            "activation_function": "silu",
            "precision": "float64",
            "seed": 11,
        },
        "use_compile": False,
    }
    params.update(overrides)
    return params


def _build_model(device: torch.device, **overrides) -> torch.nn.Module:
    """Build a tiny SeZM model in eval mode on ``device``."""
    model = get_model(_tiny_parallel_model_params(**overrides))
    model.eval()
    model.to(device)
    return model


def _build_extended_system(
    model: torch.nn.Module,
    device: torch.device,
    *,
    nloc: int = 6,
    seed: int = 3,
) -> dict[str, torch.Tensor]:
    """Build one periodic frame with ghost atoms and its edge schema.

    Returns the folded edge schema (single-domain convention) together with the
    extended atom types and the extended-to-local mapping needed to assemble the
    self-send communication plan.
    """
    rcut = float(model.get_rcut())
    sel = list(model.get_sel())
    ntypes = len(model.get_type_map())
    box_size = rcut * 2.5
    box = np.eye(3, dtype=np.float64) * box_size

    rng = np.random.default_rng(seed)
    coord_np = rng.random((1, nloc, 3), dtype=np.float64) * box_size
    atype_np = (np.arange(nloc, dtype=np.int32) % ntypes).reshape(1, nloc)

    coord_norm = normalize_coord(coord_np, np.tile(box.reshape(1, 3, 3), (1, 1, 1)))
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_norm, atype_np, box.reshape(1, 9), rcut
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        distinguish_types=not model.mixed_types(),
    )
    extended_coord = extended_coord.reshape(1, -1, 3)

    ext_coord = torch.tensor(extended_coord, dtype=torch.float64, device=device)
    ext_atype = torch.tensor(extended_atype, dtype=torch.int64, device=device)
    nlist_t = torch.tensor(nlist, dtype=torch.int64, device=device)
    mapping_t = torch.tensor(mapping, dtype=torch.int64, device=device)

    formatted = model.format_nlist(ext_coord, ext_atype, nlist_t)
    from deepmd.pt_expt.utils.edge_schema import (
        edge_schema_from_extended,
    )

    schema = edge_schema_from_extended(
        ext_coord, ext_atype[:, :nloc], formatted, mapping_t
    )
    return {
        "coord": schema.coord,
        "atype": schema.atype,
        "extended_atype": ext_atype,
        "edge_index": schema.edge_index,
        "edge_vec": schema.edge_vec,
        "edge_scatter_index": schema.edge_scatter_index,
        "edge_mask": schema.edge_mask,
        "mapping": mapping_t,
        "nloc": nloc,
        "nall": ext_coord.shape[1],
    }


def _self_comm_dict(
    mapping: torch.Tensor,
    nloc: int,
    nall: int,
) -> dict[str, torch.Tensor]:
    """Build a single self-send swap that copies each owner into its ghost slot.

    Ghost slot ``k`` reads local index ``mapping[nloc + k]`` (its owner), so the
    eager ``border_op`` self-send memcpy reproduces the folded gather exactly.
    Control tensors live on CPU per the C++ host-side dereference contract.
    """
    nghost = nall - nloc
    send_count = max(1, nghost)
    owner = mapping[0, nloc:nall].to(dtype=torch.int32).cpu().numpy()
    indices = np.ascontiguousarray(np.resize(owner, send_count).astype(np.int32))
    _SENDLIST_KEEPALIVE.append(indices)
    addr = indices.ctypes.data_as(ctypes.c_void_p).value
    cpu = torch.device("cpu")
    return {
        "send_list": torch.tensor([addr], dtype=torch.int64, device=cpu),
        "send_proc": torch.zeros(1, dtype=torch.int32, device=cpu),
        "recv_proc": torch.zeros(1, dtype=torch.int32, device=cpu),
        "send_num": torch.tensor([send_count], dtype=torch.int32, device=cpu),
        "recv_num": torch.tensor([send_count], dtype=torch.int32, device=cpu),
        "communicator": torch.zeros(1, dtype=torch.int64, device=cpu),
        "nlocal": torch.tensor(nloc, dtype=torch.int32, device=cpu),
        "nghost": torch.tensor(nghost, dtype=torch.int32, device=cpu),
    }


def _perturb_descriptor(descriptor: torch.nn.Module, *, seed: int = 0) -> None:
    """Push descriptor weights away from their near-identity initialization.

    SeZM initializes interaction blocks close to identity, so a freshly built
    model has near-zero message-passing contributions. That masks ghost-exchange
    bugs whose error is proportional to the convolution output. Perturbing every
    parameter simulates a trained model and makes the parity test sensitive to
    them.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for param in descriptor.parameters():
            noise = torch.randn(
                param.shape,
                generator=generator,
                dtype=param.dtype,
                device="cpu",
            ).to(param.device)
            param.add_(noise * 0.5)


class TestSeZMSelfCommParity(unittest.TestCase):
    """Self-send ``border_op`` must reproduce the single-domain folded path.

    The attention-residual configurations are exercised with perturbed weights:
    the depth-history feeds the SO(2) convolution, so a trained model (non-zero
    message passing) is required to catch a stale-ghost regression there.
    """

    @classmethod
    def setUpClass(cls) -> None:
        ensure_comm_registered()

    def _assert_parity(
        self,
        device: torch.device,
        rtol: float,
        atol: float,
        *,
        descriptor_overrides: dict | None = None,
    ) -> None:
        model = _build_model(device, descriptor=descriptor_overrides or {})
        # An untrained SeZM model is geometry-independent (identically zero
        # forces), for which the parity comparison below holds vacuously for any
        # ghost-exchange implementation. Perturbing the descriptor (see
        # ``_perturb_descriptor``) restores non-zero, ghost-feature-dependent
        # forces so the comparison is load-bearing.
        _perturb_descriptor(model.atomic_model.descriptor)
        sysm = _build_extended_system(model, device)
        comm = _self_comm_dict(sysm["mapping"], sysm["nloc"], sysm["nall"])

        ref = model.forward_lower(
            sysm["coord"],
            sysm["atype"],
            sysm["edge_index"],
            sysm["edge_vec"],
            sysm["edge_scatter_index"],
            sysm["edge_mask"],
            do_atomic_virial=True,
        )
        par = model.forward_lower(
            sysm["coord"],
            sysm["atype"],
            # The parallel path indexes the extended node set directly, so the
            # extended scatter index doubles as the message-passing edge_index.
            sysm["edge_scatter_index"],
            sysm["edge_vec"],
            sysm["edge_scatter_index"],
            sysm["edge_mask"],
            do_atomic_virial=True,
            comm_dict=comm,
            extended_atype=sysm["extended_atype"],
        )

        # Reject the degenerate zero-force regime so the parity assertion can
        # never pass vacuously: the reference force field must dominate the
        # comparison tolerance.
        self.assertGreater(
            ref["extended_force"].abs().max().item(),
            atol * 1e3,
            msg="reference forces are ~0; the parity check would be vacuous",
        )
        for key in ("energy", "extended_force", "virial", "extended_virial"):
            torch.testing.assert_close(
                par[key], ref[key], rtol=rtol, atol=atol, msg=f"mismatch in {key}"
            )

    def test_parity_cpu(self) -> None:
        self._assert_parity(torch.device("cpu"), rtol=1e-8, atol=1e-9)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_parity_cuda(self) -> None:
        # CUDA atomic scatter reorders accumulation, so the tolerance is looser
        # than the deterministic CPU path while still pinning correctness.
        self._assert_parity(env.DEVICE, rtol=1e-6, atol=1e-7)

    def test_parity_full_attn_res_cpu(self) -> None:
        self._assert_parity(
            torch.device("cpu"),
            rtol=1e-8,
            atol=1e-9,
            descriptor_overrides={"full_attn_res": "dependent", "so2_layers": 2},
        )

    def test_parity_block_attn_res_cpu(self) -> None:
        self._assert_parity(
            torch.device("cpu"),
            rtol=1e-8,
            atol=1e-9,
            descriptor_overrides={"block_attn_res": "dependent", "so2_layers": 2},
        )

    def test_parity_no_env_seed_single_block_cpu(self) -> None:
        # env_seed off + a single block: the only neighbour feature block 0
        # reads is the type embedding, which a rank can recompute from
        # ``extended_atype`` -- the ghost exchange is then redundant but must
        # stay exact (it copies identical type embeddings).
        self._assert_parity(
            torch.device("cpu"),
            rtol=1e-8,
            atol=1e-9,
            descriptor_overrides={"l_schedule": [1], "use_env_seed": False},
        )

    def test_parity_no_env_seed_cpu(self) -> None:
        # Multi-block without env-seed: ghost features still carry block outputs
        # that a rank cannot recompute, so the exchange is load-bearing.
        self._assert_parity(
            torch.device("cpu"),
            rtol=1e-8,
            atol=1e-9,
            descriptor_overrides={"use_env_seed": False},
        )


class TestSeZMDescriptorSelfCommParity(unittest.TestCase):
    """Descriptor-level parity isolates the ghost-exchange from the force scatter."""

    @classmethod
    def setUpClass(cls) -> None:
        ensure_comm_registered()

    def test_descriptor_parity_cpu(self) -> None:
        device = torch.device("cpu")
        model = _build_model(device, descriptor={"full_attn_res": "dependent"})
        descriptor = model.atomic_model.descriptor
        _perturb_descriptor(descriptor)
        sysm = _build_extended_system(model, device)
        comm = _self_comm_dict(sysm["mapping"], sysm["nloc"], sysm["nall"])

        ref, _ = descriptor.forward_with_edges(
            extended_coord=sysm["coord"][:, : sysm["nloc"], :],
            extended_atype=sysm["atype"],
            edge_index=sysm["edge_index"],
            edge_vec=sysm["edge_vec"],
            edge_mask=sysm["edge_mask"],
        )
        par, _ = descriptor.forward_with_edges(
            extended_coord=sysm["coord"],
            extended_atype=sysm["extended_atype"],
            edge_index=sysm["edge_scatter_index"],
            edge_vec=sysm["edge_vec"],
            edge_mask=sysm["edge_mask"],
            comm_dict=comm,
            nloc=sysm["nloc"],
        )
        torch.testing.assert_close(par, ref, rtol=1e-8, atol=1e-9)


class TestSeZMEdgeParallelCapability(unittest.TestCase):
    """The with-comm export predicate gates bridging and spin out."""

    def test_plain_model_supports_edge_parallel(self) -> None:
        model = _build_model(torch.device("cpu"))
        self.assertTrue(model.supports_edge_parallel())
        self.assertTrue(
            model.atomic_model.descriptor.has_message_passing_across_ranks()
        )

    def test_bridging_model_fails_fast(self) -> None:
        # ZBL needs real element symbols for its analytical pair potential.
        model = _build_model(
            torch.device("cpu"),
            type_map=["O", "H"],
            bridging_method="ZBL",
            bridging_r_inner=0.5,
            bridging_r_outer=1.0,
        )
        self.assertFalse(model.supports_edge_parallel())


class TestSeZMExchangeSchedule(unittest.TestCase):
    """The ghost exchange is scheduled per block, not blanket-applied.

    A block exchanges only when its SO(2) convolution reads neighbour rows that
    the local rank cannot rebuild: block 0 needs it only with env-seed/GIE (which
    fold neighbour environment into the initial state), and later blocks always
    need it (they read previous-block outputs). A purely local model
    (``use_env_seed=False`` with one block) must therefore communicate zero
    times, preserving its single-pass speed under domain decomposition.
    """

    @classmethod
    def setUpClass(cls) -> None:
        ensure_comm_registered()

    def _count_exchanges(self, descriptor_overrides: dict) -> int:
        device = torch.device("cpu")
        model = _build_model(device, descriptor=descriptor_overrides)
        sysm = _build_extended_system(model, device)
        comm = _self_comm_dict(sysm["mapping"], sysm["nloc"], sysm["nall"])
        real = sezm_block.exchange_ghost_features
        count = 0

        def counting(*args, **kwargs):
            nonlocal count
            count += 1
            return real(*args, **kwargs)

        sezm_block.exchange_ghost_features = counting
        try:
            model.forward_lower(
                sysm["coord"],
                sysm["atype"],
                sysm["edge_scatter_index"],
                sysm["edge_vec"],
                sysm["edge_scatter_index"],
                sysm["edge_mask"],
                comm_dict=comm,
                extended_atype=sysm["extended_atype"],
            )
        finally:
            sezm_block.exchange_ghost_features = real
        return count

    def test_local_single_block_skips_all_comm(self) -> None:
        self.assertEqual(
            self._count_exchanges({"l_schedule": [1], "use_env_seed": False}), 0
        )

    def test_env_seed_single_block_exchanges_once(self) -> None:
        self.assertEqual(
            self._count_exchanges({"l_schedule": [1], "use_env_seed": True}), 1
        )

    def test_no_env_seed_multi_block_skips_first(self) -> None:
        self.assertEqual(
            self._count_exchanges({"l_schedule": [1, 1, 0], "use_env_seed": False}), 2
        )

    def test_env_seed_multi_block_exchanges_every_block(self) -> None:
        self.assertEqual(
            self._count_exchanges({"l_schedule": [1, 1, 0], "use_env_seed": True}), 3
        )


if __name__ == "__main__":
    unittest.main()
