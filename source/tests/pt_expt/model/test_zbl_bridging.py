# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt ZBL bridging (review 3638077323): eager pt parity, FD force,
with-comm gate, and pt-checkpoint interop for ``bridging_method="ZBL"``.
"""

import copy

import numpy as np
import torch

from deepmd.pt.model.model import get_model as pt_get_model
from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.fitting.dpa4_ener import (
    SeZMEnergyFittingNet,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)

from ...seed import (
    GLOBAL_SEED,
)

ZBL_CONFIG = {
    "type": "dpa4",
    "type_map": ["Ni", "O"],
    "descriptor": {
        "type": "dpa4",
        "rcut": 4.0,
        "sel": 8,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "precision": "float64",
        "seed": 7,
        "random_gamma": False,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [8, 8],
        "precision": "float64",
        "seed": 7,
    },
    "bridging_method": "ZBL",
    "bridging_r_inner": 0.8,
    "bridging_r_outer": 1.2,
}


def _close_pair_system(cpu):
    generator = torch.Generator(device=cpu).manual_seed(GLOBAL_SEED + 2)
    nloc = 6
    cell = torch.rand([3, 3], dtype=torch.float64, generator=generator)
    cell = (cell + cell.T) + 6.0 * torch.eye(3)
    coord = 1.5 + 3.0 * torch.rand([nloc, 3], dtype=torch.float64, generator=generator)
    coord[1] = coord[0] + torch.tensor([0.95, 0.0, 0.0], dtype=torch.float64)
    atype = torch.tensor([[0, 0, 1, 0, 1, 1]], dtype=torch.int64)
    return coord.unsqueeze(0), atype, cell.reshape(1, 9)


class TestZBLBridgingPtExpt:
    def setup_method(self) -> None:
        cpu = torch.device("cpu")
        self.pt_model = pt_get_model(copy.deepcopy(ZBL_CONFIG)).to(torch.float64)
        self.pt_model = self.pt_model.eval().to(cpu)
        assert self.pt_model.inter_potential is not None

        pt_expt_model = get_model(copy.deepcopy(ZBL_CONFIG))
        atomic = pt_expt_model.atomic_model
        assert atomic.inter_potential is not None
        # weight copy: pt DescrptSeZM / fitting serialize to the SAME
        # backend-agnostic dict schema (incl. the InnerClamp radii)
        atomic.descriptor = DescrptDPA4.deserialize(
            self.pt_model.atomic_model.descriptor.serialize()
        )
        atomic.fitting_net = SeZMEnergyFittingNet.deserialize(
            self.pt_model.atomic_model.fitting_net.serialize()
        )
        self.pt_expt_model = pt_expt_model.to(cpu).eval()
        self.coord, self.atype, self.box = _close_pair_system(cpu)

    def test_parity_vs_pt_with_zbl(self) -> None:
        """Weight-copied fp64 parity incl. the ZBL term (energy/force/virial)."""
        out_pt = self.pt_model.forward(self.coord, self.atype, self.box)
        out_pte = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)
        for key in ("energy", "force", "virial"):
            torch.testing.assert_close(
                out_pt[key], out_pte[key], rtol=1e-12, atol=1e-12, msg=key
            )

    def test_zbl_energy_is_positive_addition(self) -> None:
        """Same weights minus the bridging key -> plain twin; diff > 0."""
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        data = self.pt_expt_model.serialize()
        assert data["bridging_method"] == "ZBL"
        plain_data = copy.deepcopy(data)
        plain_data.pop("bridging_method")
        m_plain = BaseModel.deserialize(plain_data).to(torch.device("cpu")).eval()
        e_zbl = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)[
            "energy"
        ]
        e_plain = m_plain.forward(self.coord, self.atype, box=self.box)["energy"]
        assert float((e_zbl - e_plain).sum()) > 1e-3

    def test_force_matches_finite_difference(self) -> None:
        """F = -dE/dx through the ZBL-carrying edge autograd (central FD)."""
        eps = 1e-5
        out = self.pt_expt_model.forward(self.coord, self.atype, box=self.box)
        force = out["force"].reshape(-1, 3)
        for atom, comp in ((1, 0), (2, 2)):  # close-pair atom + a far atom
            cp = self.coord.clone()
            cp[0, atom, comp] += eps
            ep = self.pt_expt_model.forward(cp, self.atype, box=self.box)["energy"]
            cm = self.coord.clone()
            cm[0, atom, comp] -= eps
            em = self.pt_expt_model.forward(cm, self.atype, box=self.box)["energy"]
            fd = -float((ep - em).sum()) / (2 * eps)
            np.testing.assert_allclose(
                float(force[atom, comp]), fd, rtol=1e-6, atol=1e-6
            )

    def test_with_comm_gate_off_for_bridging(self) -> None:
        """Bridging models never compile a with-comm artifact (single-rank)."""
        from deepmd.pt_expt.utils.serialization import (
            _needs_with_comm_artifact,
        )

        assert (
            _needs_with_comm_artifact(self.pt_expt_model, lower_kind="graph") is False
        )

    def test_pt_checkpoint_interop(self) -> None:
        """A pt-serialized ZBL SeZM checkpoint deserializes into pt_expt."""
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        pt_data = self.pt_model.serialize()
        assert str(pt_data.get("bridging_method", "none")).upper() == "ZBL"
        m2 = BaseModel.deserialize(pt_data)
        assert m2.atomic_model.inter_potential is not None
        m2 = m2.to(torch.device("cpu")).eval()
        out_pt = self.pt_model.forward(self.coord, self.atype, self.box)
        out2 = m2.forward(self.coord, self.atype, box=self.box)
        torch.testing.assert_close(
            out_pt["energy"], out2["energy"], rtol=1e-12, atol=1e-12
        )
