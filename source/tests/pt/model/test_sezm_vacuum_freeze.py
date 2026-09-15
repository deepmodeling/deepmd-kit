# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freezing a SeZM model with the isolated-atom reference.

The ``.pt2`` freeze folds the vacuum reference into the fitting bias, so the
frozen model carries no reference atoms yet reproduces the referenced eager
model: an isolated atom gives exactly its bias and a cluster gives the eager
energies.
"""

import copy
import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.pt.entrypoints.freeze_pt2 import (
    freeze_sezm_to_pt2,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)

from .test_sezm_export import (
    _CPU,
    _SKIP_OFF_COMPILE_TORCH,
    _SKIP_OFF_COMPILE_TORCH_REASON,
    _clear_default_device,
    _tiny_sezm_model_params,
)

BIAS = np.array([[-3.0], [0.5]])


@unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
class TestSeZMVacuumFreeze(unittest.TestCase):
    def test_frozen_model_folds_the_reference(self) -> None:
        self.check_frozen_model(numb_fparam=0)

    def test_frozen_model_keeps_the_reference_with_fparam(self) -> None:
        """With frame parameters the archive references from the stored vacuum table."""
        self.check_frozen_model(numb_fparam=1)

    @unittest.skipIf(not torch.cuda.is_available(), "the CUDA target needs a GPU")
    def test_frozen_model_on_a_cuda_target(self) -> None:
        """The fold and the compiled archive share the CUDA target."""
        self.check_frozen_model(numb_fparam=1, device=torch.device("cuda"))

    def check_frozen_model(self, numb_fparam: int, device: torch.device = _CPU) -> None:
        params = _tiny_sezm_model_params()
        params["fitting_net"]["vacuum_ref"] = True
        params["fitting_net"]["numb_fparam"] = numb_fparam
        fparam = None if numb_fparam == 0 else np.array([[0.7]])
        model = get_model(params)
        model.eval()
        model.to(device)
        fitting = model.atomic_model.fitting_net
        with torch.no_grad():
            fitting.bias_atom_e.copy_(
                torch.as_tensor(BIAS, dtype=fitting.bias_atom_e.dtype, device=device)
            )
        self.assertTrue(fitting.vacuum_ref)

        box_edge = params["descriptor"]["rcut"] * 3.0
        cell = (np.eye(3) * box_edge).reshape(1, 9)
        rng = np.random.default_rng(2026)
        natoms = 5
        atype = np.array([0, 1, 0, 1, 0], dtype=np.int32)
        coord = rng.random((1, natoms, 3)) * box_edge * 0.4 + box_edge * 0.3
        eager = (
            model.forward(
                torch.tensor(coord, dtype=torch.float64, device=device),
                torch.tensor(atype, dtype=torch.int64, device=device).unsqueeze(0),
                torch.tensor(cell, dtype=torch.float64, device=device),
                fparam=None
                if fparam is None
                else torch.tensor(fparam, dtype=torch.float64, device=device),
            )["atom_energy"]
            .detach()
            .cpu()
            .numpy()
        )

        import deepmd.pt_expt.utils.env as pt_expt_env
        from deepmd.infer import (
            DeepPot,
        )

        with tempfile.TemporaryDirectory() as tmp, _clear_default_device():
            wrapper = ModelWrapper(model, model_params=copy.deepcopy(params))
            ckpt = Path(tmp) / "vacuum.pt"
            torch.save({"model": wrapper.state_dict()}, ckpt)
            out = Path(tmp) / "vacuum.pt2"
            freeze_sezm_to_pt2(str(ckpt), str(out), device=device)
            # the evaluator runs on the device the archive is compiled for
            saved_device = pt_expt_env.DEVICE
            pt_expt_env.DEVICE = device
            try:
                dp = DeepPot(str(out))
                for itype in range(2):
                    _, _, _, atom_energy, _ = dp.eval(
                        np.full((1, 1, 3), box_edge / 2.0),
                        cell,
                        np.array([itype], dtype=np.int32),
                        atomic=True,
                        fparam=fparam,
                    )
                    np.testing.assert_allclose(
                        atom_energy.reshape(-1), BIAS[itype], rtol=1e-8, atol=1e-8
                    )
                _, _, _, atom_energy, _ = dp.eval(
                    coord, cell, atype, atomic=True, fparam=fparam
                )
                np.testing.assert_allclose(
                    atom_energy.reshape(-1), eager.reshape(-1), rtol=1e-8, atol=1e-8
                )
            finally:
                pt_expt_env.DEVICE = saved_device
        # the training model itself keeps its reference
        self.assertTrue(fitting.vacuum_ref)


if __name__ == "__main__":
    unittest.main()
