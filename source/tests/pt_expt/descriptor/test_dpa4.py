# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DPDescrptDPA4
from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.env import (
    PRECISION_DICT,
)

from ...common.test_mixins import (
    TestCaseSingleFrameWithNlist,
)


def make_descriptor(nt, sel, rcut, **overrides) -> DescrptDPA4:
    kwargs = {
        "ntypes": nt,
        "sel": sel,
        "rcut": rcut,
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


class TestDescrptDPA4(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    @pytest.mark.parametrize("use_env_seed", [True, False])  # env seed feature
    @pytest.mark.parametrize("use_mapping", [True, False])  # pass mapping vs None
    def test_consistency(self, use_env_seed, use_mapping) -> None:
        dtype = PRECISION_DICT["float64"]
        err_msg = f"use_env_seed={use_env_seed} use_mapping={use_mapping}"
        dd0 = make_descriptor(
            self.nt,
            self.sel_mix,
            self.rcut,
            use_env_seed=use_env_seed,
        ).to(self.device)
        coord_ext = torch.tensor(self.coord_ext, dtype=dtype, device=self.device)
        atype_ext = torch.tensor(self.atype_ext, dtype=int, device=self.device)
        nlist = torch.tensor(self.nlist, dtype=int, device=self.device)
        mapping = (
            torch.tensor(self.mapping, dtype=int, device=self.device)
            if use_mapping
            else None
        )
        rd0 = dd0(coord_ext, atype_ext, nlist, mapping)[0]
        # serialization round-trip within pt_expt
        dd1 = DescrptDPA4.deserialize(dd0.serialize())
        rd1 = dd1(coord_ext, atype_ext, nlist, mapping)[0]
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(),
            rd1.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-14,
            err_msg=err_msg,
        )
        # dpmodel (numpy) impl
        dd2 = DPDescrptDPA4.deserialize(dd0.serialize())
        rd2 = dd2.call(
            self.coord_ext,
            self.atype_ext,
            self.nlist,
            mapping=self.mapping if use_mapping else None,
        )[0]
        # CPU: strict same-math parity; CUDA: ULP / nondeterministic reduction slack
        if self.device == "cpu" or str(self.device) == "cpu":
            rtol, atol = 1e-12, 1e-14
        else:
            rtol, atol = 1e-10, 1e-12
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(),
            rd2,
            rtol=rtol,
            atol=atol,
            err_msg=err_msg,
        )

    @pytest.mark.parametrize("prec", ["float64"])  # precision
    def test_exportable(self, prec) -> None:
        dtype = PRECISION_DICT[prec]
        dd0 = make_descriptor(self.nt, self.sel_mix, self.rcut, precision=prec).to(
            self.device
        )
        dd0 = dd0.eval()
        inputs = (
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
        )
        torch.export.export(dd0, inputs)

    @pytest.mark.parametrize("prec", ["float64"])  # precision
    def test_make_fx(self, prec) -> None:
        dtype = PRECISION_DICT[prec]
        dd0 = make_descriptor(self.nt, self.sel_mix, self.rcut, precision=prec).to(
            self.device
        )
        dd0 = dd0.eval()
        coord_ext = torch.tensor(self.coord_ext, dtype=dtype, device=self.device)
        atype_ext = torch.tensor(self.atype_ext, dtype=int, device=self.device)
        nlist = torch.tensor(self.nlist, dtype=int, device=self.device)

        def fn(coord_ext, atype_ext, nlist):
            coord_ext = coord_ext.detach().requires_grad_(True)
            rd = dd0(coord_ext, atype_ext, nlist)[0]
            grad = torch.autograd.grad(rd.sum(), coord_ext, create_graph=False)[0]
            return rd, grad

        rd_eager, grad_eager = fn(coord_ext, atype_ext, nlist)
        traced = make_fx(fn)(coord_ext, atype_ext, nlist)
        rd_traced, grad_traced = traced(coord_ext, atype_ext, nlist)
        np.testing.assert_allclose(
            rd_eager.detach().cpu().numpy(),
            rd_traced.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            grad_eager.detach().cpu().numpy(),
            grad_traced.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_trainable_parameters(self) -> None:
        # `_promote_trainable_tree` must promote every weight that is a
        # trainable nn.Parameter in the reference pt SeZM implementation
        # (full 1:1 gradient parity is proven in
        # source/tests/pt/model/test_dpa4_ptexpt_grad_parity.py)
        dd0 = make_descriptor(self.nt, self.sel_mix, self.rcut, use_env_seed=True).to(
            self.device
        )
        param_names = dict(dd0.named_parameters())
        buffer_names = dict(dd0.named_buffers())
        # spot-check known trainable weights are Parameters
        for name in (
            "type_embedding.adam_type_embedding",
            "radial_basis.adam_freqs",
            "film_scale_strength_log",
            "blocks.0.so2_conv.so2_linears.0.weight_m0",
            "blocks.0.so2_conv.so2_linears.0.weight_m.0",
            "blocks.0.so2_conv.non_linearities.0.gate_linear.weight",
            "blocks.0.post_so2_norm.adam_scale",
            "blocks.0.ffns.0.act.grid_op.left_proj.weight",
            "output_ffn.so3_linear_1.weight",
        ):
            assert name in param_names, f"{name} not promoted to Parameter"
            assert param_names[name].requires_grad
        # spot-check constants stay buffers (pt registers them as buffers)
        for name in (
            "mean",
            "stddev",
            "blocks.0.post_so2_norm.balance_weight",
            "blocks.0.so2_conv.rotate_inv_rescale_full",
        ):
            assert name in buffer_names, f"{name} should stay a buffer"
            assert name not in param_names
        # wigner tables must never be trainable
        assert not any("wigner" in n.lower() for n in param_names)
        # all promoted parameters are float and trainable
        for name, p in param_names.items():
            assert p.is_floating_point(), name
            assert p.requires_grad, name

    @pytest.mark.parametrize(
        "via_deserialize", [False, True]
    )  # constructor vs round-trip
    def test_trainable_false_freezes_all_parameters(self, via_deserialize) -> None:
        # trainable=False must freeze every parameter, including ParameterList
        # entries such as SO2Linear.weight_m (mmax>=1) that dpmodel_setattr
        # converts with requires_grad=True
        dd0 = make_descriptor(
            self.nt, self.sel_mix, self.rcut, use_env_seed=True, trainable=False
        ).to(self.device)
        if via_deserialize:
            dd0 = DescrptDPA4.deserialize(dd0.serialize())
        params = dict(dd0.named_parameters())
        assert any(".weight_m." in n for n in params)  # mmax>=1 exercised
        frozen = [n for n, p in params.items() if not p.requires_grad]
        assert frozen == list(params), (
            f"trainable=False left parameters trainable: "
            f"{sorted(set(params) - set(frozen))}"
        )


# `use_amp` is a pt-runtime CUDA autocast switch with no dpmodel/pt_expt effect;
# constructing the descriptor with it truthy must emit a warn-once message.
@pytest.mark.parametrize("use_amp", [True, False])  # truthy warns, falsy is silent
def test_use_amp_warns_once(use_amp, caplog, monkeypatch) -> None:
    import logging

    import deepmd.dpmodel.descriptor.dpa4 as dpa4_mod

    # reset the warn-once flag so the assertion is deterministic regardless of
    # test ordering (other constructions in the suite may have already warned)
    monkeypatch.setattr(dpa4_mod, "_USE_AMP_WARNED", False)

    def _construct() -> None:
        make_descriptor(2, [10, 10], 4.0, use_amp=use_amp)

    with caplog.at_level(logging.WARNING, logger=dpa4_mod.log.name):
        _construct()
    matches = [r for r in caplog.records if "use_amp" in r.getMessage()]
    if use_amp:
        assert len(matches) == 1, caplog.text
        # second construction must NOT warn again (warn-once per process)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=dpa4_mod.log.name):
            _construct()
        assert not [r for r in caplog.records if "use_amp" in r.getMessage()]
    else:
        assert not matches, caplog.text
