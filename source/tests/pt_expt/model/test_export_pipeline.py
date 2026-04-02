# SPDX-License-Identifier: LGPL-3.0-or-later
"""Full export pipeline tests mirroring _trace_and_export in serialization.py.

Each test case exercises the complete dp freeze pipeline:
  1. Build model via get_model
  2. Serialize → deserialize round-trip
  3. Eager reference
  4. make_fx tracing with tracing_mode="symbolic"
  5. torch.export.export with dynamic shapes
  6. .pte save → load round-trip
  7. Verify loaded matches eager (same shapes)
  8. Verify loaded matches eager (different shapes)
  9. Verify fparam actually affects output (when with_fparam=True)
"""

import tempfile

import numpy as np
import pytest
import torch

import deepmd.pt_expt.utils.env as _env
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.model import (
    BaseModel,
)
from deepmd.pt_expt.utils.serialization import (
    _build_dynamic_shapes,
    _collect_metadata,
    _make_sample_inputs,
)

CONFIGS = {
    "se_e2_a": {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [6, 6],
            "rcut": 4.0,
            "rcut_smth": 0.5,
            "neuron": [8, 16],
            "axis_neuron": 4,
            "seed": 1,
        },
        "fitting_net": {"neuron": [16, 16], "seed": 1},
    },
    "dpa1": {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa1",
            "sel": 12,
            "rcut": 4.0,
            "rcut_smth": 0.5,
            "neuron": [8, 16],
            "axis_neuron": 4,
            "attn": 4,
            "attn_layer": 1,
            "attn_dotr": True,
            "seed": 1,
        },
        "fitting_net": {"neuron": [16, 16], "seed": 1},
    },
    "dpa2": {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa2",
            "repinit": {
                "rcut": 4.0,
                "rcut_smth": 0.5,
                "nsel": 12,
                "neuron": [6, 12],
                "axis_neuron": 3,
                "seed": 1,
            },
            "repformer": {
                "rcut": 3.0,
                "rcut_smth": 0.5,
                "nsel": 6,
                "nlayers": 2,
                "g1_dim": 8,
                "g2_dim": 4,
                "seed": 1,
            },
        },
        "fitting_net": {"neuron": [16, 16], "seed": 1},
    },
}


def _get_config(descriptor_type: str, with_fparam: bool) -> dict:
    """Return a deep copy of the config with optional fparam."""
    import copy

    config = copy.deepcopy(CONFIGS[descriptor_type])
    if with_fparam:
        config["fitting_net"]["numb_fparam"] = 2
    return config


class TestExportPipeline:
    @pytest.mark.parametrize("descriptor_type", ["se_e2_a", "dpa1", "dpa2"])
    @pytest.mark.parametrize("with_fparam", [False, True])  # frame parameter
    def test_export_pipeline(self, descriptor_type, with_fparam) -> None:
        config = _get_config(descriptor_type, with_fparam)

        # 1. Build model via get_model (same as dp freeze)
        model = get_model(config)
        model.to("cpu")
        model.eval()

        # 2. Serialize → deserialize round-trip (same as dp freeze)
        model_data = model.serialize()
        model2 = BaseModel.deserialize(model_data)
        model2.to("cpu")
        model2.eval()

        # 3. Create sample inputs on CPU for tracing (nframes=5 as in _trace_and_export)
        orig_device = _env.DEVICE
        _env.DEVICE = torch.device("cpu")
        try:
            inputs_trace = _make_sample_inputs(model2, nframes=5, nloc=7)
        finally:
            _env.DEVICE = orig_device
        ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam = inputs_trace

        # 4. Eager reference
        eager_out = model2.forward_common_lower(
            ext_coord.detach().requires_grad_(True),
            ext_atype,
            nlist_t,
            mapping_t,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=True,
        )

        # 5. Trace with symbolic mode (same as dp freeze)
        traced = model2.forward_common_lower_exportable(
            ext_coord,
            ext_atype,
            nlist_t,
            mapping_t,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=True,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )

        # 6. Export with dynamic shapes (same as dp freeze)
        dynamic_shapes = _build_dynamic_shapes(
            ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam
        )
        exported = torch.export.export(
            traced,
            (ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam),
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )

        # 7. .pte save → load round-trip
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=True) as tmp:
            tmpfile = tmp.name
        torch.export.save(exported, tmpfile)
        loaded = torch.export.load(tmpfile).module()

        # Clean up temp file
        import os

        os.unlink(tmpfile)

        # 8. Verify: traced output matches eager (same shapes as trace)
        traced_out = traced(ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam)
        for key in eager_out:
            np.testing.assert_allclose(
                eager_out[key].detach().cpu().numpy(),
                traced_out[key].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"traced vs eager (same shape): {key}",
            )

        # 9. Verify: loaded (.pte) output matches eager (same shapes)
        loaded_out = loaded(ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam)
        for key in eager_out:
            np.testing.assert_allclose(
                eager_out[key].detach().cpu().numpy(),
                loaded_out[key].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"loaded (.pte) vs eager (same shape): {key}",
            )

        # 10. Verify: loaded output matches eager with DIFFERENT shapes
        #     (tests that dynamic shapes work)
        _env.DEVICE = torch.device("cpu")
        try:
            inputs_infer = _make_sample_inputs(model2, nframes=3, nloc=11)
        finally:
            _env.DEVICE = orig_device
        (
            ext_coord2,
            ext_atype2,
            nlist_t2,
            mapping_t2,
            fparam2,
            aparam2,
        ) = inputs_infer

        eager_out2 = model2.forward_common_lower(
            ext_coord2.detach().requires_grad_(True),
            ext_atype2,
            nlist_t2,
            mapping_t2,
            fparam=fparam2,
            aparam=aparam2,
            do_atomic_virial=True,
        )
        loaded_out2 = loaded(
            ext_coord2, ext_atype2, nlist_t2, mapping_t2, fparam2, aparam2
        )
        for key in eager_out2:
            np.testing.assert_allclose(
                eager_out2[key].detach().cpu().numpy(),
                loaded_out2[key].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"loaded (.pte) vs eager (different shape): {key}",
            )

        # 11. Verify: metadata correctness
        metadata = _collect_metadata(model2)
        assert metadata["type_map"] == config["type_map"]
        assert metadata["dim_fparam"] == (2 if with_fparam else 0)
        assert metadata["rcut"] == model2.get_rcut()
        assert metadata["sel"] == model2.get_sel()
        assert metadata["mixed_types"] == model2.mixed_types()

        # 12. Verify: fparam actually affects output (when with_fparam=True)
        if with_fparam:
            fparam_ones = torch.ones_like(fparam)
            loaded_out_fp1 = loaded(
                ext_coord, ext_atype, nlist_t, mapping_t, fparam_ones, aparam
            )
            # Output with fparam=0 should differ from fparam=1
            assert not np.allclose(
                loaded_out["energy"].detach().cpu().numpy(),
                loaded_out_fp1["energy"].detach().cpu().numpy(),
            ), (
                "Changing fparam did not change output — "
                "fparam may be baked in as a constant"
            )
