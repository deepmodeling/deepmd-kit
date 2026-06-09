"""Tests for MFT downstream_task_type='property' branch.

These cover the paper-faithful (arXiv:2601.08486) DOWNSTREAM=property
configuration: a fresh property fitting_net + property loss for the
downstream head, while the aux branch keeps its ener fitting_net pulled
from the ckpt.

Back-compat: callers that don't pass downstream_task_type stay on the
legacy ener path (used by mp_data MFT sensitivity-analysis experiments).
"""

from __future__ import annotations

import pytest

from deepmd.dpa_adapt.config.manager import MFTConfigManager
from deepmd.dpa_adapt.mft import MFTFineTuner


class _FakePropertyTuner:
    """Tuner-shaped object configured for downstream_task_type='property'.
    Bypasses MFTFineTuner.__init__ so tests don't need a real ckpt."""
    pretrained = "/share/DPA-3.1-3M.pt"
    aux_branch = "SPICE2"
    aux_prob = 0.5
    aux_type_map = ["H", "C", "N", "O"]
    downstream_type_map = ["H", "C", "N", "O"]
    # aux fitting_net pulled from ckpt — an ener config (the actual SPICE2 head)
    fitting_net_params = {"type": "ener", "neuron": [240, 240, 240]}
    downstream_task_type = "property"
    property_name = "homo"
    task_dim = 1
    intensive = True
    learning_rate = 1e-3
    stop_lr = 1e-5
    max_steps = 1000
    batch_size = "auto:32"
    seed = 42
    output_dir = "/tmp/mft_property_test"
    save_freq = 500
    disp_freq = 100
    train_data = "/data/qm9_train"
    aux_data = "/data/spice2"
    valid_data = None


class _FakeEnerTuner:
    """Legacy back-compat tuner. NO downstream_task_type attr at all —
    must still build a valid ener-mode config (mp_data sensitivity callers
    construct tuners this way)."""
    pretrained = "/share/DPA-3.1-3M.pt"
    aux_branch = "MP_traj_v024_alldata_mixu"
    aux_prob = 0.5
    aux_type_map = ["Cu", "O"]
    downstream_type_map = ["Cu", "O"]
    fitting_net_params = {"type": "ener", "neuron": [240, 240, 240]}
    learning_rate = 1e-3
    stop_lr = 1e-5
    max_steps = 1000
    batch_size = "auto:32"
    seed = 42
    output_dir = "/tmp/mft_ener_test"
    save_freq = 500
    disp_freq = 100
    train_data = "/data/downstream"
    aux_data = "/data/aux"
    valid_data = None


# ---------------------------------------------------------------------------
# Property task: config shape
# ---------------------------------------------------------------------------

def test_property_task_config_has_property_fitting_net():
    """DOWNSTREAM fitting_net must be type='property' with the right
    property_name / task_dim / intensive, NOT the aux ener fitting_net."""
    config = MFTConfigManager(_FakePropertyTuner()).build()
    fn = config["model"]["model_dict"]["property"]["fitting_net"]
    assert fn["type"] == "property"
    assert fn["property_name"] == "homo"
    assert fn["task_dim"] == 1
    assert fn["intensive"] is True
    assert fn["neuron"] == [240, 240, 240]
    assert fn["activation_function"] == "tanh"
    assert fn["seed"] == 42
    # Required for DPA-3.1-3M multi-task case-embedding layer.
    assert fn["dim_case_embd"] == 31


def test_property_task_config_has_property_loss():
    """DOWNSTREAM loss must be type='property' with mse + mae/rmse metrics."""
    config = MFTConfigManager(_FakePropertyTuner()).build()
    loss = config["loss_dict"]["property"]
    assert loss["type"] == "property"
    assert loss["loss_func"] == "mse"
    assert "mae" in loss["metric"]
    assert "rmse" in loss["metric"]


def test_property_task_no_force_pref_in_loss():
    """The ener-task force/virial prefs MUST NOT leak into property loss.
    This is the regression that made MFT/homo training useless: the loss
    forced the model to predict zero forces against QM9 labels that don't
    have forces."""
    config = MFTConfigManager(_FakePropertyTuner()).build()
    loss = config["loss_dict"]["property"]
    for forbidden in (
        "start_pref_f", "limit_pref_f",
        "start_pref_v", "limit_pref_v",
        "start_pref_e", "limit_pref_e",
    ):
        assert forbidden not in loss, (
            f"property loss must not contain {forbidden}; "
            f"got loss={loss!r}"
        )


def test_property_task_no_property_name_in_loss():
    """deepmd 3.1.3 strict-mode dargs rejects unknown keys inside
    loss_property — property_name belongs on fitting_net, not loss.
    (Verified empirically; see manager.py _build_property_loss docstring.)"""
    config = MFTConfigManager(_FakePropertyTuner()).build()
    loss = config["loss_dict"]["property"]
    assert "property_name" not in loss


# ---------------------------------------------------------------------------
# Property task: aux branch is unaffected
# ---------------------------------------------------------------------------

def test_property_task_aux_branch_keeps_ener_fitting_net():
    """The aux branch (SPICE2 force-field) must keep its ener fitting_net.
    Only DOWNSTREAM gets the new property head."""
    config = MFTConfigManager(_FakePropertyTuner()).build()
    aux_fn = config["model"]["model_dict"]["SPICE2"]["fitting_net"]
    assert aux_fn["type"] == "ener"
    assert aux_fn == {"type": "ener", "neuron": [240, 240, 240]}


def test_property_task_aux_branch_keeps_ener_loss():
    """The aux branch loss must remain ener-style (it has forces+virials)."""
    config = MFTConfigManager(_FakePropertyTuner()).build()
    aux_loss = config["loss_dict"]["SPICE2"]
    assert aux_loss["type"] == "ener"
    assert "start_pref_f" in aux_loss


def test_property_task_extensive_property():
    """When intensive=False, the property head reflects that — extensive
    properties like total dipole moment use sum-pool."""
    class _T(_FakePropertyTuner):
        property_name = "total_dipole"
        intensive = False
    config = MFTConfigManager(_T()).build()
    fn = config["model"]["model_dict"]["property"]["fitting_net"]
    assert fn["intensive"] is False
    assert fn["property_name"] == "total_dipole"


def test_property_task_multidim_task_dim():
    """task_dim > 1 is honored (e.g. multitask HOMO+LUMO regression)."""
    class _T(_FakePropertyTuner):
        task_dim = 2
        property_name = "homo_lumo"
    config = MFTConfigManager(_T()).build()
    fn = config["model"]["model_dict"]["property"]["fitting_net"]
    assert fn["task_dim"] == 2


# ---------------------------------------------------------------------------
# Back-compat: ener mode is unchanged
# ---------------------------------------------------------------------------

def test_ener_task_unchanged_when_no_attr():
    """Tuners without downstream_task_type attr (existing mp_data callers)
    must still get the legacy ener-mode config: DOWNSTREAM reuses the aux
    fitting_net and gets an ener loss with force/virial prefs."""
    config = MFTConfigManager(_FakeEnerTuner()).build()
    md = config["model"]["model_dict"]
    # DOWNSTREAM fitting_net == aux fitting_net (the legacy behavior)
    assert md["DOWNSTREAM"]["fitting_net"] == md["MP_traj_v024_alldata_mixu"]["fitting_net"]
    assert md["DOWNSTREAM"]["fitting_net"]["type"] == "ener"
    # ener loss with force/virial prefs
    loss = config["loss_dict"]["DOWNSTREAM"]
    assert loss["type"] == "ener"
    assert loss["start_pref_f"] == 100
    assert loss["start_pref_v"] == 0.02


def test_ener_task_explicit_attr_unchanged():
    """Explicitly setting downstream_task_type='ener' is equivalent to
    not setting it at all."""
    t = _FakeEnerTuner()
    t.downstream_task_type = "ener"
    config = MFTConfigManager(t).build()
    md = config["model"]["model_dict"]
    assert md["DOWNSTREAM"]["fitting_net"]["type"] == "ener"
    assert config["loss_dict"]["DOWNSTREAM"]["type"] == "ener"


# ---------------------------------------------------------------------------
# MFTFineTuner.__init__: argument validation
# ---------------------------------------------------------------------------

def test_property_task_requires_property_name(monkeypatch):
    """downstream_task_type='property' without property_name must raise."""
    import torch

    monkeypatch.setattr(
        torch, "load",
        lambda *a, **kw: {
            "model": {
                "_extra_state": {
                    "model_params": {
                        "model_dict": {"SPICE2": {"fitting_net": {"type": "ener"}}}
                    }
                }
            }
        },
    )
    with pytest.raises(ValueError, match="property_name"):
        MFTFineTuner(
            pretrained="/does/not/exist.pt",
            aux_branch="SPICE2",
            downstream_task_type="property",
            # property_name omitted on purpose
        )


def test_property_task_property_name_must_be_identifier(monkeypatch):
    """property_name with slashes/spaces is rejected."""
    import torch

    monkeypatch.setattr(
        torch, "load",
        lambda *a, **kw: {
            "model": {
                "_extra_state": {
                    "model_params": {
                        "model_dict": {"SPICE2": {"fitting_net": {"type": "ener"}}}
                    }
                }
            }
        },
    )
    with pytest.raises(ValueError, match="property_name"):
        MFTFineTuner(
            pretrained="/does/not/exist.pt",
            aux_branch="SPICE2",
            downstream_task_type="property",
            property_name="homo lumo",  # invalid identifier
        )


def test_invalid_downstream_task_type_raises(monkeypatch):
    """Typos like 'properties' or 'energy' must raise immediately."""
    import torch

    monkeypatch.setattr(
        torch, "load",
        lambda *a, **kw: {
            "model": {
                "_extra_state": {
                    "model_params": {
                        "model_dict": {"SPICE2": {"fitting_net": {"type": "ener"}}}
                    }
                }
            }
        },
    )
    with pytest.raises(ValueError, match="downstream_task_type"):
        MFTFineTuner(
            pretrained="/does/not/exist.pt",
            aux_branch="SPICE2",
            downstream_task_type="properties",  # typo
        )


def test_property_task_stores_attrs(monkeypatch):
    """The MFTFineTuner exposes downstream_task_type / property_name /
    task_dim / intensive so MFTConfigManager can read them."""
    import torch

    monkeypatch.setattr(
        torch, "load",
        lambda *a, **kw: {
            "model": {
                "_extra_state": {
                    "model_params": {
                        "model_dict": {"SPICE2": {"fitting_net": {"type": "ener"}}}
                    }
                }
            }
        },
    )
    t = MFTFineTuner(
        pretrained="/does/not/exist.pt",
        aux_branch="SPICE2",
        downstream_task_type="property",
        property_name="lumo",
        task_dim=1,
        intensive=True,
    )
    assert t.downstream_task_type == "property"
    assert t.property_name == "lumo"
    assert t.task_dim == 1
    assert t.intensive is True


def test_ener_default_when_unspecified(monkeypatch):
    """Back-compat: not passing downstream_task_type defaults to 'ener'."""
    import torch

    monkeypatch.setattr(
        torch, "load",
        lambda *a, **kw: {
            "model": {
                "_extra_state": {
                    "model_params": {
                        "model_dict": {"Foo": {"fitting_net": {"type": "ener"}}}
                    }
                }
            }
        },
    )
    t = MFTFineTuner(pretrained="/does/not/exist.pt", aux_branch="Foo")
    assert t.downstream_task_type == "ener"
    assert t.property_name is None
