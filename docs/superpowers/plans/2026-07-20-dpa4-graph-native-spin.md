# DPA4 Native Spin on the NeighborGraph Route — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the pt-native spin scheme (`SeZMNativeSpinModel`, spin `scheme: "native"`) to dpmodel + pt_expt as `DPA4NativeSpinModel`, riding EXCLUSIVELY the NeighborGraph (graph) lower: eager energy/force/force_mag in pt_expt, graph-kind `.pt2` freeze, C++ `DeepSpinPTExpt` graph route, and a single-rank LAMMPS `pair_style deepspin` test.

**Architecture:** Spin is a per-LOCAL-atom `(nf, nloc, 3)` input that (a) conditions the descriptor (`spin_embedding`: l=0 magnitude into the type embedding, l=1 direction into the backbone and per-edge source-spin features) and (b) is a second autograd leaf: `force_mag = -dE/dspin`, computed in the same graph autograd assembly that already produces force/virial from `edge_vec`. Ghost spin is NEVER needed: the NeighborGraph is owner-folded (`src = mapping[neighbor] ∈ [0, nloc)`), so per-local-atom spin fully determines every edge's source spin. The descriptor trunk (`_call_graph_common → _call_graph_impl`) already consumes `spin`; this plan threads it through the model-level graph seam (which currently drops it), adds the wrapper model class, and extends the graph `.pt2` ABI with one positional `spin` tensor.

**Tech Stack:** dpmodel (array_api_compat numpy/torch), pt_expt (`@torch_module`, make_fx/AOTI), C++ AOTI runner (`DeepSpinPTExpt.cc`), LAMMPS plugin, pytest + gtest.

## Global Constraints

- **Branch:** commit directly onto `feat-graph-dpa4` (user decision 2026-07-20). Commit with `git commit --no-verify`; run `ruff check` / `ruff format` manually. **Never** add a coauthor line or "Generated with Claude Code" anywhere.
- **`deepmd/pt/` is READ-ONLY** — parity reference only; never modify.
- **Naming (user decision 2026-07-20):** class `DPA4NativeSpinModel`, files `dpa4_native_spin_model.py`, registered/serialized type `"dpa4_native_spin"`; `"sezm_native_spin"` (the pt wire string) accepted as a deserialize ALIAS. Consistent with `DescrptDPA4` / `"dpa4_ener"` naming.
- **Spin rides ONLY the graph route at model level.** The model-level dense (nlist) lower never receives spin; `call_common` raises `NotImplementedError` if `spin is not None` and the graph route is not taken. The descriptor-level dense adapter (`DescrptDPA4.call(..., spin=)`) keeps its existing spin support (it routes through the shared graph trunk and is the parity reference).
- **Single-rank only.** Graph-kind spin `.pt2` carries NO with-comm artifact (`has_comm_artifact=False`); C++ multi-rank on a graph-kind spin model fails fast. Multi-rank spin-graph is a documented follow-up.
- **Out of scope (Plan A, separate):** charge-spin FiLM and SFPG bridging on the graph route. A native-spin model with `add_chg_spin_ebd=True` is rejected at build in this plan.
- **Fresh DPA4 is edge/spin-INDEPENDENT** (zero-init output projections): every parity/sensitivity test MUST jitter weights via `source/tests/dpa4_fixtures.py:jitter_zero_arrays` or it is vacuous. Verify each new guard has teeth (break it once, see it fail).
- **Tolerances:** weight-copied fp64 parity vs pt: rtol/atol 1e-12 (CPU); finite-difference force/force_mag: atol 1e-6 (mirror pt test); eager-vs-.pt2: 1e-10. NEVER loosen a tolerance to pass; never skip a test to bypass a bug.
- **Tests:** pytest with `@pytest.mark.parametrize` (one param per line + trailing comment), `setup_method`, classes without `unittest.TestCase` where the harness allows. pt_expt export-tracing tests run the model on CPU (make_fx tracing is CPU-only by design); artifact-input tests use `_env.DEVICE`. numpydoc docstrings (underlined section headers); no free-form `See Also`.
- **Read `doc/development/testing.md` before writing/running tests.**
- The graph-spin `.pt2` positional ABI (this plan's contract, owner Task 5):
  `atype(N,), n_node(nf,), n_local(nf,), edge_index(2,E), edge_vec(E,3), edge_mask(E,), destination_order(E,), destination_row_ptr(N+1,), source_order(E,), source_row_ptr(N+1,), spin(N,3), [fparam(nf,ndf) if dim_fparam>0], [aparam(N,nda) if dim_aparam>0]` — spin is index 10, ALWAYS present for spin artifacts, before the conditional tail. Outputs (dict order): `atom_energy(N,1), energy(nf,1), force(N,3), force_mag(N,3), virial(nf,9), [atom_virial(N,9)]`.

## Responsibility Matrix (one execution site per responsibility)

| Responsibility | ONE owning site | Pinned by |
|---|---|---|
| spin input canonicalization (shape/dtype cast to `(nf,nloc,3)` compute precision) | `DPA4NativeSpinModel.call`/`forward` (wrapper, per backend) | Task 2/4 tests |
| spin flattening to node axis `(N,3)` | `_call_common_graph` (same place aparam flattens) | Task 1/3 |
| spin autograd-leaf creation | pt_expt `forward_common_lower_graph` (same site as the `edge_vec` leaf, make_model.py:614) | Task 3 |
| `energy_derv_r_mag = -dE/dspin` | pt_expt `fit_output_to_model_output_graph` (same backward family as force/virial) | Task 3 FD test |
| descriptor spin consumption | `_call_graph_impl` (UNCHANGED — already implemented) | existing golden tests |
| `mask_mag` | wrapper model (`spin_mask[atype] > 0`) | Task 2/4 |
| model-level dense-lower spin rejection | dpmodel `call_common` (before dense dispatch) | Task 2 negative test |
| graph `.pt2` spin ABI (index 10) | `DPA4NativeSpinModel.forward_lower_graph_exportable` (pt_expt); serialization/deep_eval/C++ MIRROR it | Task 5/6/7/9 |
| multi-rank rejection | metadata `has_comm_artifact=False` (freeze) + C++ dispatch fail-fast | Task 6/9 |

## File Structure

- `deepmd/dpmodel/descriptor/dpa4.py` — `call_graph` gains `spin`; `uses_graph_lower` stops gating on `spin_embedding`.
- `deepmd/dpmodel/atomic_model/dp_atomic_model.py` — `forward_atomic_graph` gains + forwards `spin`.
- `deepmd/dpmodel/model/make_model.py` — `call_common`, `_call_common_graph`, `call_common_lower_graph` gain `spin`; dense-route spin rejection.
- `deepmd/dpmodel/model/dpa4_native_spin_model.py` — NEW: dpmodel wrapper (energy-only outputs + mask_mag).
- `deepmd/dpmodel/model/model.py` + `base_model.py` — builder dispatch + deserialize routing (`dpa4_native_spin`, alias `sezm_native_spin`).
- `deepmd/pt_expt/model/make_model.py` — `forward_common_lower_graph` + `_call_common_graph` gain `spin` (leaf creation).
- `deepmd/pt_expt/model/edge_transform_output.py` — `fit_output_to_model_output_graph` gains `spin_leaf` → emits `energy_derv_r_mag`.
- `deepmd/pt_expt/model/dpa4_native_spin_model.py` — NEW: pt_expt wrapper + `forward_lower_graph_exportable` (ABI owner).
- `deepmd/pt_expt/model/get_model.py` + `model/__init__.py` — native-scheme dispatch replacing the blanket spin rejection.
- `deepmd/pt_expt/utils/serialization.py` — graph-kind spin freeze (detection, sample inputs, dynamic shapes, metadata, no with-comm).
- `deepmd/pt_expt/infer/deep_eval.py` — `_eval_model_spin` graph branch.
- `source/tests/infer/gen_dpa4_spin.py` — NEW fixture generator (`deeppot_dpa4_spin_graph.{yaml,pt2,expected}`).
- `source/api_cc/src/DeepSpinPTExpt.cc` (+ `include/commonPT.h` spin remap helper) — C++ graph route.
- `source/api_cc/tests/test_deepspin_dpa4_graph_ptexpt.cc` — NEW gtest.
- `source/lmp/tests/test_lammps_dpa4_spin_graph_pt2.py` — NEW LAMMPS test.
- `doc/model/dpa4.md` — native-spin section.
- Tests: `source/tests/common/dpmodel/test_dpa4_call_graph.py` (extend), `source/tests/common/dpmodel/test_dpa4_native_spin_model.py` (NEW), `source/tests/pt_expt/model/test_dpa4_native_spin.py` (NEW), `source/tests/pt_expt/model/test_dpa4_export.py` (extend).

---

### Task 1: dpmodel — thread `spin` through the graph lower chain + gate flip

**Files:**
- Modify: `deepmd/dpmodel/descriptor/dpa4.py` (`call_graph` ~line 1672; `uses_graph_lower` ~line 2321)
- Modify: `deepmd/dpmodel/atomic_model/dp_atomic_model.py` (`forward_atomic_graph` ~line 297)
- Modify: `deepmd/dpmodel/model/make_model.py` (`call_common_lower_graph` ~line 723, `forward_common_atomic_graph` ~line 647)
- Test: `source/tests/common/dpmodel/test_dpa4_call_graph.py`

**Interfaces:**
- Consumes: existing `_call_graph_common(graph, atype_flat, *, nf, n_out_nodes, force_embedding, charge_spin, spin, comm_dict)` (already accepts spin; UNCHANGED).
- Produces: `DescrptDPA4.call_graph(graph, atype, type_embedding=None, comm_dict=None, spin=None)`; `forward_atomic_graph(..., spin=None)`; `call_common_lower_graph(..., spin=None)` — `spin` is flat `(N, 3)` on the node axis everywhere below the wrapper. `uses_graph_lower()` no longer returns False for `spin_embedding is not None`.

- [ ] **Step 1: Write the failing tests** — append to `source/tests/common/dpmodel/test_dpa4_call_graph.py` (reuse that file's existing fixtures/builders; it already imports `jitter_zero_arrays` and builds jittered descriptors and graphs):

```python
def _make_spin_descriptor(self):
    """Jittered DPA4 with spin_embedding enabled (use_spin on type 0)."""
    dd = self._make_descriptor(use_spin=[True, False])  # extend the file's builder to pass use_spin through to DescrptDPA4
    jitter_zero_arrays(dd, seed=7)
    return dd

def test_call_graph_spin_sensitivity(self):
    """call_graph(spin=...) must change the output (teeth: spin reaches the trunk)."""
    dd = self._make_spin_descriptor()
    graph, atype_flat = self._make_graph()   # file's existing jittered-graph helper
    rng = np.random.default_rng(3)
    spin = rng.normal(size=(atype_flat.shape[0], 3))
    out0, _ = dd.call_graph(graph, atype_flat)
    out1, _ = dd.call_graph(graph, atype_flat, spin=spin)
    assert not np.allclose(out0, out1)

def test_call_graph_spin_matches_dense_adapter(self):
    """Graph-lower spin path == dense adapter spin path (shared trunk, 1e-12)."""
    dd = self._make_spin_descriptor()
    # dense inputs from the file's TestCaseSingleFrameWithNlist-style fixture
    coord_ext, atype_ext, nlist, mapping = self._dense_inputs()
    nf, nloc, _ = nlist.shape
    rng = np.random.default_rng(3)
    spin = rng.normal(size=(nf * nloc, 3))
    ref, *_ = dd.call(coord_ext, atype_ext, nlist, mapping=mapping,
                      spin=spin.reshape(nf, nloc, 3))
    graph, atype_flat = dd._graph_from_padded_nlist(  # or the file's equivalent adapter helper
        np.reshape(coord_ext, (nf, -1, 3)), atype_ext, nlist, mapping
    )
    out, _ = dd.call_graph(graph, atype_flat, spin=spin)
    np.testing.assert_allclose(
        out.reshape(nf, nloc, -1), ref, rtol=1e-12, atol=1e-12
    )
```

And FLIP the existing gate expectation in `test_uses_graph_lower_feature_gates` (currently asserts a non-None `spin_embedding` disables the graph lower, ~line 270-278): spin_embedding must now leave `uses_graph_lower() is True`; charge_spin_embedding and bridging_switch still disable it.

- [ ] **Step 2: Run tests, verify failures**

Run: `python -m pytest source/tests/common/dpmodel/test_dpa4_call_graph.py -k "spin or feature_gates" -x -q`
Expected: `test_call_graph_spin_sensitivity` FAILS (`call_graph() got an unexpected keyword argument 'spin'`); gate test FAILS on the flipped assertion.

- [ ] **Step 3: Implement**

`dpa4.py` — `call_graph` (add param + forward; extend docstring Parameters):

```python
def call_graph(
    self,
    graph: NeighborGraph,
    atype: Array,
    type_embedding: Array | None = None,
    comm_dict: dict[str, Array] | None = None,
    spin: Array | None = None,
) -> tuple[Array, None]:
    ...
    n_nodes = atype.shape[0]
    x_scalar, _ = self._call_graph_common(
        graph, atype, spin=spin, comm_dict=comm_dict
    )
```

Docstring addition for `spin`: "Per-node spin vectors with shape (N, 3) on the flat node axis, or None. Consumed by ``spin_embedding`` (l=0 magnitude into the type embedding, l=1 into the backbone and per-edge source features). Ghost-free graphs need only per-local-atom spin."

`dpa4.py` — `uses_graph_lower`: DELETE the two lines
```python
        if self.spin_embedding is not None:
            return False
```
and update its docstring (spin no longer rides only the dense signature; charge/spin FiLM and SFPG bridging still do).

`dp_atomic_model.py` — `forward_atomic_graph`: add `spin: Array | None = None` parameter (after `charge_spin`), document it ("flat (N, 3) per-node spin, forwarded to the descriptor's ``call_graph``; None for spin-less models"), and forward it:

```python
gg, rot_mat = self.descriptor.call_graph(
    graph, atype, type_embedding=type_embedding, comm_dict=comm_dict, spin=spin
)
```

`make_model.py` (dpmodel) — `forward_common_atomic_graph` (~line 647) and `call_common_lower_graph` (~line 723): add `spin: Array | None = None` to both signatures (after `charge_spin`), type-cast it alongside `charge_spin` in `_input_type_cast` usage (~line 778 — add `spin` to the cast tuple the same way `charge_spin` is cast), and forward down: `forward_common_atomic_graph(..., spin=spin)` → `self.atomic_model.forward_common_atomic_graph(graph, atype, fparam=..., aparam=..., charge_spin=..., spin=spin, ...)` — note the atomic-model wrapper chain (`base_atomic_model`/`make_base_atomic_model` graph forwards, if they intermediate) must pass `spin` through unchanged; grep `forward_common_atomic_graph` and thread every layer.

- [ ] **Step 4: Run tests**

Run: `python -m pytest source/tests/common/dpmodel/test_dpa4_call_graph.py source/tests/common/dpmodel/test_descrpt_dpa4.py -q`
Expected: ALL PASS (including pre-existing golden pins — spin=None default must be value-neutral).

- [ ] **Step 5: Teeth check + commit**

Temporarily drop `spin=spin` from the `call_graph` forward, confirm `test_call_graph_spin_sensitivity` fails, restore.

```bash
git add deepmd/dpmodel/descriptor/dpa4.py deepmd/dpmodel/atomic_model/dp_atomic_model.py deepmd/dpmodel/model/make_model.py source/tests/common/dpmodel/test_dpa4_call_graph.py
git commit --no-verify -m "feat(dpmodel): thread native spin through the DPA4 graph lower chain"
```

---

### Task 2: dpmodel — `DPA4NativeSpinModel` wrapper + builder dispatch + dense rejection

**Files:**
- Create: `deepmd/dpmodel/model/dpa4_native_spin_model.py`
- Modify: `deepmd/dpmodel/model/make_model.py` (`call_common` ~line 272: `spin` param + graph forwarding + dense rejection; `_call_common_graph` ~line 439: `spin` param + flatten)
- Modify: `deepmd/dpmodel/model/model.py` (builder + `get_model` dispatch)
- Modify: `deepmd/dpmodel/model/base_model.py` (~line 130: deserialize special-case)
- Test: `source/tests/common/dpmodel/test_dpa4_native_spin_model.py` (NEW)

**Interfaces:**
- Consumes: Task 1's `call_common_lower_graph(..., spin=None)`; `Spin` from `deepmd.utils.spin`; `jitter_zero_arrays` from `source/tests/dpa4_fixtures.py`.
- Produces:
  - `make_model.call_common(..., spin: Array | None = None, ...)` — graph route forwards spin; dense route raises `NotImplementedError("model-level spin rides only the NeighborGraph lower; ...")` when `spin is not None`.
  - `_call_common_graph(cc, atype, bb, fp, ap, method, do_atomic_virial=False, spin=None)` — flattens `(nf,nloc,3)` → `(N,3)` and forwards.
  - `class DPA4NativeSpinModel(NativeOP)` with `__init__(self, backbone_model, spin: Spin)`, `call(coord, atype, spin, box=None, fparam=None, aparam=None, do_atomic_virial=False)`, `model_output_def()`, `serialize()/deserialize()` (type `"dpa4_native_spin"`), attribute pass-throughs (`get_rcut`, `get_type_map`, `get_dim_fparam`, `get_dim_aparam`, `get_sel_type`, `has_message_passing`, `atomic_output_def`, ... — mirror the delegation set of `deepmd/dpmodel/model/spin_model.py`).
  - `get_dpa4_native_spin_model(data) -> DPA4NativeSpinModel` in `model.py`; `get_model` routes `"spin" in data and data["spin"].get("scheme", "deepspin") == "native"` (descriptor type must be dpa4/sezm family, else `NotImplementedError`); `get_spin_model` gains a guard rejecting dpa4/sezm descriptors (virtual-atom scheme unsupported for SeZM).
  - `BaseModel.deserialize` routes type strings `"dpa4_native_spin"` AND `"sezm_native_spin"` to `DPA4NativeSpinModel.deserialize`.

- [ ] **Step 1: Write the failing tests** — `source/tests/common/dpmodel/test_dpa4_native_spin_model.py`:

```python
import numpy as np
import pytest

from deepmd.dpmodel.model.model import get_model

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from dpa4_fixtures import jitter_zero_arrays

NATIVE_SPIN_CONFIG = {
    "type_map": ["Ni", "O"],
    "descriptor": {
        "type": "dpa4",
        "rcut": 4.0,
        # copy the remaining minimal DPA4 keys from test_dpa4_call_graph.py's builder
    },
    "fitting_net": {"type": "dpa4_ener", "neuron": [8, 8]},
    "spin": {"use_spin": [True, False], "scheme": "native"},
}


class TestDPA4NativeSpinModel:
    def setup_method(self):
        self.model = get_model(NATIVE_SPIN_CONFIG)
        jitter_zero_arrays(self.model, seed=11)
        rng = np.random.default_rng(5)
        self.nf, self.nloc = 1, 6
        self.coord = rng.uniform(0.5, 5.5, size=(self.nf, self.nloc, 3))
        self.atype = np.array([[0, 0, 1, 0, 1, 1]], dtype=np.int64)
        self.spin = rng.normal(size=(self.nf, self.nloc, 3))
        self.box = 8.0 * np.eye(3, dtype=np.float64)[None]

    def test_call_returns_energy_and_mask_mag(self):
        out = self.model.call(self.coord, self.atype, self.spin, box=self.box)
        assert out["energy"].shape == (self.nf, 1)
        assert out["atom_energy"].shape == (self.nf, self.nloc, 1)
        np.testing.assert_array_equal(
            out["mask_mag"][..., 0], self.atype == 0
        )  # use_spin=[True, False]

    def test_spin_sensitivity(self):
        out0 = self.model.call(self.coord, self.atype, self.spin, box=self.box)
        out1 = self.model.call(self.coord, self.atype, 2.0 * self.spin, box=self.box)
        assert not np.allclose(out0["energy"], out1["energy"])

    def test_serialize_roundtrip(self):
        data = self.model.serialize()
        assert data["type"] == "dpa4_native_spin"
        from deepmd.dpmodel.model.base_model import BaseModel
        model2 = BaseModel.deserialize(data)
        out0 = self.model.call(self.coord, self.atype, self.spin, box=self.box)
        out1 = model2.call(self.coord, self.atype, self.spin, box=self.box)
        np.testing.assert_allclose(out0["energy"], out1["energy"], rtol=1e-12)

    def test_sezm_native_spin_alias_deserializes(self):
        data = self.model.serialize()
        data["type"] = "sezm_native_spin"  # pt wire string
        from deepmd.dpmodel.model.base_model import BaseModel
        model2 = BaseModel.deserialize(data)
        out1 = model2.call(self.coord, self.atype, self.spin, box=self.box)
        np.testing.assert_allclose(
            self.model.call(self.coord, self.atype, self.spin, box=self.box)["energy"],
            out1["energy"], rtol=1e-12,
        )

    def test_dense_route_spin_raises(self):
        with pytest.raises(NotImplementedError, match="NeighborGraph"):
            self.model.backbone_model.call_common(
                self.coord, self.atype, self.box,
                spin=self.spin, neighbor_graph_method="legacy",
            )

    def test_deepspin_scheme_with_dpa4_raises(self):
        cfg = {**NATIVE_SPIN_CONFIG, "spin": {"use_spin": [True, False]}}
        with pytest.raises(NotImplementedError):
            get_model(cfg)
```

(Adjust the DPA4 descriptor minimal keys by copying the smallest working config from `test_dpa4_call_graph.py` / `test_descrpt_dpa4.py`; the descriptor must receive `use_spin` — mirror how the pt builder `_get_sezm_native_spin_model` injects it into the descriptor params.)

- [ ] **Step 2: Run tests, verify failure**

Run: `python -m pytest source/tests/common/dpmodel/test_dpa4_native_spin_model.py -x -q`
Expected: FAIL — native scheme not dispatched (`get_model` builds a virtual SpinModel or crashes).

- [ ] **Step 3: Implement `make_model.py` spin on the high-level `call_common`**

Add `spin: Array | None = None` to `call_common`'s signature (documented: "(nf, nloc, 3) per-local-atom spin; only the graph route consumes it"). After the existing `graph_method` resolution (the `cs is not None` block ends ~line 373), add:

```python
            # model-level spin rides ONLY the NeighborGraph lower
            if spin is not None and graph_method is None:
                raise NotImplementedError(
                    "model-level spin rides only the NeighborGraph lower; the "
                    "dense (nlist) route has no spin support -- use a graph "
                    "neighbor_graph_method"
                )
            if graph_method is not None:
                model_predict = self._call_common_graph(
                    cc, atype, bb, fp, ap, graph_method, do_atomic_virial,
                    spin=sp,
                )
```

where `sp` is the input-precision-cast spin (cast alongside `cc`/`fp`/`ap` in `_input_type_cast`). In `_call_common_graph` add `spin: Array | None = None` and flatten + forward:

```python
            model_predict = self.call_lower_graph(
                ...,
                aparam=(...),
                spin=(
                    xp.reshape(spin, (nf * nloc, 3)) if spin is not None else None
                ),
            )
```

Mirror the same `spin=None` parameter + forward in the pt_expt override `deepmd/pt_expt/model/make_model.py:_call_common_graph` (line 709) — forwarding into `forward_common_lower_graph(..., spin=spin_flat)` — BUT leave `forward_common_lower_graph` itself unchanged until Task 3 (pass spin only if not None to keep Task 2 green: `**({"spin": spin_flat} if spin_flat is not None else {})`). Simpler alternative: do the pt_expt `_call_common_graph` threading entirely in Task 3; in that case dpmodel-only here.

- [ ] **Step 4: Implement the wrapper class** — `deepmd/dpmodel/model/dpa4_native_spin_model.py` (mirror `spin_model.py` structure; key parts):

```python
import numpy as np

from deepmd.dpmodel.common import NativeOP
from deepmd.dpmodel.output_def import ModelOutputDef
from deepmd.utils.spin import Spin


class DPA4NativeSpinModel(NativeOP):
    """DPA4/SeZM native-spin model on the NeighborGraph route.

    Wraps a standard DPA4 energy backbone; ``spin`` is a per-local-atom
    (nf, nloc, 3) input consumed by the descriptor's ``spin_embedding``
    (never by virtual atoms). dpmodel produces energy + mask_mag only;
    force/force_mag come from autograd in derived backends (pt_expt).
    """

    def __init__(self, backbone_model, spin: Spin) -> None:
        super().__init__()
        self.backbone_model = backbone_model
        self.spin = spin
        self.spin_mask = self.spin.get_spin_mask()  # (ntypes_real,) 0/1

    def model_output_def(self) -> ModelOutputDef:
        backbone_def = self.backbone_model.atomic_output_def()
        backbone_def["energy"].magnetic = True
        return ModelOutputDef(backbone_def)

    def call(self, coord, atype, spin, box=None, fparam=None, aparam=None,
             do_atomic_virial=False):
        nf, nloc = atype.shape[:2]
        model_ret = self.backbone_model.call_common(
            coord, atype, box=box, fparam=fparam, aparam=aparam,
            do_atomic_virial=do_atomic_virial, spin=spin,
            neighbor_graph_method="dense",  # dpmodel: opt into the carry-all graph
        )
        out = {
            "atom_energy": model_ret["energy"],
            "energy": model_ret["energy_redu"],
            "mask_mag": (self.spin_mask[atype] > 0)[..., None],
        }
        # derivative name-holders (dpmodel graph route is energy-only)
        for kk_src, kk_dst in (
            ("energy_derv_r", "force"),
            ("energy_derv_r_mag", "force_mag"),
            ("energy_derv_c_redu", "virial"),
        ):
            if model_ret.get(kk_src) is not None:
                out[kk_dst] = np.squeeze(model_ret[kk_src], axis=-2)
            else:
                out[kk_dst] = None
        return out

    def serialize(self) -> dict:
        return {
            "@class": "Model",
            "@version": 1,
            "type": "dpa4_native_spin",
            "backbone_model": self.backbone_model.serialize(),
            "spin": self.spin.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DPA4NativeSpinModel":
        from deepmd.dpmodel.model.dp_model import DPModelCommon  # match spin_model.py's backbone rebuild
        data = data.copy()
        data.pop("@class", None)
        data.pop("type", None)
        spin = Spin.deserialize(data.pop("spin"))
        backbone_model = _rebuild_backbone(data.pop("backbone_model"))  # mirror spin_model.py deserialize exactly
        return cls(backbone_model=backbone_model, spin=spin)
```

Add the delegation methods (`get_rcut`, `get_type_map`, `get_ntypes`, `get_dim_fparam`, `get_dim_aparam`, `get_sel_type`, `get_model_def_script`, `has_message_passing`, `atomic_output_def`, `get_nnei`, `get_nsel`, ...) by copying the delegation block of `spin_model.py` verbatim minus the virtual-atom-specific ones. Copy `spin_model.py`'s backbone deserialize mechanics for `_rebuild_backbone`.

Builder in `model.py`:

```python
def get_dpa4_native_spin_model(data: dict) -> "DPA4NativeSpinModel":
    data = copy.deepcopy(data)
    if data["descriptor"]["type"] not in ("dpa4", "DPA4", "sezm", "SeZM"):
        raise NotImplementedError(
            "spin scheme 'native' requires the DPA4/SeZM descriptor"
        )
    spin_cfg = data.pop("spin")
    spin = Spin(
        use_spin=spin_cfg["use_spin"],
        virtual_scale=spin_cfg.get("virtual_scale", 1.0),
    )
    data["descriptor"]["use_spin"] = spin_cfg["use_spin"]
    if data["descriptor"].get("add_chg_spin_ebd", False):
        raise NotImplementedError(
            "charge-spin FiLM with native spin on the graph route is a follow-up"
        )
    backbone_model = get_standard_model(data)
    return DPA4NativeSpinModel(backbone_model=backbone_model, spin=spin)
```

`get_model` dispatch (replace the `if "spin" in data:` line):

```python
        if "spin" in data:
            if data["spin"].get("scheme", "deepspin") == "native":
                return get_dpa4_native_spin_model(data)
            return get_spin_model(data)
```

and in `get_spin_model` add at top:

```python
    if data["descriptor"]["type"] in ("dpa4", "DPA4", "sezm", "SeZM"):
        raise NotImplementedError(
            "the virtual-atom (deepspin) scheme is not supported for DPA4/SeZM; "
            "use spin scheme 'native'"
        )
```

`base_model.py` (~line 130, next to the `"spin_ener"` branch):

```python
        elif data.get("type") in ("dpa4_native_spin", "sezm_native_spin"):
            from deepmd.dpmodel.model.dpa4_native_spin_model import (
                DPA4NativeSpinModel,
            )
            return DPA4NativeSpinModel.deserialize(data)
```

(Whole-model conversion of pt-SERIALIZED `sezm_native_spin` dicts — pt's `atomic_model` structure — is NOT claimed; the alias covers the type string with dpmodel structure. Record as a known limitation.)

- [ ] **Step 5: Run tests**

Run: `python -m pytest source/tests/common/dpmodel/test_dpa4_native_spin_model.py source/tests/common/dpmodel/test_dpa4_call_graph.py -q`
Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add deepmd/dpmodel/model/ source/tests/common/dpmodel/test_dpa4_native_spin_model.py
git commit --no-verify -m "feat(dpmodel): DPA4NativeSpinModel on the NeighborGraph route"
```

---

### Task 3: pt_expt — `force_mag` autograd in the graph assembly

**Files:**
- Modify: `deepmd/pt_expt/model/make_model.py` (`forward_common_lower_graph` line 505; `_call_common_graph` line 709)
- Modify: `deepmd/pt_expt/model/edge_transform_output.py` (`fit_output_to_model_output_graph` line 149)
- Test: `source/tests/pt_expt/model/test_dpa4_native_spin.py` (NEW; FD + leaf mechanics)

**Interfaces:**
- Consumes: Task 1's `forward_common_atomic_graph(..., spin=...)` chain (shared dpmodel code called with torch tensors).
- Produces:
  - `forward_common_lower_graph(..., spin: torch.Tensor | None = None, ...)` — creates the spin autograd leaf next to the `edge_vec` leaf and passes `spin_leaf` down.
  - `fit_output_to_model_output_graph(..., spin_leaf: torch.Tensor | None = None)` — for every `r_differentiable` reducible output, when `spin_leaf is not None`, additionally emits `<var>_derv_r_mag` `(N, *shape, 3)` = `-d(<var>_redu)/d(spin)`.
  - pt_expt `_call_common_graph(..., spin=None)` — flattens and forwards (completes the Task 2 seam).

- [ ] **Step 1: Write the failing tests** — `source/tests/pt_expt/model/test_dpa4_native_spin.py`:

```python
import numpy as np
import pytest
import torch

from deepmd.pt_expt.utils import env as _env

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from dpa4_fixtures import jitter_zero_arrays

# build the pt_expt BACKBONE energy model from the same descriptor/fitting config
# as NATIVE_SPIN_CONFIG in the dpmodel test (no spin key -> standard model),
# via deepmd.pt_expt.model.get_model.get_model


def _finite_diff_mag(model_fn, spin, eps=1e-4):
    fm = np.zeros_like(spin)
    for i in np.ndindex(*spin.shape):
        sp = spin.copy(); sp[i] += eps; ep = model_fn(sp)
        sm = spin.copy(); sm[i] -= eps; em = model_fn(sm)
        fm[i] = -(ep - em) / (2 * eps)
    return fm


class TestGraphForceMag:
    def setup_method(self):
        # jittered pt_expt backbone with use_spin descriptor, moved to _env.DEVICE
        self.model = _build_jittered_backbone()  # helper in this file
        ...

    def test_force_mag_matches_finite_difference(self):
        """force_mag from the graph autograd == -dE/dspin by central FD (atol 1e-6)."""
        out = self.model.call_common(
            self.coord, self.atype, self.box, spin=self.spin,
            do_atomic_virial=False,
        )
        fm = out["energy_derv_r_mag"]
        fd = _finite_diff_mag(lambda sp: float(
            self.model.call_common(self.coord, self.atype, self.box,
                                   spin=torch.as_tensor(sp, device=self.coord.device),
                                   )["energy_redu"].sum()
        ), self.spin.cpu().numpy())
        np.testing.assert_allclose(
            fm.squeeze(-2).detach().cpu().numpy().reshape(fd.shape), fd, atol=1e-6
        )

    def test_force_unchanged_by_spin_leaf_wiring(self):
        """spin=None output (energy/force) is bit-identical to before the change."""
        out0 = self.model.call_common(self.coord, self.atype, self.box)
        assert "energy_derv_r_mag" not in out0
        # golden: compare energy against the committed no-spin golden value in
        # test_dpa4_call_graph.py-style pinning (reuse that pattern)
```

(Write `_build_jittered_backbone` to construct the pt_expt standard DPA4 model with `use_spin=[True, False]` in the descriptor config, `jitter_zero_arrays`, `.to(_env.DEVICE)`; keep the system tiny — 4-6 atoms — because FD loops over 3N components. Device-conditional tolerance is unnecessary at atol 1e-6.)

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest source/tests/pt_expt/model/test_dpa4_native_spin.py -x -q`
Expected: FAIL — `call_common() got an unexpected keyword argument 'spin'` resolved by Task 2's dpmodel change but `energy_derv_r_mag` missing (no mag grad yet).

- [ ] **Step 3: Implement**

`make_model.py` (pt_expt) `forward_common_lower_graph`: add `spin: torch.Tensor | None = None` (after `charge_spin`); create the leaf next to edge_vec's:

```python
            # make edge_vec the autograd leaf for the energy backward
            edge_vec = edge_vec.detach().requires_grad_(True)
            if spin is not None:
                # second autograd leaf: force_mag = -dE/dspin (native spin)
                spin = spin.detach().requires_grad_(True)
```

forward into the atomic call and the transform:

```python
            atomic_ret = self.atomic_model.forward_common_atomic_graph(
                graph, atype, fparam=fparam, aparam=aparam,
                charge_spin=charge_spin, spin=spin, comm_dict=comm_dict,
            )
            return fit_output_to_model_output_graph(
                ...,
                n_local=n_local,
                spin_leaf=spin,
            )
```

(The `cuda_infer_level() >= 2` fused path at line 629: guard it with `and spin is None` — the fused custom-op pipeline has no mag output.)

`edge_transform_output.py` `fit_output_to_model_output_graph`: add keyword `spin_leaf: torch.Tensor | None = None` (document: "per-node (N, 3) autograd leaf; when given, every r_differentiable reducible output additionally emits ``<var>_derv_r_mag = -d<var>_redu/dspin`` in the same backward family"). In the per-component loop (after the `edge_energy_deriv` call, ~line 299-313):

```python
        mag_list: list[torch.Tensor] = []
        for c in range(size):
            force, atom_vir, vir = edge_energy_deriv(...)
            ff_list.append(force.reshape(N, 1, 3))
            if spin_leaf is not None:
                (g_s,) = torch.autograd.grad(
                    svv[:, c].sum(),
                    spin_leaf,
                    create_graph=create_graph,
                    retain_graph=True,
                )
                mag_list.append((-g_s).reshape(N, 1, 3))
            ...
        model_ret[kk_derv_r] = torch.cat(ff_list, dim=-2).reshape([N, *shap, 3])
        if spin_leaf is not None:
            from deepmd.dpmodel.output_def import get_deriv_name_mag
            kk_derv_r_mag, _ = get_deriv_name_mag(kk)
            model_ret[kk_derv_r_mag] = torch.cat(mag_list, dim=-2).reshape(
                [N, *shap, 3]
            )
```

(NOTE and document the deliberate deviation from pt: pt computes `grad(E, [edge_vec, spin])` jointly in `transform_output.py:212`; here the mag grad is a second `autograd.grad` with `retain_graph=True` — same math, keeps `edge_energy_deriv`'s signature untouched.)

pt_expt `_call_common_graph` (line 709): add `spin: torch.Tensor | None = None`, flatten `(nf,nloc,3)` → `(N,3)`, forward `spin=spin_flat` into `forward_common_lower_graph` (completes Task 2's seam).

- [ ] **Step 4: Run tests**

Run: `python -m pytest source/tests/pt_expt/model/test_dpa4_native_spin.py source/tests/pt_expt/model/test_dpa4_graph_lower.py -q`
Expected: ALL PASS (existing graph-lower tests pin spin=None neutrality).

- [ ] **Step 5: Commit**

```bash
git add deepmd/pt_expt/model/make_model.py deepmd/pt_expt/model/edge_transform_output.py source/tests/pt_expt/model/test_dpa4_native_spin.py
git commit --no-verify -m "feat(pt_expt): force_mag autograd on the DPA4 graph lower"
```

---

### Task 4: pt_expt — `DPA4NativeSpinModel` wrapper + dispatch + pt parity

**Files:**
- Create: `deepmd/pt_expt/model/dpa4_native_spin_model.py`
- Modify: `deepmd/pt_expt/model/get_model.py` (`get_sezm_model` spin rejection at lines 143-146)
- Modify: `deepmd/pt_expt/model/__init__.py` (export)
- Test: extend `source/tests/pt_expt/model/test_dpa4_native_spin.py`

**Interfaces:**
- Consumes: dpmodel `DPA4NativeSpinModel` (Task 2), pt_expt backbone `call_common(..., spin=)` (Task 3), `@torch_module` from `deepmd/pt_expt/common.py`.
- Produces:
  - `@torch_module class DPA4NativeSpinModel(DPA4NativeSpinModelDP)` with `forward(coord, atype, spin, box=None, fparam=None, aparam=None, do_atomic_virial=False) -> dict[str, torch.Tensor]` returning `atom_energy, energy, force, force_mag, virial, [atom_virial], mask_mag` (real tensors — pt_expt graph route has autograd, so `force`/`force_mag` are NOT None-holders here; override the dpmodel `call` translation accordingly).
  - `get_sezm_model` builds it when `data["spin"]["scheme"] == "native"`; the deepspin-scheme rejection stays.
  - `model_type = "ener"`-style registration NOT needed; deserialize routes through the dpmodel base-model branch + a pt_expt mapping registered via `register_dpmodel_mapping` if the wrapper holds NativeOP children (follow how `pt_expt/model/spin_ener_model.py` handles its backbone conversion).

- [ ] **Step 1: Write the failing tests** — extend `test_dpa4_native_spin.py`:

```python
class TestDPA4NativeSpinModelPtExpt:
    def setup_method(self):
        from deepmd.pt_expt.model.get_model import get_model
        self.model = get_model(NATIVE_SPIN_CONFIG).to(_env.DEVICE)  # same config as dpmodel test
        jitter_zero_arrays(self.model, seed=11)
        ...

    def test_forward_keys_and_mask(self):
        out = self.model.forward(self.coord, self.atype, self.spin, box=self.box)
        for k in ("energy", "atom_energy", "force", "force_mag", "virial", "mask_mag"):
            assert k in out
        assert out["force_mag"].shape == (self.nf, self.nloc, 3)
        # zero force_mag on non-spin types (use_spin=[True, False])
        non_spin = (self.atype == 1)
        assert torch.all(out["force_mag"][non_spin] == 0)

    def test_parity_vs_pt_native_spin_model(self):
        """Weight-copied fp64 parity vs deepmd.pt SeZMNativeSpinModel (CPU, 1e-12)."""
        # build the pt model from the same config via
        # deepmd.pt.model.model.get_sezm_spin_model / _get_sezm_native_spin_model,
        # jitter the PT model, serialize its descriptor+fitting, deserialize into
        # the pt_expt model (component-wise weight copy, the established
        # test_dpa4_dpmodel_parity mechanism), run both on CPU, compare
        # energy / force / force_mag at rtol=atol=1e-12.
```

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest source/tests/pt_expt/model/test_dpa4_native_spin.py -k PtExpt -x -q`
Expected: FAIL — `get_model` raises `NotImplementedError("Spin DPA4/SeZM models are not supported...")`.

- [ ] **Step 3: Implement**

`deepmd/pt_expt/model/dpa4_native_spin_model.py`:

```python
import torch

from deepmd.dpmodel.model.dpa4_native_spin_model import (
    DPA4NativeSpinModel as DPA4NativeSpinModelDP,
)
from deepmd.pt_expt.common import torch_module


@torch_module
class DPA4NativeSpinModel(DPA4NativeSpinModelDP):
    """pt_expt native-spin DPA4 model (NeighborGraph route, autograd force_mag)."""

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        spin: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.backbone_model.call_common(
            coord, atype, box=box, fparam=fparam, aparam=aparam,
            do_atomic_virial=do_atomic_virial, spin=spin,
            # pt_expt default-flip resolves None -> carry-all graph for DPA4
        )
        spin_mask = torch.as_tensor(
            self.spin_mask, device=atype.device
        )
        out = {
            "atom_energy": model_ret["energy"],
            "energy": model_ret["energy_redu"],
            "force": model_ret["energy_derv_r"].squeeze(-2),
            "force_mag": model_ret["energy_derv_r_mag"].squeeze(-2),
            "virial": model_ret["energy_derv_c_redu"].squeeze(-2),
            "mask_mag": (spin_mask[atype] > 0).unsqueeze(-1),
        }
        if do_atomic_virial:
            out["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        return out
```

(Zero-out `force_mag` rows of non-spin types via `out["force_mag"] = out["force_mag"] * out["mask_mag"]` IF the descriptor does not already produce exactly-zero grads there — check against the pt behavior: pt's FD test asserts exact zeros come out of the math itself; mirror whatever pt does, do NOT double-mask if the grads are already zero. If masking is needed here, pt's `SeZMNativeSpinModel` does it the same way — copy its site.)

`get_model.py` — replace the blanket rejection in `get_sezm_model` (lines 143-146):

```python
    if "spin" in data:
        if data["spin"].get("scheme", "deepspin") != "native":
            raise NotImplementedError(
                "the virtual-atom (deepspin) scheme is not supported for "
                "DPA4/SeZM in the pt_expt backend; use spin scheme 'native'"
            )
        return _get_dpa4_native_spin_model(data)
```

with `_get_dpa4_native_spin_model` mirroring the dpmodel builder (Task 2) but constructing the pt_expt backbone (existing `get_sezm_model` body for the non-spin part) and returning the pt_expt `DPA4NativeSpinModel`. Export the class from `deepmd/pt_expt/model/__init__.py`.

- [ ] **Step 4: Run tests (full spin file + regression)**

Run: `python -m pytest source/tests/pt_expt/model/test_dpa4_native_spin.py source/tests/pt_expt/model/test_dpa4_graph_lower.py -q`
Expected: ALL PASS; parity at 1e-12.

- [ ] **Step 5: Commit**

```bash
git add deepmd/pt_expt/model/dpa4_native_spin_model.py deepmd/pt_expt/model/get_model.py deepmd/pt_expt/model/__init__.py source/tests/pt_expt/model/test_dpa4_native_spin.py
git commit --no-verify -m "feat(pt_expt): DPA4NativeSpinModel wrapper with graph-route force_mag"
```

---

### Task 5: pt_expt — graph-spin exportable (ABI owner) + trace tests

**Files:**
- Modify: `deepmd/pt_expt/model/dpa4_native_spin_model.py` (add `forward_lower_graph_exportable`)
- Modify: `deepmd/pt_expt/model/ener_model.py` ONLY if `_translate_energy_keys` is extended (preferred: spin-local translate helper in the new file — do NOT touch the energy translate)
- Test: extend `source/tests/pt_expt/model/test_dpa4_native_spin.py`

**Interfaces:**
- Consumes: `forward_common_lower_graph_exportable` pattern from `ener_model.py:444-588` (two-layer make_fx trace) and `make_model.py:974` (`forward_common_lower_graph_exportable`).
- Produces: `DPA4NativeSpinModel.forward_lower_graph_exportable(...)` tracing a closure with the **positional ABI from Global Constraints** (spin at index 10, then conditional fparam/aparam) and returning the output dict in order `atom_energy, energy, force, force_mag, virial, [atom_virial]` — a spin-aware local translate:

```python
def _translate_spin_energy_keys(model_ret, *, do_atomic_virial):
    out = {}
    out["atom_energy"] = model_ret["energy"]
    out["energy"] = model_ret["energy_redu"]
    out["force"] = model_ret["energy_derv_r"].squeeze(-2)
    out["force_mag"] = model_ret["energy_derv_r_mag"].squeeze(-2)
    out["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
    if do_atomic_virial:
        out["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
    return out
```

- [ ] **Step 1: Write the failing tests** (extend `test_dpa4_native_spin.py`):

```python
    def test_graph_lower_exportable_symbolic_trace(self):
        """make_fx traces the graph-spin closure on CPU; outputs include force_mag."""
        model = _build_native_spin_model_cpu()  # jittered, CPU (export tracing is CPU-only)
        traced, sample = _trace_spin_graph(model)  # helper: build 12-tensor sample per ABI, call forward_lower_graph_exportable
        out = traced(*sample)
        # eager reference through model.forward on the same physical system
        ...

    def test_graph_lower_exportable_torch_export(self):
        """torch.export.export succeeds with dynamic nedge dim."""
```

Mirror the `test_exportable`/`test_make_fx` pattern of the pt_expt test trio (simplest reference: `test_se_t.py`; graph-flavor reference: `test_dpa4_graph_lower.py::test_symbolic_trace`/`test_torch_export`).

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest source/tests/pt_expt/model/test_dpa4_native_spin.py -k exportable -x -q`
Expected: FAIL — `forward_lower_graph_exportable` not defined on the spin model.

- [ ] **Step 3: Implement** — copy `ener_model.py:forward_lower_graph_exportable` (lines 444-588) into `dpa4_native_spin_model.py` and modify EXACTLY these points:
  1. the traced closure signature and the `make_fx(...)` argument tuple insert `spin` at position 10: `(atype, n_node, n_local, edge_index, edge_vec, edge_mask, destination_order, destination_row_ptr, source_order, source_row_ptr, spin, fparam, aparam)`;
  2. the inner call forwards `spin=spin` into `forward_common_lower_graph_exportable`'s underlying `forward_common_lower_graph` (thread the parameter through `make_model.py:forward_common_lower_graph_exportable`, line 974-1098, adding `spin: torch.Tensor | None = None` to its closure the same way `charge_spin` is threaded);
  3. key translation uses `_translate_spin_energy_keys` (force_mag mandatory);
  4. no `charge_spin` slot (rejected at build for native spin in this plan);
  5. no with-comm variant (single-rank only).

- [ ] **Step 4: Run tests**

Run: `python -m pytest source/tests/pt_expt/model/test_dpa4_native_spin.py -q`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add deepmd/pt_expt/model/dpa4_native_spin_model.py deepmd/pt_expt/model/make_model.py source/tests/pt_expt/model/test_dpa4_native_spin.py
git commit --no-verify -m "feat(pt_expt): graph-spin .pt2 exportable ABI (spin at index 10)"
```

---

### Task 6: serialization — graph-kind spin freeze + metadata + no with-comm

**Files:**
- Modify: `deepmd/pt_expt/utils/serialization.py`:
  - spin detection ~line 1331 (`is_spin = type == "spin_ener"`) → also `"dpa4_native_spin"`/`"sezm_native_spin"` (native flavor flag `is_native_spin`);
  - model rebuild ~line 1338 → route native types to `DPA4NativeSpinModel.deserialize`;
  - graph rejection ~line 1374-1377 → allow native spin on `lower_kind == "graph"`; KEEP rejecting virtual `"spin_ener"` graph and native-spin `"dpa1_canonical"`;
  - `build_synthetic_graph_inputs` (line 409, tuple at 535-548) → `want_spin` flag inserting a `(N, 3)` fp sample at index 10;
  - `_build_graph_dynamic_shapes` (line 685) → spec entry for spin at index 10 (same node-axis spec pattern as flat `aparam`); `_with_comm` variant NOT extended (spin never with-comm);
  - `_needs_with_comm_artifact` → native spin ⇒ `False` (before the graph⇒True rule);
  - `_collect_metadata` (line 937) → for native spin emit `is_spin=True`, `ntypes_spin`, `use_spin`, and `output_keys` from the spin translate (`atom_energy, energy, force, force_mag, virial[, atom_virial]`);
  - trace call site: graph-kind native spin traces `model.forward_lower_graph_exportable` (Task 5).
- Test: extend `source/tests/pt_expt/model/test_dpa4_export.py`

**Interfaces:**
- Consumes: Task 5 exportable; existing dense-spin metadata conventions (`is_spin`, `ntypes_spin`, `use_spin` at serialization.py:1009-1013).
- Produces: `deeppot`-style graph `.pt2` whose metadata has `lower_input_kind: "graph"`, `is_spin: true`, `has_comm_artifact: false`, `has_message_passing: true`, `output_keys: [atom_energy, energy, force, force_mag, virial]`; NO embedded `forward_lower_with_comm.pt2` sidecar.

- [ ] **Step 1: Write the failing test** (extend `test_dpa4_export.py` with a new parametrize row or a dedicated class):

```python
def test_native_spin_graph_freeze(tmp_path):
    """Native-spin DPA4 freezes to a graph-kind .pt2: metadata + no sidecar + eval parity."""
    model_file = tmp_path / "dpa4_spin_graph.pt2"
    _freeze_native_spin(model_file)  # helper mirroring the file's existing graph freeze helper, config = NATIVE_SPIN_CONFIG
    import json, zipfile
    with zipfile.ZipFile(model_file) as z:
        names = z.namelist()
        md = json.loads(z.read(next(n for n in names if n.endswith("metadata.json"))))
    assert md["lower_input_kind"] == "graph"
    assert md["is_spin"] is True
    assert md["has_comm_artifact"] is False
    assert "force_mag" in md["output_keys"]
    assert not any(n.endswith("forward_lower_with_comm.pt2") for n in names)

def test_virtual_spin_graph_freeze_still_rejected(...):
    """spin_ener (virtual) graph freeze keeps raising NotImplementedError."""
```

- [ ] **Step 2: Run, verify failure** — `python -m pytest source/tests/pt_expt/model/test_dpa4_export.py -k spin -x -q` → FAIL at the graph rejection (`NotImplementedError: graph-form .pt2 export is not supported for spin models`).

- [ ] **Step 3: Implement** the serialization changes listed under Files. The rejection becomes:

```python
        if lower_kind in ("graph", "dpa1_canonical") and is_spin:
            if not is_native_spin or lower_kind == "dpa1_canonical":
                raise NotImplementedError(
                    "graph-form .pt2 export supports only the native spin "
                    "scheme (dpa4_native_spin); virtual-atom spin models "
                    "export with the dense lower"
                )
```

`build_synthetic_graph_inputs`: sample spin `torch.zeros((n_node_total, 3), dtype=prec)` is NOT acceptable (zero spin may hit degenerate branches — norm(0)); use a small deterministic non-zero sample, e.g. `0.1 + 0.05 * torch.arange(n_node_total * 3).reshape(N, 3)` in the sample dtype. `_needs_with_comm_artifact`: add as the FIRST rule `if is_native_spin(model): return False` (implement the predicate on the model class: `DPA4NativeSpinModel.dense_lower_supports_comm`-style — reuse `has_message_passing_across_ranks() -> False` on the wrapper by delegating to a wrapper-level override returning False, and document why: ghost spin exchange unimplemented).

- [ ] **Step 4: Run** — `python -m pytest source/tests/pt_expt/model/test_dpa4_export.py source/tests/pt_expt/utils/test_graph_pt2_metadata.py -q` → ALL PASS (existing kind-aware metadata rows unaffected).

- [ ] **Step 5: Commit**

```bash
git add deepmd/pt_expt/utils/serialization.py deepmd/pt_expt/model/dpa4_native_spin_model.py source/tests/pt_expt/model/test_dpa4_export.py
git commit --no-verify -m "feat(pt_expt): graph-kind .pt2 freeze for native-spin DPA4 (no with-comm)"
```

---

### Task 7: DeepEval — graph-spin fast path

**Files:**
- Modify: `deepmd/pt_expt/infer/deep_eval.py`:
  - `_is_spin` detection (~lines 300-306, 341, 552) → include native types;
  - nlist-backend guard (~lines 260-274): allow the graph backend for native-spin graph artifacts;
  - `_eval_model_spin` (line 1605): add a `lower_input_kind == "graph"` branch building the 12-tensor input per the ABI (copy the CSR/graph input construction of `_eval_model_graph` lines 1920-1934, insert owned-atom `spin_t (N,3)` at index 10, no charge_spin slot) and unpacking `force_mag` from the outputs by name.
- Test: extend `source/tests/pt_expt/model/test_dpa4_native_spin.py` (or the export test file) with a DeepEval-vs-eager parity test.

**Interfaces:**
- Consumes: Task 6's frozen `.pt2`; `metadata["output_keys"]` naming (`force_mag`).
- Produces: `DeepEval.eval(..., spin=...)` works on a graph-kind native-spin `.pt2`, returning energy/force/force_mag/virial; parity vs the eager pt_expt model at rtol/atol 1e-10.

- [ ] **Step 1: Failing test** — freeze (reuse Task 6 helper), then:

```python
def test_deep_eval_graph_spin_parity(tmp_path):
    model_file = _freeze_native_spin(tmp_path / "m.pt2")
    from deepmd.infer.deep_pot import DeepPot  # or the DeepSpin evaluator entry the dense spin tests use — mirror test file for deeppot_dpa_spin
    ...
    e, f, fm, v = _eval_spin(model_file, coord, box, atype, spin)
    e_ref, f_ref, fm_ref, v_ref = _eager_reference(...)
    np.testing.assert_allclose(e, e_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fm, fm_ref, rtol=1e-10, atol=1e-10)
```

(Copy the evaluator-entry pattern from the existing dense-spin inference tests — grep `deeppot_dpa_spin` under `source/tests/` for the Python-side evaluator usage.)

- [ ] **Step 2: Run, verify failure** — the graph-kind spin artifact either raises in `_eval_model_spin` (unknown kind) or takes a wrong branch.

- [ ] **Step 3: Implement** the deep_eval changes listed under Files.

- [ ] **Step 4: Run** — spin tests + `python -m pytest source/tests/pt_expt/model/test_dpa4_export.py -q` (no regression on energy graph eval).

- [ ] **Step 5: Commit**

```bash
git add deepmd/pt_expt/infer/deep_eval.py source/tests/pt_expt/model/test_dpa4_native_spin.py
git commit --no-verify -m "feat(pt_expt): DeepEval graph fast path for native-spin .pt2"
```

---

### Task 8: fixtures — `gen_dpa4_spin.py` + expected refs

**Files:**
- Create: `source/tests/infer/gen_dpa4_spin.py`
- Create (generated, committed): `source/tests/infer/deeppot_dpa4_spin_graph.yaml`, `.expected` (the `.pt2` itself is platform-specific and generated per-machine, matching gen_dpa4.py conventions — commit exactly what gen_dpa4.py commits, no more)

**Interfaces:**
- Consumes: `gen_dpa4.py`'s structure (yaml → dpmodel → jitter → freeze graph `.pt2` → sanity checks → `write_expected_ref`); `gen_spin.py`'s spin expected-ref fields (`expected_e/f/fm/tot_v`).
- Produces: fixture named `deeppot_dpa4_spin_graph.pt2` + `.expected` consumed by Task 9 (C++) and Task 10 (LAMMPS).

- [ ] **Step 1:** Write `gen_dpa4_spin.py`: import the jitter helper from `gen_dpa4` (`from gen_dpa4 import _jitter_zero_arrays` — same directory, avoids a third copy), define the native-spin yaml (NATIVE_SPIN_CONFIG shape: dpa4 descriptor + `use_spin` + `spin: {use_spin: [...], scheme: native}`, water/NiO-like 2-type system), build via dpmodel `get_model`, jitter, serialize to yaml, convert to pt_expt, freeze graph-kind `.pt2` (the Task 6 route), run a sanity eval (energy finite, force_mag non-zero on spin types, zero on non-spin types), and write the `.expected` file with `e/f/fm/v` reference values evaluated through DeepEval on the fixed coordinate set (mirror `gen_spin.py`'s `write_expected_ref` call and field names so the C++ harness parses it).
- [ ] **Step 2:** Run: `cd source/tests/infer && python gen_dpa4_spin.py` — expect "Done!" and the three artifacts.
- [ ] **Step 3:** Spot-check metadata: `unzip -p deeppot_dpa4_spin_graph.pt2 '*/metadata.json' | python -m json.tool | grep -E 'is_spin|lower_input_kind|has_comm|force_mag'`.
- [ ] **Step 4: Commit** (yaml + expected + gen script; NOT the .pt2 if gen_dpa4.py doesn't commit its .pt2 — match its `.gitignore` state):

```bash
git add source/tests/infer/gen_dpa4_spin.py source/tests/infer/deeppot_dpa4_spin_graph.yaml source/tests/infer/deeppot_dpa4_spin_graph.expected
git commit --no-verify -m "test(infer): native-spin DPA4 graph .pt2 fixture generator"
```

---

### Task 9: C++ — `DeepSpinPTExpt` graph route + gtest

**Files:**
- Modify: `source/api_cc/src/DeepSpinPTExpt.cc` (init ~line 168; dispatch ~lines 619, 754-807; standalone compute ~1179; output unpack ~844)
- Modify: `source/api_cc/include/commonPT.h` (add `remap_graph_spin_outputs_to_dense_keys` next to `remap_graph_outputs_to_dense_keys`, line 722)
- Create: `source/api_cc/tests/test_deepspin_dpa4_graph_ptexpt.cc`

**Interfaces:**
- Consumes: shared graph builders (`createEdgeTensors`/`compactEdgeTensors`/`GraphTensorPack`/`canonicalizeGraphPayload`/`buildGraphTensors`/`applyPairExclusion` in commonPT.h) — reuse as-is; the ABI (spin index 10); metadata keys from Task 6.
- Produces:
  - `DeepSpinPTExpt::init` reads `lower_input_is_graph_` from `lower_input_kind == "graph"`;
  - `run_model_graph(...)`: 10 base tensors + `spin (N,3)` + conditional `fparam`/`aparam` (mirror `DeepPotPTExpt.cc:507-537` exactly, inserting the spin push UNCONDITIONALLY between the base block and the dim-gated tail);
  - graph branches in the LAMMPS-nlist and standalone `compute` paths, single-rank ONLY: `fold_to_local=true`, spin tensor from the owned-atom spin input `(nloc, 3)`; if multi-rank (`comm active` or `nghost` participation implies with-comm) → `throw deepmd::deepmd_exception("multi-rank inference is not supported for graph-kind spin .pt2 models (has_comm_artifact=false); ...")` — this must trigger on the SAME condition the energy path uses to select with-comm;
  - `remap_graph_spin_outputs_to_dense_keys`: extends the energy remap with `force_mag (N,3)` → zero-padded `(nf, nall, 1, 3)` `energy_derv_r_mag` (copy the force zero-pad/scatter pattern at commonPT.h:742-744);
  - gtest `TestInferDeepSpinDpa4GraphPtExpt` modeled on `test_deeppot_dpa_ptexpt_spin.cc` (same `deepmd::DeepSpin` API, `ValueTypes` typed suite), model `../../tests/infer/deeppot_dpa4_spin_graph.pt2`, refs from `.expected`, with `skip_if_artifact_missing`-style guard matching the dpa4 graph row's convention.

- [ ] **Step 1:** Write the gtest (energy/force/force_mag/virial vs `.expected`, plus `test_get_use_spin`).
- [ ] **Step 2:** Build: `cmake --build source/build_tests -j2 && cmake --install source/build_tests --prefix dp_test` — expect the new test compiled; run `ctest --test-dir source/build_tests -R DeepSpinDpa4Graph --output-on-failure` → FAIL (unknown lower kind / graph artifact rejected).
- [ ] **Step 3:** Implement the C++ changes listed under Files.
- [ ] **Step 4:** Re-build + re-run the filtered ctest → PASS. Also run the existing spin rows: `ctest --test-dir source/build_tests -R DeepSpin --output-on-failure` → no regression.
- [ ] **Step 5: Commit**

```bash
git add source/api_cc/src/DeepSpinPTExpt.cc source/api_cc/include/commonPT.h source/api_cc/tests/test_deepspin_dpa4_graph_ptexpt.cc
git commit --no-verify -m "feat(cc): DeepSpinPTExpt NeighborGraph route for native-spin graph .pt2"
```

---

### Task 10: LAMMPS — single-rank `pair_style deepspin` graph test

**Files:**
- Create: `source/lmp/tests/test_lammps_dpa4_spin_graph_pt2.py`

**Interfaces:**
- Consumes: fixture `deeppot_dpa4_spin_graph.pt2` (Task 8); `test_lammps_spin_pt2.py`'s spin-LAMMPS scaffolding (`write_lmp_data_spin`, `pair_style deepspin`, sp-atom style); `test_lammps_dpa4_graph_pt2.py`'s reference-comparison pattern (LAMMPS vs Python DeepEval on identical coordinates).
- Produces: 3 tests — `test_pair_deepspin` (energy+force+force_mag vs Python eval, atol 1e-8), `test_pair_deepspin_virial`, and `test_pair_deepspin_mpi_fails_fast` (2-rank run must abort with the Task 9 multi-rank message, NOT silently succeed — copy the `_run_mpi_subprocess` + returncode/timeout assertions from `test_lammps_dpa4_graph_pt2.py`'s empty-rank test).

- [ ] **Step 1:** Write the test file (copy scaffolding; keep the cell/atom count small; forces `atol=1e-8, rtol=0` per the dpa2 precedent).
- [ ] **Step 2:** Run: `pytest source/lmp/tests/test_lammps_dpa4_spin_graph_pt2.py -v --tb=short` (env: `LAMMPS_PLUGIN_PATH`/`LD_LIBRARY_PATH` per `doc/development/testing.md`) → all PASS.
- [ ] **Step 3: Commit**

```bash
git add source/lmp/tests/test_lammps_dpa4_spin_graph_pt2.py
git commit --no-verify -m "test(lmp): single-rank LAMMPS deepspin on the DPA4 graph .pt2"
```

---

### Task 11: training smoke, docs, battery, wrap-up

**Files:**
- Test: extend `source/tests/pt_expt/model/test_dpa4_native_spin.py` (training smoke)
- Modify: `doc/model/dpa4.md` (native-spin section)
- Modify: `.superpowers/sdd/progress.md` (ledger)

**Interfaces:**
- Consumes: pt_expt training stack (`training.py` `ener_spin` loss branch, `wrapper.py` spin threading — both exist); a spin-labeled dataset (grep the dataset used by existing pt/pt_expt spin training tests, e.g. the NiO spin data under `source/tests/`).
- Produces: 2-step training smoke (loss finite, force_mag gradient flows — assert a spin-embedding parameter's grad is non-zero after one step); docs section stating: native scheme only, graph route only, single-rank only, per-local-atom spin, `force_mag = -dE/dspin`, virtual/deepspin scheme rejected, multi-rank + charge-spin FiLM + bridging = follow-ups.

- [ ] **Step 1:** Training smoke test: build a trainer from a minimal input json (native spin config + `loss: {type: ener_spin}`) on the spin dataset, run 2 steps, assert finite loss and non-zero grad on a jitter-free spin-embedding weight. If the pt_expt trainer needs no change, this is test-only; if it needs a dispatch fix (e.g. model-type routing for `dpa4_native_spin` in `training.py`/`wrapper.py`), make the minimal fix in the same commit.
- [ ] **Step 2:** Docs: add "Native spin (magnetic) DPA4" subsection to `doc/model/dpa4.md`.
- [ ] **Step 3:** Full local battery:

```bash
python -m pytest source/tests/common/dpmodel/test_dpa4_call_graph.py source/tests/common/dpmodel/test_descrpt_dpa4.py source/tests/common/dpmodel/test_dpa4_native_spin_model.py -q
python -m pytest source/tests/pt_expt/model/test_dpa4_native_spin.py source/tests/pt_expt/model/test_dpa4_graph_lower.py source/tests/pt_expt/model/test_dpa4_export.py source/tests/pt_expt/utils/test_graph_pt2_metadata.py -q
python -m pytest source/tests/pt/model/test_dpa4_dpmodel_parity.py -q   # pt untouched — must stay green
ruff check deepmd source/tests && ruff format --check deepmd source/tests
```

- [ ] **Step 4:** Ledger + commit docs; report known limitations (PR-body material): (1) single-rank only — no ghost-spin exchange, graph-kind spin `.pt2` carries `has_comm_artifact=false`, C++ fails fast multi-rank; (2) charge-spin FiLM + native spin combination rejected (Plan A follow-up); (3) whole-model conversion of pt-serialized `sezm_native_spin` checkpoints not claimed (alias covers type string with dpmodel structure only); (4) dpmodel backend energy-only (force/force_mag from pt_expt autograd); (5) CUDA validation pending a remote GPU session (fixtures + LAMMPS + C++ rows).

```bash
git add doc/model/dpa4.md .superpowers/sdd/progress.md
git commit --no-verify -m "docs(dpa4): native-spin graph route documentation + ledger"
```

---

## Self-Review Notes

- **Spec coverage:** spin threading (T1/T3), model class both backends (T2/T4), export ABI (T5), freeze (T6), Python inference (T7), fixtures (T8), C++ (T9), LAMMPS (T10), training/docs (T11). Multi-rank explicitly out of scope with fail-fast pins (T6 metadata test + T9 throw + T10 MPI test).
- **Type consistency:** `spin` is `(nf, nloc, 3)` above the wrapper, `(N, 3)` flat at/below `_call_common_graph`; ABI index 10 used consistently in T5/T6/T7/T9; type string `"dpa4_native_spin"` + alias `"sezm_native_spin"` in T2/T6; output key `force_mag` in T5/T6/T7/T9/T10.
- **Known plan risks (implementers: verify, don't assume):** exact kwarg spelling of the descriptor `use_spin` config key (mirror pt `_get_sezm_native_spin_model`); whether DPA4's spin grads are exactly zero for non-spin types without wrapper masking (mirror pt, no double-masking); the `_input_type_cast` insertion point for spin; `.expected`-file field names must match what the C++ spin harness parses (`expected_fm` etc. — copy from `test_deeppot_dpa_ptexpt_spin.cc`).
