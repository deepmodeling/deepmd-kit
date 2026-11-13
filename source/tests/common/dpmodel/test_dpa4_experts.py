# SPDX-License-Identifier: LGPL-3.0-or-later
"""Mixture of experts (MoE) of the dpmodel DPA4 descriptor and of the fitting networks."""

import copy
import dataclasses
import json

import numpy as np
import pytest
from dargs.dargs import (
    ArgumentKeyError,
    ArgumentTypeError,
)

from deepmd.dpmodel import (
    NativeOP,
)
from deepmd.dpmodel.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.descriptor.dpa4_nn.so3 import (
    ChannelLinear,
    FocusLinear,
    SO3Linear,
)
from deepmd.dpmodel.fitting.ener_fitting import (
    EnergyFittingNet,
)
from deepmd.dpmodel.model.base_model import BaseModel as BaseModelDP
from deepmd.dpmodel.model.expert_merge import (
    check_expert_state,
    check_slot_elements,
    merge_experts,
    slot_elements,
)
from deepmd.dpmodel.model.model import (
    get_model,
)
from deepmd.dpmodel.utils.expert_linear import (
    ExpertWeights,
    is_expert_parameter,
    linear_with_experts,
)
from deepmd.dpmodel.utils.expert_route import (
    ALL_ELEMENTS,
    ExpertRoute,
    ExpertScenario,
    RowLayout,
    frame_states,
    layout_rows,
    normalize_moe,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
)
from deepmd.dpmodel.utils.serialization import (
    array_to_record,
    record_to_array,
    traverse_model_dict,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.model_preset import (
    expand_model_preset,
)

from ...dpa4_fixtures import (
    jitter_zero_arrays,
)

TYPE_MAP = ["H", "C", "O"]
# Expert configurations of the descriptor and of the fitting network.
VARIANTS = {
    "full": ({}, {}),
    "descriptor": ({"n_experts": 3}, None),
    "fitting": (None, {"n_experts": 2}),
    "task": (
        {"n_experts": 0, "element": False},
        {"n_experts": 0, "element": False},
    ),
    "element": ({"n_experts": 0}, {"n_experts": 0}),
    "router": ({"n_experts": 4, "element": False}, None),
    "listed": (
        {"n_experts": 2, "element": ["H", "O"], "router": {"dim": 32, "layers": 3}},
        {"n_experts": 2, "element": ["C"]},
    ),
}
EXPERT_ARRAYS = ("element_weight", "element_bias", "expert_weight", "expert_bias")
# Isolated-atom energies of the types, which make a `vacuum_ref` effective.
PRESET_BIAS = (-13.6, -150.0, -430.1)

COORD = np.random.default_rng(0).uniform(0.0, 7.0, (2, 6, 3))
ATYPE = np.array([[0, 1, 2, 0, 1, 0], [1, 1, 0, 0, 2, 2]])
BOX = np.tile((np.eye(3) * 7.0).reshape(1, 9), (2, 1))


def expert_layers(module, seen=None):
    """Yield every expert map below a module or a network collection."""
    seen = set() if seen is None else seen
    if id(module) in seen:
        return
    seen.add(id(module))
    if isinstance(module, ExpertWeights):
        yield module
    if isinstance(module, list):
        values = module
    else:
        values = list(vars(module).values())
    for value in values:
        if hasattr(value, "_networks"):
            value = [net for net in value._networks if net is not None]
        if isinstance(value, (list, NativeOP)):
            yield from expert_layers(value, seen)


def components(model):
    atomic_model = model.atomic_model
    return atomic_model.descriptor, atomic_model.fitting_net


def set_component_state(component, draw):
    """Replace every expert array and task increment of a component by ``draw(array)``."""
    for layer in expert_layers(component):
        for name in EXPERT_ARRAYS:
            value = getattr(layer, name)
            if value is not None:
                setattr(layer, name, draw(value))
    component.task_weight = [draw(value) for value in component.task_weight]
    component.task_bias = [draw(value) for value in component.task_bias]
    if component.adam_task_embedding is not None:
        component.adam_task_embedding = draw(component.adam_task_embedding)


def set_expert_state(model, draw):
    """Replace every expert array and task increment of the model by ``draw(array)``."""
    for component in components(model):
        set_component_state(component, draw)


def randomize_experts(model, seed):
    """Give every expert array and task increment a small random value."""
    rng = np.random.default_rng(seed)
    set_expert_state(model, lambda value: rng.normal(0.0, 0.05, value.shape))


def build_model(variant=None, *, jitter=False, descriptor=None, **fitting):
    descriptor_moe, fitting_moe = VARIANTS[variant] if variant else (None, None)
    model = expand_model_preset(
        {"preset": "dpa4.5-nano-v20261111", "type_map": TYPE_MAP}
    )
    model["descriptor"].update(
        {
            "sel": 40,
            "seed": 1,
            "precision": "float64",
            "add_chg_spin_ebd": True,
            "default_chg_spin": [0.0, 1.0],
            **copy.deepcopy(descriptor or {}),
        }
    )
    model["fitting_net"].update(seed=1, precision="float64", **fitting)
    if fitting.get("vacuum_ref"):
        # The reference acts on an output whose bias the preset fixes.
        model["preset_out_bias"] = {
            "energy": dict(zip(TYPE_MAP, PRESET_BIAS, strict=True))
        }
    if descriptor_moe is not None:
        model["descriptor"]["moe"] = copy.deepcopy(descriptor_moe)
    if fitting_moe is not None:
        model["fitting_net"]["moe"] = copy.deepcopy(fitting_moe)
    config = normalize(
        {
            "model": model,
            "learning_rate": {"start_lr": 1e-3},
            "loss": {},
            "training": {"training_data": {"systems": ["x"]}, "numb_steps": 1},
        }
    )
    data = copy.deepcopy(config["model"])
    data["type"] = "standard"
    model = get_model(data)
    assert model.atomic_model.fitting_net.needs_vacuum_descriptor() == bool(
        fitting.get("vacuum_ref")
    )
    if not jitter:
        return model
    # A fresh DPA4 zero-initializes its residual output projections, so its
    # energy would not depend on the geometry at all. Jittering them makes the
    # messages, and with them the expert maps, reach the energy; the expert
    # weights keep the zero start of a fresh model.
    model = BaseModelDP.deserialize(
        jitter_zero_arrays(model.serialize(), np.random.default_rng(7))
    )
    set_expert_state(model, np.zeros_like)
    return model


def energy(model, coord=COORD, atype=ATYPE, box=BOX, key="energy_redu", **kwargs):
    nf = coord.shape[0]
    out = model.call_common(
        coord.reshape(nf, -1), atype, box, neighbor_graph_method="dense", **kwargs
    )
    return out[key]


def merged_copy(model, **scenario):
    merged = copy.deepcopy(model)
    merge_experts(merged, ExpertScenario(**scenario))
    return merged


@pytest.mark.parametrize(
    ("moe", "charge_spin"),
    [
        ({"rank": 4}, True),
        ({"n_experts": -1}, True),
        ({"mat_prob": 1.0}, True),
        ({"mat_prob": -0.1}, True),
        ({"element": ["H", "H"]}, True),
        ({"router": {"dim": 0}}, True),
        ({"router": {"layers": 0}}, True),
        ({"router": {"dropout": 0.25}}, True),
        ({"element": "false"}, True),
        # A router without the charge/spin of the frames reads the element.
        ({"n_experts": 2, "element": False}, False),
    ],
)
def test_invalid_moe_configuration_raises(moe, charge_spin) -> None:
    with pytest.raises(ValueError):
        normalize_moe(moe, charge_spin=charge_spin)


@pytest.mark.parametrize(
    ("section", "options", "error", "message"),
    [
        (
            "descriptor",
            {"moe": {"element": ["H", "Xx"]}},
            ValueError,
            "type map does not hold",
        ),
        (
            "descriptor",
            {"moe": {"n_experts": {"ffn": 8}}},
            ArgumentTypeError,
            "n_experts",
        ),
        (
            "descriptor",
            {"moe": {"router": {"dropout": 0.25}}},
            ArgumentKeyError,
            "dropout",
        ),
        ("fitting_net", {"moe": {"element": False}}, ValueError, "needs `element`"),
        (
            "fitting_net",
            {"moe": {}, "dim_case_embd": 2},
            ValueError,
            "dim_case_embd",
        ),
        # Without the charge/spin of the frames the router reads the element.
        (
            "descriptor",
            {"moe": {"element": False}, "add_chg_spin_ebd": False},
            ValueError,
            "needs `element`",
        ),
    ],
)
def test_input_with_unusable_moe_is_refused(section, options, error, message) -> None:
    model = expand_model_preset(
        {"preset": "dpa4.5-nano-v20261111", "type_map": TYPE_MAP}
    )
    model[section].update(copy.deepcopy(options))
    with pytest.raises(error, match=message):
        normalize(
            {
                "model": model,
                "learning_rate": {"start_lr": 1e-3},
                "loss": {},
                "training": {"training_data": {"systems": ["x"]}, "numb_steps": 1},
            }
        )


def test_moe_defaults() -> None:
    assert normalize_moe(None) is None
    assert normalize_moe({}) == {
        "n_experts": 8,
        "element": True,
        "mat_prob": 0.37,
        "router": {"dim": 64, "layers": 1},
    }
    # An empty list of elements gives no element weights of its own.
    assert normalize_moe({"element": []})["element"] is False


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("chunk", [1, 8, 64])
def test_row_layout_groups_rows(static, chunk) -> None:
    rng = np.random.default_rng(3)
    for n_rows, n_groups in ((1, 4), (37, 5), (700, 3), (150, 400)):
        group = rng.integers(0, n_groups, size=n_rows)
        layout = RowLayout.build(group, n_groups, chunk=chunk, static=static)
        assert layout.chunk == chunk
        x = rng.normal(size=(n_rows, 3))
        rows = layout.to_chunks(x)
        n_chunks = layout.chunk_group.shape[0]
        assert rows.shape[0] == n_chunks * chunk
        np.testing.assert_array_equal(layout.from_chunks(rows), x)
        # The free rows of a chunk repeat rows, so every layout row is a row.
        assert set(layout.source.tolist()) <= set(range(n_rows))
        # Every row lies in a chunk of its own group.
        np.testing.assert_array_equal(
            layout.chunk_group[layout.position // chunk], group
        )
        assert len(set(layout.position.tolist())) == n_rows
        in_use = sum(
            -(-count // chunk) for count in np.bincount(group, minlength=n_groups)
        )
        if static:
            assert n_chunks == (n_rows + (chunk - 1) * n_groups) // chunk >= in_use
            assert layout.chunk_group.max() < n_groups
        else:
            assert n_chunks == in_use


@pytest.mark.parametrize("n_frames", [1, 2, 6])
def test_frame_states_index_the_distinct_states(n_frames: int) -> None:
    charge_spin = np.array(
        [[0.0, 1.0], [1.0, 2.0], [0.0, 1.0], [-1.0, 2.0], [1.0, 2.0], [0.0, 3.0]]
    )[:n_frames]
    state, state_frame = frame_states(charge_spin)
    n_states = len({tuple(frame) for frame in charge_spin.tolist()})
    assert sorted(set(state.tolist())) == list(range(n_states))
    for first in range(len(charge_spin)):
        for second in range(len(charge_spin)):
            same = np.array_equal(charge_spin[first], charge_spin[second])
            assert (state[first] == state[second]) == same
    # The frame of a state has that state.
    np.testing.assert_array_equal(charge_spin[state_frame[state]], charge_spin)


def slot_route(layer, task, slot, n_slots, coeff=None):
    """Route of one map for rows by their element slot, with the coefficients of every slot."""
    layout, chunk_slot, _ = layout_rows(slot, n_slots)
    return ExpertRoute(
        [task],
        layout,
        chunk_slot,
        None if coeff is None else coeff[chunk_slot][:, None],
    ).unit(0, [layer], 0)


def apply(layer, route, x):
    """The map on rows ``x`` with the weights of a route."""
    return route.leave(layer(route.enter(x), route.map(0)))


class TestExpertMaps:
    """An expert map against the plain map of every row."""

    N_EXPERTS = 3
    N_ELEMENTS = 4

    @classmethod
    def maps(cls):
        common = {"precision": "float64", "seed": 3}
        experts = (cls.N_EXPERTS, cls.N_ELEMENTS)
        return [
            (
                linear_with_experts(
                    ChannelLinear, experts, in_channels=8, out_channels=6, **common
                ),
                (4, 8),
            ),
            (
                linear_with_experts(
                    FocusLinear,
                    experts,
                    in_channels=8,
                    out_channels=6,
                    n_focus=2,
                    **common,
                ),
                (2, 8),
            ),
            (
                linear_with_experts(
                    SO3Linear,
                    experts,
                    lmax=2,
                    in_channels=8,
                    out_channels=6,
                    n_focus=2,
                    mlp_bias=True,
                    **common,
                ),
                (9, 2, 8),
            ),
            (
                linear_with_experts(
                    NativeLayer,
                    experts,
                    8,
                    6,
                    activation_function="tanh",
                    resnet=True,
                    **common,
                ),
                (8,),
            ),
        ]

    @classmethod
    def randomized(cls, rng):
        """Every map with random MoE weights and a random task increment."""
        for layer, shape in cls.maps():
            for name in EXPERT_ARRAYS:
                value = getattr(layer, name)
                if value is not None:
                    setattr(layer, name, rng.normal(0.0, 0.3, value.shape))
            weight_shape, bias_shape = layer.task_shapes()
            task = (
                rng.normal(0.0, 0.3, weight_shape),
                None if bias_shape is None else rng.normal(0.0, 0.3, bias_shape),
            )
            yield layer, shape, task

    @staticmethod
    def zero_task(layer):
        weight_shape, bias_shape = layer.task_shapes()
        return (
            np.zeros(weight_shape),
            None if bias_shape is None else np.zeros(bias_shape),
        )

    def test_fresh_map_is_the_plain_map(self) -> None:
        rng = np.random.default_rng(0)
        n_slots = 1 + self.N_ELEMENTS
        for layer, shape in self.maps():
            x = rng.normal(size=(11, *shape))
            route = slot_route(
                layer,
                self.zero_task(layer),
                rng.integers(0, n_slots, size=11),
                n_slots,
                rng.dirichlet(np.ones(self.N_EXPERTS), size=n_slots),
            )
            plain = layer.PLAIN.deserialize(layer.PLAIN.serialize(layer))
            np.testing.assert_allclose(
                apply(layer, route, x), plain(x), rtol=1e-12, atol=1e-12
            )
            # Rows that share one set of weights need no layout.
            single = ExpertRoute(
                [self.zero_task(layer)], None, None, np.ones((1, 1, self.N_EXPERTS))
            ).unit(0, [layer], 0)
            np.testing.assert_allclose(
                apply(layer, single, x), plain(x), rtol=1e-12, atol=1e-12
            )

    def test_rows_take_the_weights_of_their_conditions(self) -> None:
        rng = np.random.default_rng(1)
        n_rows = 150
        n_slots = 1 + self.N_ELEMENTS
        for layer, shape, task in self.randomized(rng):
            x = rng.normal(size=(n_rows, *shape))
            coeff = rng.dirichlet(np.ones(self.N_EXPERTS), size=n_slots)
            slot = rng.integers(0, n_slots, size=n_rows)
            out = apply(layer, slot_route(layer, task, slot, n_slots, coeff), x)
            # The merge keeps the slot of a row, and is the plain map on
            # the rows of slot 0.
            for row in (0, 7, n_rows - 1):
                keep = [slot[row]] if slot[row] else []
                merged = layer.merge(task, coeff, keep)
                one = x[row : row + 1]
                if keep:
                    single = slot_route(
                        merged, self.zero_task(layer), np.ones(1, dtype=np.int64), 2
                    )
                    expected = apply(merged, single, one)
                else:
                    assert type(merged) is layer.PLAIN
                    expected = merged(one)
                np.testing.assert_allclose(
                    out[row : row + 1], expected, rtol=1e-12, atol=1e-12
                )

    def test_serialization_round_trip(self) -> None:
        rng = np.random.default_rng(2)
        for layer, _, _ in self.randomized(rng):
            restored = type(layer).deserialize(layer.serialize())
            for name in EXPERT_ARRAYS:
                value = getattr(layer, name)
                if value is not None:
                    np.testing.assert_array_equal(getattr(restored, name), value)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
class TestExpertModel:
    def test_initial_model_equals_the_plain_model(self, variant) -> None:
        np.testing.assert_array_equal(
            energy(build_model(variant)), energy(build_model())
        )

    def test_symmetries_and_serialization(self, variant) -> None:
        model = build_model(variant, jitter=True)
        randomize_experts(model, 2)
        reference = energy(model)
        assert np.abs(reference - energy(build_model(jitter=True))).max() > 1e-3
        # The symmetry checks below are only meaningful while the energy
        # responds to the geometry.
        assert np.abs(energy(model, 1.3 * COORD) - reference).max() > 1e-6
        rotation, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
        rotated = energy(
            model,
            COORD @ rotation.T,
            ATYPE,
            (BOX.reshape(2, 3, 3) @ rotation.T).reshape(2, 9),
        )
        np.testing.assert_allclose(rotated, reference, atol=1e-10)
        np.testing.assert_allclose(energy(model, -COORD), reference, atol=1e-10)
        order = np.random.default_rng(5).permutation(6)
        np.testing.assert_allclose(
            energy(model, COORD[:, order], ATYPE[:, order]), reference, atol=1e-10
        )
        restored = BaseModelDP.deserialize(model.serialize())
        np.testing.assert_array_equal(energy(restored), reference)
        # The record survives the JSON form of an exported model.
        record = json.loads(
            json.dumps(traverse_model_dict(model.serialize(), array_to_record))
        )
        restored = BaseModelDP.deserialize(traverse_model_dict(record, record_to_array))
        np.testing.assert_array_equal(energy(restored), reference)

    def test_merge_keeping_the_elements_is_exact(self, variant) -> None:
        model = build_model(variant, jitter=True, vacuum_ref=True)
        randomize_experts(model, 6)
        has_elements = any(
            component.moe is not None and component.moe["element"] is not False
            for component in components(model)
        )
        if not has_elements:
            with pytest.raises(ValueError, match="no weights of elements"):
                merged_copy(model, elements=ALL_ELEMENTS)
            return
        merged = merged_copy(model, elements=ALL_ELEMENTS)
        for component in components(merged):
            assert not component.has_expert_router()
            assert component.branch_local_parameters() == [] or all(
                not np.any(value) for value in component.branch_local_parameters()
            )
        np.testing.assert_allclose(
            energy(merged, key="energy"), energy(model, key="energy"), atol=1e-12
        )

    def test_plain_merge_is_the_model_without_element_weights(self, variant) -> None:
        model = build_model(variant, jitter=True, vacuum_ref=True)
        randomize_experts(model, 8)
        merged = merged_copy(model)
        # The merged model has the structure of the model without `moe`.
        plain = build_model(vacuum_ref=True)
        for merged_part, plain_part in zip(
            components(merged), components(plain), strict=True
        ):
            assert not merged_part.has_experts()
            record = merged_part.serialize()
            reference = plain_part.serialize()
            assert set(record) == set(reference)
        # It is the model on frames that use no element-specific weight.
        expected = energy(without_element_weights(model), key="energy")
        np.testing.assert_allclose(energy(merged, key="energy"), expected, atol=1e-12)

    def test_reference_removes_the_moe_weights_of_isolated_atoms(self, variant) -> None:
        # One atom per frame, in the ground state its reference is taken in.
        atype = np.arange(len(TYPE_MAP)).reshape(-1, 1)
        coord = np.zeros((len(TYPE_MAP), 1, 3))
        box = np.tile((np.eye(3) * 40.0).reshape(1, 9), (len(TYPE_MAP), 1))

        model = build_model(variant, jitter=True, vacuum_ref=True)
        state = np.asarray(model.atomic_model.vacuum_conditions()["charge_spin"])

        def isolated(model):
            return energy(model, coord, atype, box, charge_spin=state)

        expected = isolated(model)
        randomize_experts(model, 10)
        has_elements = any(
            component.moe is not None and component.moe["element"] is not False
            for component in components(model)
        )
        merged = merged_copy(model, elements=ALL_ELEMENTS if has_elements else ())
        for evaluated in (model, without_element_weights(model), merged):
            np.testing.assert_allclose(isolated(evaluated), expected, atol=1e-10)
        # Without the reference the MoE weights move these energies.
        plain = build_model(variant, jitter=True)
        unreferenced = isolated(plain)
        randomize_experts(plain, 10)
        assert np.abs(isolated(plain) - unreferenced).min() > 1e-6

    def test_merged_charge_spin_is_the_default_input(self, variant) -> None:
        model = build_model(variant, jitter=True)
        randomize_experts(model, 9)
        if not model.atomic_model.descriptor.has_expert_router():
            # Without a router the weights do not depend on the state.
            with pytest.raises(ValueError, match="no router"):
                merged_copy(model, charge_spin=[1.0, 2.0])
            return
        merged = merged_copy(model, charge_spin=[1.0, 2.0])
        assert merged.atomic_model.descriptor.get_default_chg_spin() == [1.0, 2.0]
        state = np.array([[1.0, 2.0]] * 2)
        reference = energy(without_element_weights(model), charge_spin=state)
        for kwargs in ({}, {"charge_spin": state}):
            np.testing.assert_allclose(energy(merged, **kwargs), reference, atol=1e-12)


def with_mat_draws(model, draws=None):
    """
    The model in a training forward whose frames take the given MAT samples.

    Every component trains the frames whose sample is below 0.5 merge-aware,
    without its element-specific weights; without ``draws`` it does so in
    every frame.
    """

    class Drawn(type(model.atomic_model)):
        training = True

        def expert_condition(self, graph, atype, charge_spin, training):
            condition = DPAtomicModel.expert_condition(
                self, graph, atype, charge_spin, training=False
            )
            nf = graph.n_node.shape[0]
            return dataclasses.replace(
                condition,
                mat_draw=np.zeros(nf) if draws is None else np.asarray(draws),
            )

    drawn = copy.deepcopy(model)
    drawn.atomic_model.__class__ = Drawn
    for component in components(drawn):
        if component.moe is not None:
            component.moe = {**component.moe, "mat_prob": 0.5}
    return drawn


def without_element_weights(model):
    """The model on frames that every component evaluates without element weights."""
    return with_mat_draws(model)


class TestConditions:
    def test_charge_spin_and_task_move_the_router(self) -> None:
        model = build_model("full", jitter=True)
        randomize_experts(model, 3)
        descriptor = model.atomic_model.descriptor
        router = descriptor.expert_router
        slots = np.arange(1 + descriptor.n_elements)
        states = descriptor.charge_spin_embedding(np.array([[0.0, 1.0], [1.0, 2.0]]))

        def coefficients(frame):
            return router(
                descriptor.adam_task_embedding,
                np.full_like(slots, frame),
                slots,
                states,
            )

        reference = coefficients(0)
        np.testing.assert_allclose(reference.sum(axis=-1), 1.0, atol=1e-12)
        # The coefficients follow the state of the frame, the element and the task.
        assert np.abs(coefficients(1) - reference).max() > 1e-6
        assert np.abs(reference[1:] - reference[:1]).max() > 1e-6
        descriptor.adam_task_embedding = np.random.default_rng(3).normal(
            0.0, 1.0, descriptor.adam_task_embedding.shape
        )
        assert np.abs(coefficients(0) - reference).max() > 1e-6

    @pytest.mark.parametrize("variant", ["full", "element"])
    def test_frames_of_a_batch_take_their_own_conditions(self, variant) -> None:
        model = build_model(variant, jitter=True, vacuum_ref=True)
        randomize_experts(model, 12)
        state = np.array([[0.0, 1.0], [1.0, 2.0]])
        draws = [0.1, 0.9]
        batch = energy(with_mat_draws(model, draws), charge_spin=state)
        for frame in range(2):
            single = energy(
                with_mat_draws(model, draws[frame : frame + 1]),
                COORD[frame : frame + 1],
                ATYPE[frame : frame + 1],
                BOX[frame : frame + 1],
                charge_spin=state[frame : frame + 1],
            )
            np.testing.assert_allclose(batch[frame], single[0], atol=1e-12)
        # The first frame runs without its element weights, the second with them.
        dropped = energy(without_element_weights(model), charge_spin=state)
        kept = energy(model, charge_spin=state)
        np.testing.assert_allclose(batch[0], dropped[0], atol=1e-12)
        np.testing.assert_allclose(batch[1], kept[1], atol=1e-12)
        assert np.abs(dropped - kept).min() > 1e-6

    def test_kept_elements_compute_the_model_on_their_atoms(self) -> None:
        model = build_model("full", jitter=True)
        randomize_experts(model, 5)
        merged = merged_copy(model, elements=("H", "O"))
        for component in components(merged):
            assert component.moe["element"] == ["H", "O"]
            assert component.moe["n_experts"] == 0
        atype = np.array([[0, 2, 2, 0, 0, 0], [2, 2, 0, 0, 2, 2]])
        np.testing.assert_allclose(
            energy(merged, atype=atype), energy(model, atype=atype), atol=1e-12
        )
        # Carbon uses the weights shared by the elements.
        assert np.abs(energy(merged) - energy(model)).max() > 1e-6
        np.testing.assert_allclose(
            energy(merged_copy(merged)), energy(merged_copy(model)), atol=1e-12
        )

    def test_merged_model_folds_its_vacuum_reference(self) -> None:
        model = build_model("full", jitter=True, vacuum_ref=True)
        randomize_experts(model, 4)
        merged = merged_copy(model, elements=("H", "O"))
        atype = np.array([[0, 2, 2, 0, 0, 0], [2, 2, 0, 0, 2, 2]])
        reference = energy(merged, atype=atype)
        np.testing.assert_allclose(reference, energy(model, atype=atype), atol=1e-12)
        # Without a router the reference of an atom depends on its type alone.
        merged.atomic_model.fold_vacuum_reference()
        assert not merged.atomic_model.fitting_net.needs_vacuum_descriptor()
        np.testing.assert_allclose(energy(merged, atype=atype), reference, atol=1e-12)

    def test_merge_refuses_what_it_cannot_keep_or_drop(self, caplog) -> None:
        # The descriptor of this model has weights for H and O, the fitting for C.
        model = build_model("listed", jitter=True)
        randomize_experts(model, 2)
        reference = energy(model)
        for elements in (("C",), ("H",)):
            with pytest.raises(ValueError, match="no weights of their own"):
                merge_experts(model, ExpertScenario(elements=elements))
        task_only = build_model("task")
        with pytest.raises(ValueError, match="no weights of elements"):
            merge_experts(task_only, ExpertScenario(elements=ALL_ELEMENTS))
        # A refused merge leaves the model as it was.
        for refused in (model, task_only):
            assert all(component.has_experts() for component in components(refused))
        assert model.atomic_model.has_expert_router()
        np.testing.assert_array_equal(energy(model), reference)

        untrained = build_model(descriptor={"moe": {"n_experts": 0, "mat_prob": 0.0}})
        with pytest.raises(ValueError, match="mat_prob"):
            merged_copy(untrained)
        # Keeping a part of the elements serves systems of these elements.
        with caplog.at_level("WARNING"):
            merged_copy(untrained, elements=("H",))
        assert "serves systems of ['H'] only" in caplog.text
        caplog.clear()
        with caplog.at_level("WARNING"):
            merged_copy(untrained, elements=ALL_ELEMENTS)
        assert not caplog.text
        with pytest.raises(ValueError, match="no MoE"):
            merged_copy(build_model())

    def test_element_slots_are_matched_by_name(self) -> None:
        model = build_model("element")
        assert slot_elements(model) == [TYPE_MAP, TYPE_MAP]
        assert slot_elements(build_model("task")) == [None, None]
        listed = build_model("listed")
        assert slot_elements(listed) == [["H", "O"], ["C"]]
        # A model takes the element weights of another one slot by slot.
        check_slot_elements(model, build_model("full"))
        check_slot_elements(model, build_model("task"))
        check_slot_elements(listed, merged_copy(listed, elements=ALL_ELEMENTS))
        with pytest.raises(ValueError, match=r"belong to \['H', 'O'\]"):
            check_slot_elements(model, listed)
        reordered = build_model(
            descriptor={"moe": {"n_experts": 0, "element": ["O", "H"]}}
        )
        with pytest.raises(ValueError, match="in this order"):
            check_slot_elements(reordered, listed)

    def test_router_without_conditions_is_refused_off_the_graph(self) -> None:
        descriptor = build_model("full").atomic_model.descriptor
        with pytest.raises(ValueError, match="neighbor-graph"):
            descriptor._expert_route(np.zeros(4, dtype=np.int64), None)

    def test_fitting_requirements(self) -> None:
        with pytest.raises(ValueError, match="dim_case_embd"):
            build_model("fitting", dim_case_embd=2)

    def test_type_map_change_keeps_the_element_weights(self) -> None:
        model = build_model("fitting", jitter=True)
        randomize_experts(model, 4)
        fitting = model.atomic_model.fitting_net
        changed = copy.deepcopy(fitting)
        changed.change_type_map(["O", "N", "H", "C"])
        assert changed.moe["element"] == TYPE_MAP
        assert changed.element_slot == [3, 0, 1, 2]
        rng = np.random.default_rng(1)
        descriptor = rng.normal(size=(1, 3, fitting.dim_descrpt))
        np.testing.assert_allclose(
            changed.call(descriptor, np.array([[2, 3, 0]]))["energy"],
            fitting.call(descriptor, np.array([[0, 1, 2]]))["energy"],
            atol=1e-12,
        )


@pytest.mark.parametrize("resnet_dt", [False, True])
def test_energy_fitting_network_has_moe(resnet_dt) -> None:
    options = {
        "ntypes": len(TYPE_MAP),
        "dim_descrpt": 8,
        "neuron": [16, 16],
        "resnet_dt": resnet_dt,
        "mixed_types": True,
        "precision": "float64",
        "type_map": TYPE_MAP,
        "seed": 1,
    }
    rng = np.random.default_rng(2)
    descriptor = rng.normal(size=(2, 7, 8))
    atype = rng.integers(0, len(TYPE_MAP), size=(2, 7))

    def output(fitting, types=atype):
        return fitting.call(descriptor, types)["energy"]

    plain = EnergyFittingNet(**options)
    fitting = EnergyFittingNet(**options, moe={"n_experts": 2})
    # The batched products of the chunks round as the plain product does not.
    np.testing.assert_allclose(output(fitting), output(plain), atol=1e-13)
    set_component_state(fitting, lambda value: rng.normal(0.0, 0.05, value.shape))
    reference = output(fitting)
    assert np.abs(reference - output(plain)).max() > 1e-6
    np.testing.assert_array_equal(
        output(EnergyFittingNet.deserialize(fitting.serialize())), reference
    )

    # Every element kept: the same output without router and task increments.
    merged = fitting.merge_experts(ExpertScenario(elements=ALL_ELEMENTS))
    assert merged.has_experts() and not merged.has_expert_router()
    np.testing.assert_allclose(output(merged), reference, atol=1e-12)
    # One element kept: its atoms keep their output and the others change.
    hydrogen = fitting.merge_experts(ExpertScenario(elements=("H",)))
    only_hydrogen = np.zeros_like(atype)
    np.testing.assert_allclose(
        output(hydrogen, only_hydrogen), output(fitting, only_hydrogen), atol=1e-12
    )
    assert np.abs(output(hydrogen) - reference).max() > 1e-6
    # No element kept: the record of the fitting built without MoE,
    # which computes what the kept-hydrogen fitting computes for other atoms.
    dropped = fitting.merge_experts(ExpertScenario())
    assert not dropped.has_experts()
    record, expected = dropped.serialize(), plain.serialize()
    assert record["nets"]["network_type"] == expected["nets"]["network_type"]
    assert set(record["@variables"]) == set(expected["@variables"])
    np.testing.assert_allclose(
        output(dropped, np.full_like(atype, 1)),
        output(hydrogen, np.full_like(atype, 1)),
        atol=1e-12,
    )

    with pytest.raises(ValueError, match="mixed_types"):
        EnergyFittingNet(**{**options, "mixed_types": False}, moe={})
    with pytest.raises(ValueError, match="needs `element`"):
        EnergyFittingNet(**options, moe={"element": False})


def test_plain_records_load_in_pt() -> None:
    """A DPA4 record without MoE loads in pt."""
    from deepmd.pt.model.atomic_model import DPAtomicModel as DPAtomicModelPT

    model = expand_model_preset({"preset": "dpa4-nano-v20260911", "type_map": TYPE_MAP})
    model["descriptor"].update(sel=40, seed=1)
    config = normalize(
        {
            "model": model,
            "learning_rate": {"start_lr": 1e-3},
            "loss": {},
            "training": {"training_data": {"systems": ["x"]}, "numb_steps": 1},
        }
    )
    data = copy.deepcopy(config["model"])
    data["type"] = "standard"
    record = get_model(data).atomic_model.serialize()
    assert "moe" not in record["fitting"]
    DPAtomicModelPT.deserialize(record)


def test_native_spin_merge_keeps_the_moment_input() -> None:
    config = {
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
            "add_chg_spin_ebd": True,
            "default_chg_spin": [0.0, 1.0],
            "moe": {"n_experts": 3},
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [8, 8],
            "precision": "float64",
            "moe": {"n_experts": 2},
        },
        "spin": {"use_spin": [True, False], "scheme": "native"},
    }
    # A fresh model starts spin-free; jittering its zero arrays (spin maps and
    # MoE weights among them) makes the moments and the experts matter.
    model = BaseModelDP.deserialize(
        jitter_zero_arrays(
            get_model(copy.deepcopy(config)).serialize(), np.random.default_rng(9)
        )
    )
    merged = merged_copy(model, charge_spin=[0.0, 3.0], elements=ALL_ELEMENTS)
    assert merged.has_spin() and merged.get_type_map() == ["Ni", "O"]
    rng = np.random.default_rng(5)
    coord = rng.uniform(0.5, 5.5, size=(1, 6, 3))
    atype = np.array([[0, 0, 1, 0, 1, 1]])
    box = 8.0 * np.eye(3)[None]
    spin = rng.normal(size=(1, 6, 3))
    state = np.array([[0.0, 3.0]])
    expected = model.call(coord, atype, spin, box=box, charge_spin=state)
    actual = merged.call(coord, atype, spin, box=box)
    np.testing.assert_allclose(
        actual["atom_energy"], expected["atom_energy"], atol=1e-12
    )
    flipped = merged.call(coord, atype, 2.0 * spin, box=box)
    assert np.abs(flipped["energy"] - actual["energy"]).max() > 1e-8


def test_sharing_keeps_the_task_and_merges_the_envelope_mass() -> None:
    base, replica = (
        build_model("full", descriptor={"density_floor": 0.5}).atomic_model.descriptor
        for _ in range(2)
    )
    base.envelope_mass[...] = 2.0
    base.envelope_count = 100.0
    replica.envelope_mass[...] = 5.0
    replica.envelope_count = 50.0
    replica.share_params(base, 0, model_prob=2.0)
    # (100 * 2 + 2 * 50 * 5) / (100 + 2 * 50)
    np.testing.assert_allclose(replica.envelope_mass, 3.5)
    assert replica.envelope_mass is base.envelope_mass
    null_mass = base.blocks[0].so2_conv.adamw_attn_z_bias_raw
    assert null_mass is not None
    np.testing.assert_allclose(np.log1p(np.exp(null_mass)), 0.5 * 3.5, rtol=1e-6)
    # The networks and the router are shared; the task is the replica's own.
    assert replica.blocks is base.blocks
    assert replica.expert_router is base.expert_router
    assert replica.task_weight[0] is not base.task_weight[0]
    assert replica.adam_task_embedding is not base.adam_task_embedding
    # A resumed replica carries its own statistics, which the share leaves alone.
    resumed = build_model(
        "full", descriptor={"density_floor": 0.5}
    ).atomic_model.descriptor
    resumed.envelope_mass[...] = 9.0
    resumed.envelope_count = 50.0
    resumed.share_params(base, 0, model_prob=2.0, resume=True)
    # The first share already merged the two counts (100 + 2 * 50).
    np.testing.assert_allclose(base.envelope_mass, 3.5)
    assert base.envelope_count == 200.0


def _array_names(model) -> set[str]:
    """Names of the stored arrays of a dpmodel model."""
    names: set[str] = set()

    def collect(node, path):
        if isinstance(node, np.ndarray):
            names.add(path)
        elif isinstance(node, dict):
            for key, value in node.items():
                collect(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                collect(value, f"{path}.{index}")

    collect(model.serialize(), "")
    return names


def _states(source, target):
    """Missing and unexpected keys between the states of two models."""
    ours = _array_names(target)
    theirs = _array_names(source)
    return (
        [name for name in sorted(ours) if name not in theirs],
        [name for name in sorted(theirs) if name not in ours],
    )


@pytest.mark.parametrize("variant", ["task", "element", "descriptor", "full"])
def test_a_plain_model_starts_any_moe_model(variant) -> None:
    """Every term MoE adds starts at zero, so the plain state is the start."""
    plain = build_model()
    moe = build_model(variant)
    missing, unexpected = _states(plain, moe)
    assert not unexpected
    assert any("task_weight" in name for name in missing)
    check_expert_state(missing, unexpected)


@pytest.mark.parametrize(("variant", "grown"), [("task", "full"), ("element", "full")])
def test_a_task_model_starts_a_model_with_more_moe(variant, grown) -> None:
    """MoE grows from the task to the elements and the experts."""
    source = build_model(variant)
    target = build_model(grown)
    missing, unexpected = _states(source, target)
    assert not unexpected
    check_expert_state(missing, unexpected)


def test_a_state_check_refuses_a_model_that_lacks_plain_parameters() -> None:
    """A source missing a shared weight cannot start the target."""
    with pytest.raises(ValueError, match="does not add"):
        check_expert_state(["atomic_model.descriptor.blocks.0.ffns.0.weight"], [])


def test_a_state_check_refuses_a_source_with_extra_parameters() -> None:
    """MoE adds weights to a model; it never drops them."""
    with pytest.raises(ValueError, match="holds parameters that this model"):
        check_expert_state([], ["atomic_model.descriptor.task_weight.0"])


def test_every_moe_parameter_is_recognized() -> None:
    """The state of a model differs from the plain state by MoE terms only."""
    plain = build_model()
    full = build_model("full")
    missing, _ = _states(plain, full)
    assert all(is_expert_parameter(name) for name in missing)
