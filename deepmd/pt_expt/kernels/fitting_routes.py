# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freeze-time measurement of the fitting route of compressed DPA4C.

The fused energy-force operator of compressed DPA4C evaluates its fitting
network over runs of nodes, either through the fused kernel, in one of the
shapes compiled for the hidden width, or through the per-layer kernels. Every
route evaluates the same contractions in the same order and so returns the
same bits; the routes differ only in speed. At a common clock a route takes
the time of the waves that its grids take on the device, which depends on how
full the last of them is, while a board at its power limit clocks each route by
the power that the route itself draws. The operator derives a route from rules
when it is given no table. At freeze time, on the deployment device,
:func:`tune_fitting_routes` measures instead which route serves each run size
and records the result as a fitting-route table, which names a route only
where the measurement departs from the rules and leaves them every other run
size, so that the frozen program runs the code of an untuned one wherever the
two agree. The table is passed to the operator as a literal of the frozen
program (:func:`fitting_routes`), which falls back to the rules on a device of
another fingerprint.

The run sizes are divided into bands between which no compared route changes
its number of waves and no rule changes its route
(``graph_fitting_route_bands``). A route is fastest over a whole band only if
the fill of its last wave does not decide; the measurement therefore times both
ends of every band, and where the two ends disagree it bisects the band for the
run size at which the winner changes. Each sample times the compared routes on
the canonical graph of a periodic bulk system of exactly the run size of
atoms, as replays of a captured CUDA graph of the whole operator, which is how
LAMMPS evaluates it.

A sample times the routes in two ways. Within each capture of the operator the
replays of the routes alternate, so that every route runs at the clocks of a
lightly loaded board, and a route other than that of the rules is a candidate
only where its median gain over the rules exceeds :data:`TIE` and the
half-range of its per-capture gains, half the difference between the largest
and the smallest of these gains; a deployment captures the operator once and
sees the gain of one capture. At a common clock the fitting network takes the
same time whatever the node types of the system, so the alternating replays run
the node types of the sample alone. A train is a run of consecutive replays of
one route, timed after a lead-in of further replays of that route; trains time
the routes at the clocks of a board held at its power limit, where a route that
draws more power can lose what it wins at a common clock. The loss depends on
the node types of the system, its composition: the descriptor may run other
scans for one node type than for several, and so sets the power and the length
of the step around the fitting network. On an L20 at its 350 W limit, the
per-layer kernels of a trained Air model at 12,320 nodes gain 1.1% on trains
with every node carbon and lose 1.4% with three node types. A candidate
therefore takes a run size only where it loses by no more than :data:`TIE` on
the trains of two compositions: every node of one type, and the node types of
the sample. Where a deployment leaves the device idle part of the time, the
gain of a route lies between its gains in the alternating replays and on the
trains, as on a V100 at its 300 W limit, where the gain on trains rises towards
that of the alternating replays as the device idles more. On these compositions
a route that the measurement selects is therefore not slower than the rules at
a load in between, beyond the tie and the noise of the measurements.
"""

from __future__ import (
    annotations,
)

import dataclasses
import logging
import math
import statistics
import time
from typing import (
    Any,
)

import torch

from deepmd.pt_expt.kernels.autotune import (
    register_autotuner,
)

log = logging.getLogger(__name__)

#: First word of a fitting-route table (graph_fitting.cu).
ROUTE_TABLE_MAGIC = 0x52545431
#: Route of the per-layer kernels (graph_ops.h).
PER_LAYER = -1
#: Choice of a table that leaves the run to the rules of the operator
#: (graph_ops.h).
RULES = 0
#: Relative gain within which a route counts as equally fast as that of the
#: rules. A route other than that of the rules is a candidate where its median
#: gain over the rules in the alternating replays of a sample exceeds its bound
#: (:func:`_bound`), the tie or a larger half-range of its per-capture gains. A
#: candidate replaces the route of the rules only if it loses to that route by
#: no more than the tie on the trains of one route at a time.
TIE = 0.01
#: Atoms per Å³ of the synthetic systems: diamond, the density of the benchmark
#: systems.
DENSITY = 0.176
#: Seed of the synthetic systems. The system of a run size is drawn from this
#: seed and the size, so that it does not depend on the order of the samples.
SEED = 20261018
#: Largest run size measured for an operator that evaluates the whole node
#: axis at once (a node tile of zero), beyond which the choice of the last
#: interval extends; an operator of node tiles is measured up to its tile.
MAX_RUN_NODES = 131_072
#: Run sizes over which every route keeps the grids of its kernels.
BLOCK = 32
#: Node types of the synthetic systems, drawn at random up to the type count of
#: the model, as a system of a few elements.
SYNTHETIC_TYPES = 3
#: Captures of the operator per route in the stages of one sample, and the
#: timed replays of every capture. The speed of a captured graph depends on
#: where its memory lies: at large runs the medians of different captures of
#: one route scatter by several percent, more than the replays of one capture
#: do, so a sample decides on the gains of several captures. The first replays of a
#: capture understate the differences between routes, so every capture takes
#: untimed warm-up replays first.
STAGE_CAPTURES = (1, 2, 4)
STAGE_REPLAYS = 3
WARMUP_REPLAYS = 2
#: Margins of the stages but the last. A sample decides after such a stage
#: where the median gain of every route other than that of the rules lies
#: farther than the margin of the stage from its bound, and after the first
#: stage only where no route exceeds its bound, so that a departure from the
#: rules rests on at least three captures, those of the first two stages; the
#: last stage always decides.
STAGE_MARGINS = (0.04, 0.015)
#: Replays per capture of a route whose replay takes less than
#: SHORT_REPLAY_MS, which cost little.
SHORT_STAGE_REPLAYS = 9
SHORT_REPLAY_MS = 2.0
#: Least device time, in ms, of the replays of a route that lead in to its
#: timed train, and the device time of the chunks in which the lead-in is
#: timed. A board at its power limit settles the clock of its multiprocessors
#: on the power draw of a route within about 0.4 s of switching to it (an L20
#: at its 350 W limit), so a train is timed only after half a second of its own
#: replays.
LEAD_IN_MS = 500.0
LEAD_IN_CHUNK_MS = 50.0
#: Least number of timed replays of a train, and the device time, in ms, that
#: they span at about the replay time of the lead-in.
TRAIN_REPLAYS = 9
TRAIN_MS = 100.0
#: Captures of the trains of each composition of a sample
#: (:meth:`_Bench._train_gain`).
TRAIN_CAPTURES = 2

#: Measured tables: (device fingerprint, network key) -> fitting-route table.
_ROUTE_TABLES: dict[tuple, tuple[int, ...]] = {}
#: Fingerprints of the CUDA devices measured in this process, by device index;
#: a table of the backend device is read while tracing, which calls no operator.
_FINGERPRINTS: dict[int, tuple[int, ...]] = {}


def network_key(fitting: Any) -> tuple | None:
    """Return the key of a fitting network in the route tables.

    Parameters
    ----------
    fitting : Any
        The pt_expt energy fitting network.

    Returns
    -------
    tuple or None
        Input width, hidden widths, identity residuals and activation code, or
        ``None`` for a network the fused operator cannot evaluate.
    """
    from deepmd.pt_expt.kernels.graph_fitting import (
        fitting_eligible,
        fitting_operator_arguments,
    )

    if not fitting_eligible(fitting):
        return None
    network = fitting_operator_arguments(fitting)
    widths = tuple(int(w.shape[1]) for w in network.weights)
    return (
        int(network.weights[0].shape[0]),
        widths,
        tuple(int(r) for r in network.residuals),
        int(network.activation),
    )


def _fingerprint(device: torch.device) -> tuple[int, ...]:
    """Return the fingerprint that a table measured on ``device`` carries."""
    index = device.index or 0
    if index not in _FINGERPRINTS:
        _FINGERPRINTS[index] = tuple(
            int(v) for v in torch.ops.deepmd.graph_fitting_route_fingerprint(index)
        )
    return _FINGERPRINTS[index]


def fitting_routes(fitting: Any) -> list[int]:
    """Return the fitting-route table of a network for the backend device.

    Parameters
    ----------
    fitting : Any
        The pt_expt energy fitting network.

    Returns
    -------
    list[int]
        The table that :func:`tune_fitting_routes` measured for the network on
        the backend device in this process, or an empty list, with which the
        operator takes the routes of its rules.
    """
    if not _ROUTE_TABLES:
        return []
    import deepmd.pt_expt.utils.env as env

    fingerprint = (
        _FINGERPRINTS.get(env.DEVICE.index or 0) if env.DEVICE.type == "cuda" else None
    )
    key = network_key(fitting)
    if fingerprint is None or key is None:
        return []
    return list(_ROUTE_TABLES.get((fingerprint, key), ()))


def _bulk_graph(rcut: float, node_count: int, device: torch.device) -> Any:
    """Return the canonical graph of a periodic bulk system of ``node_count`` atoms.

    The atoms occupy ``node_count`` sites, drawn at random, of a simple cubic
    lattice of ``ceil(node_count ** (1/3))`` sites per side in a periodic box
    of the density :data:`DENSITY`, each displaced by up to 0.1 Å. The density
    is uniform, every atom has the full neighbourhood of the bulk, and the
    nearest distances stay near the site spacing. The edges are all pairs
    within ``rcut``, the periodic images of a box smaller than ``2 * rcut``
    included, an atom's own images among them: the ghost-free canonical graph
    that LAMMPS gives ``node_count`` local atoms. They are sorted by bond
    length, as the LAMMPS builder writes the rows.
    """
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_pairs,
    )
    from deepmd.pt_expt.utils.vesin_search import (
        VesinNeighborSearch,
    )

    generator = torch.Generator(device=device).manual_seed(SEED + node_count)
    side = math.ceil(node_count ** (1.0 / 3.0))
    length = (node_count / DENSITY) ** (1.0 / 3.0)
    sites = torch.randperm(
        side**3, dtype=torch.int64, device=device, generator=generator
    )[:node_count]
    cells = torch.stack((sites // side**2, sites // side % side, sites % side), dim=-1)
    positions = (cells.to(torch.float64) + 0.5) * (length / side)
    positions += 0.1 * (
        2.0
        * torch.rand(
            positions.shape, dtype=torch.float64, device=device, generator=generator
        )
        - 1.0
    )
    box = torch.eye(3, dtype=torch.float64, device=device) * length
    center, source, vector = VesinNeighborSearch().compute(
        positions, box, rcut, return_vectors=True
    )
    order = torch.argsort(torch.linalg.vector_norm(vector, dim=-1), stable=True)
    n_node = torch.tensor([node_count], dtype=torch.int64, device=device)
    graph = canonical_graph_from_pairs(
        center[order], source[order], vector[order], n_node, node_count
    )
    # The operator reads the source CSR only for the per-atom virial, which
    # LAMMPS declines; it takes two-entry placeholders in its place.
    return dataclasses.replace(
        graph,
        source_row_ptr=graph.source_row_ptr.new_zeros(2),
        source_order=graph.source_order.new_zeros(2),
    )


class _Bench:
    """Times the fused operator of one model on synthetic bulk systems.

    The bench keeps four counters, which :func:`tune_fitting_routes` logs:

    - ``samples``, the samples of run sizes that it times, those of the
      bisections of bands included;
    - ``checked``, the samples that it times on trains;
    - ``overturned``, the samples in which the trains refuse the candidate of
      the largest gain in the alternating replays; a sample of two routes
      besides that of the rules, which runs no alternating replays, never
      counts;
    - ``replays``, the replays of the alternating rounds of the samples and
      those of the trains, their lead-ins included.
    """

    def __init__(self, model: Any, device: torch.device, largest: int) -> None:
        import copy

        from deepmd.pt_expt.model.make_model import (
            fused_atom_bias,
        )

        learned, pair_potential = model.atomic_model.fused_decomposition()
        self.descriptor = copy.deepcopy(learned.descriptor).to(device).eval()
        self.fitting = copy.deepcopy(learned.fitting_net).to(device).eval()
        self.atom_bias = fused_atom_bias(model.atomic_model, learned).to(device)
        self.pair_table = (
            None if pair_potential is None else pair_potential.pair_table.to(device)
        )
        self.device = device
        self.rcut = float(self.descriptor.get_rcut())
        generator = torch.Generator(device=device).manual_seed(SEED)
        self.types = min(SYNTHETIC_TYPES, int(self.descriptor.get_ntypes()))
        self.atype = torch.randint(
            0, self.types, (largest,), device=device, generator=generator
        )
        self.one_type = torch.zeros_like(self.atype)
        self.spin = None
        if getattr(self.descriptor, "spin", None) is not None:
            moments = torch.randn(largest, 3, device=device, generator=generator)
            self.spin = moments / torch.linalg.vector_norm(
                moments, dim=-1, keepdim=True
            )
        self.fingerprint = _fingerprint(device)
        self.samples = 0
        self.checked = 0
        self.overturned = 0
        self.replays = 0

    def single(self, choice: int) -> list[int]:
        """Return the table that names ``choice`` for every run size."""
        return [ROUTE_TABLE_MAGIC, *self.fingerprint, 1, choice]

    def _inputs(self, node_count: int) -> tuple[Any, torch.Tensor, Any]:
        """Return the graph, the types and the moments of one run size."""
        graph = _bulk_graph(self.rcut, node_count, self.device)
        spin = None if self.spin is None else self.spin[:node_count]
        return graph, self.atype[:node_count], spin

    def _compositions(self, inputs: tuple) -> list[tuple]:
        """Return the compositions of the trains of a sample.

        The trains run the graph of the sample with every node of the first
        node type and with the node types of the sample; for a model of one
        node type the two coincide, and the trains run it once.
        """
        graph, atype, spin = inputs
        one_type = (graph, self.one_type[: atype.shape[0]], spin)
        return [one_type] if self.types == 1 else [one_type, inputs]

    def _capture(
        self, inputs: tuple, choices: list[int], rotation: int
    ) -> dict[int, Any]:
        """Return a captured CUDA graph of the operator on ``inputs`` per route.

        The routes are captured in the order of ``choices`` rotated by
        ``rotation``, so that over the captures of a sample no route keeps one
        place in the shared memory pool.
        """
        from deepmd.pt_expt.kernels.dpa4c.canonical import (
            dpa4c_canonical_compress_energy_force,
        )

        graph, atype, spin = inputs
        stream = torch.cuda.Stream(self.device)
        graphs: dict[int, Any] = {}
        # The graphs of one sample share one memory pool; they replay one at a
        # time on one stream, and nothing reads their outputs.
        pool = None
        shift = rotation % len(choices)
        for choice in choices[shift:] + choices[:shift]:
            table = self.single(choice)

            def call(table: list[int] = table) -> Any:
                return dpa4c_canonical_compress_energy_force(
                    self.descriptor,
                    self.fitting,
                    graph,
                    atype,
                    self.atom_bias,
                    spin,
                    self.pair_table,
                    fitting_routes=table,
                )

            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                call()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, pool=pool, stream=stream):
                call()
            pool = captured.pool()
            graphs[choice] = captured
        for captured in graphs.values():
            for _ in range(WARMUP_REPLAYS):
                captured.replay()
        torch.cuda.synchronize(self.device)
        return graphs

    def _replay(
        self, graphs: dict[int, Any], replays: int, times: dict[int, list[float]]
    ) -> None:
        """Append the times of ``replays`` alternating replays of every route."""
        choices = list(graphs)
        events = []
        for replay in range(replays):
            shift = replay % len(choices)
            for choice in choices[shift:] + choices[:shift]:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                graphs[choice].replay()
                end.record()
                events.append((choice, begin, end))
        torch.cuda.synchronize(self.device)
        for choice, begin, end in events:
            times[choice].append(begin.elapsed_time(end))
        self.replays += replays * len(choices)

    def winner(self, node_count: int, rule: int, choices: list[int]) -> int:
        """Return the fastest route on one run size.

        Within one capture of the operator the replays of the routes
        alternate, so every route replays at the same clocks of the device,
        and the gain of a route over the route of the rules is the ratio of
        their median replay times, less one. A route is a candidate where its
        median gain over the captures of the sample exceeds its bound
        (:func:`_bound`); a deployment captures the operator once and replays
        that capture, so a gain within the half-range of the per-capture gains
        is not one that it reliably sees. The sample takes further captures
        while the gain of any route lies near its bound, and a departure from
        the rules rests on at least three captures (:data:`STAGE_MARGINS`).
        The candidates are checked on trains of one route at a time in the
        order of their gains, and the first that holds there takes the run
        size (:meth:`_departure`).

        The bisection of a band compares the two routes that won at its ends.
        Where neither of them is that of the rules, ``choices`` holds these two
        alone, and the trains decide between them (:meth:`_faster_on_trains`);
        where one of them is, the sample runs as above.
        """
        if len(choices) == 1:
            return choices[0]
        self.samples += 1
        inputs = self._inputs(node_count)
        if rule not in choices:
            first, second = choices
            return self._faster_on_trains(inputs, first, second)
        # One untimed round brings the clocks of the device to the load of the
        # sample, and sets the replays of its captures.
        graphs = self._capture(inputs, choices, 0)
        short = self._replay_ms(graphs[choices[0]]) < SHORT_REPLAY_MS
        replays = SHORT_STAGE_REPLAYS if short else STAGE_REPLAYS
        self._replay(graphs, replays, {choice: [] for choice in choices})
        del graphs
        rivals = [choice for choice in choices if choice != rule]
        gains: dict[int, list[float]] = {choice: [] for choice in rivals}
        captures = 0
        for stage, count in enumerate(STAGE_CAPTURES):
            for _ in range(count):
                graphs = self._capture(inputs, choices, captures)
                captures += 1
                times: dict[int, list[float]] = {choice: [] for choice in choices}
                self._replay(graphs, replays, times)
                del graphs
                medians = {choice: statistics.median(v) for choice, v in times.items()}
                for choice in rivals:
                    gains[choice].append(medians[rule] / medians[choice] - 1.0)
            margins = [
                statistics.median(gains[choice]) - _bound(gains[choice])
                for choice in rivals
            ]
            settled = stage > 0 or max(margins) <= 0.0
            if stage == len(STAGE_MARGINS) or (
                settled
                and all(abs(margin) > STAGE_MARGINS[stage] for margin in margins)
            ):
                break
        candidates = sorted(
            (
                choice
                for choice, margin in zip(rivals, margins, strict=True)
                if margin > 0.0
            ),
            key=lambda choice: statistics.median(gains[choice]),
            reverse=True,
        )
        return self._departure(inputs, rule, candidates)

    def _departure(self, inputs: tuple, rule: int, candidates: list[int]) -> int:
        """Return the first of ``candidates`` that holds on trains, or the rules.

        A candidate holds where its median gain over the route of the rules on
        the trains is at least ``-TIE`` for every composition of
        :meth:`_compositions`: within the tie, two routes count as equally fast,
        on trains as in the alternating replays. The check of a candidate stops
        at the first composition on which it loses by more. A sample counts as
        overturned where the trains refuse the candidate of the largest gain,
        whether a later candidate or the route of the rules then takes the run
        size.
        """
        if not candidates:
            return rule
        self.checked += 1
        for candidate in candidates:
            if all(
                self._train_gain(nodes, rule, candidate) >= -TIE
                for nodes in self._compositions(inputs)
            ):
                self.overturned += candidate != candidates[0]
                return candidate
        self.overturned += 1
        return rule

    def _faster_on_trains(self, inputs: tuple, first: int, second: int) -> int:
        """Return the faster of two routes on trains of one route at a time.

        The second route is faster where the mean, over the compositions of
        :meth:`_compositions`, of its median gains over the first on the trains
        is positive, and the first route otherwise.
        """
        self.checked += 1
        gains = [
            self._train_gain(nodes, first, second)
            for nodes in self._compositions(inputs)
        ]
        return second if statistics.mean(gains) > 0.0 else first

    def _train_gain(self, inputs: tuple, incumbent: int, rival: int) -> float:
        """Return the median gain of ``rival`` over ``incumbent`` on trains.

        Each capture times a train of each route (:meth:`_train_ms`); the
        captures alternate the order of the two routes, so that a drift of the
        clocks cancels in the median gain.
        """
        gains = []
        for capture in range(TRAIN_CAPTURES):
            graphs = self._capture(inputs, [incumbent, rival], capture)
            order = (incumbent, rival) if capture % 2 == 0 else (rival, incumbent)
            times = {choice: self._train_ms(graphs[choice]) for choice in order}
            del graphs
            gains.append(times[incumbent] / times[rival] - 1.0)
        return statistics.median(gains)

    def _train_ms(self, graph: Any) -> float:
        """Return the median replay time, in ms, of a train of one route.

        The timed replays follow a lead-in of at least :data:`LEAD_IN_MS` of
        device time, after which a board at its power limit has settled its
        clocks on the route. The lead-in accumulates device time in chunks of
        about :data:`LEAD_IN_CHUNK_MS`, sized from the time of its first
        replay. The train spans about :data:`TRAIN_MS` of device time at the
        mean replay time of the lead-in, in at least :data:`TRAIN_REPLAYS`
        replays.
        """
        replay_ms = self._replay_ms(graph)
        chunk = max(1, math.ceil(LEAD_IN_CHUNK_MS / replay_ms))
        lead_in_ms, lead_in = replay_ms, 1
        while lead_in_ms < LEAD_IN_MS:
            lead_in_ms += self._replay_ms(graph, chunk)
            lead_in += chunk
        count = max(TRAIN_REPLAYS, math.ceil(TRAIN_MS * lead_in / lead_in_ms))
        events = []
        for _ in range(count):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            events.append((begin, end))
        torch.cuda.synchronize(self.device)
        self.replays += lead_in + count
        return statistics.median([begin.elapsed_time(end) for begin, end in events])

    def _replay_ms(self, graph: Any, count: int = 1) -> float:
        """Return the time, in ms, of ``count`` consecutive replays of ``graph``."""
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count):
            graph.replay()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop)


def _bound(gains: list[float]) -> float:
    """Return the median gain that a route must exceed in the alternating replays.

    The bound is :data:`TIE`, or the half-range of the per-capture gains of
    the route where that is larger.
    """
    return max(TIE, 0.5 * (max(gains) - min(gains)))


def _choices(device: torch.device, width: int, node_count: int) -> list[int]:
    return [
        PER_LAYER,
        *(
            int(k)
            for k in torch.ops.deepmd.graph_fitting_route_choices(
                device.index or 0, width, node_count
            )
        ),
    ]


def _measure(
    bench: _Bench, ws: list[torch.Tensor], largest: int
) -> tuple[list[int], list[int]]:
    """Return the inclusive upper bounds and the choices of the measured intervals.

    The intervals cover the run sizes up to ``largest``; the last one extends
    beyond it.
    """
    device = bench.device
    width = int(ws[0].shape[1])
    din = int(ws[0].shape[0])
    bounds = [
        int(b)
        for b in torch.ops.deepmd.graph_fitting_route_bands(
            device.index or 0, width, din, largest
        )
    ]
    intervals: list[tuple[int, int, int]] = []
    low = 1
    for high in bounds:
        rule = int(torch.ops.deepmd.graph_fitting_route_rule(ws, high))
        choices = _choices(device, width, high)
        if rule not in choices:
            choices.append(rule)
        first = bench.winner(low, rule, choices)
        last = first if high == low else bench.winner(high, rule, choices)
        # Every route keeps the grids of its kernels over a block of run sizes
        # (32 k, 32 k + 32], since its panels and tiles hold 32 or 64 nodes; a
        # band whose two ends lie in one block has one winner, the rules' if
        # they take one of the two.
        lo, hi = math.ceil(low / BLOCK), math.ceil(high / BLOCK)
        if first != last and lo == hi:
            first = last = rule if rule in (first, last) else last
        if first == last:
            intervals.append((high, first, rule))
        else:
            # The winner changes inside the band: bisect over the blocks for
            # the last block that the first winner serves.
            while hi - lo > 1:
                middle = (lo + hi) // 2
                if bench.winner(BLOCK * middle, rule, [first, last]) == first:
                    lo = middle
                else:
                    hi = middle
            intervals.append((BLOCK * lo, first, rule))
            intervals.append((high, last, rule))
        low = high + 1
    # The table records the rules for every interval that they route as
    # measured, so that the operator evaluates them there as it does without a
    # table; naming their route instead would skip that evaluation and its side
    # effects, such as the loading of the fused kernels that the occupancy
    # queries of the rules perform under lazy module loading.
    merged: list[tuple[int, int]] = []
    for high, route, rule in intervals:
        choice = RULES if route == rule else route
        if merged and merged[-1][1] == choice:
            merged[-1] = (high, choice)
        else:
            merged.append((high, choice))
    return [high for high, _ in merged[:-1]], [choice for _, choice in merged]


def tune_fitting_routes(
    model: torch.nn.Module, level: int, device: torch.device
) -> None:
    """Measure the fitting routes of a compressed DPA4C model on ``device``.

    A no-op for a model that does not take the fused energy-force operator of
    compressed DPA4C, and for a network and device whose table is already
    measured in this process.

    Parameters
    ----------
    model : torch.nn.Module
        The model about to be frozen.
    level : int
        The ``DP_TRITON_INFER`` level, which the measurement does not read.
    device : torch.device
        The CUDA device the frozen model is compiled for.
    """
    del level
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        canonical_model_eligible,
        op_available,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        fitting_operator_arguments,
        node_tile,
    )

    if not op_available() or not canonical_model_eligible(model):
        return
    learned, _ = model.atomic_model.fused_decomposition()
    key = network_key(learned.fitting_net)
    if key is None:
        return
    fingerprint = _fingerprint(device)
    if (fingerprint, key) in _ROUTE_TABLES:
        return
    width = key[1][0]
    # The frozen operator evaluates runs of at most its node tile, which the
    # freeze reads as this process does.
    largest = node_tile() or MAX_RUN_NODES
    if not any(
        torch.ops.deepmd.graph_fitting_route_choices(device.index or 0, width, nodes)
        for nodes in (1, largest)
    ):
        # No fused shape of the width runs on the device, so the per-layer
        # kernels serve every run, as the rules route them.
        return
    start = time.perf_counter()
    # The neighbor search of the samples and the events that time their
    # replays act on the current device, so the measurement makes the device
    # of the freeze current; the search launched on another device than that
    # of its positions would read them out of bounds.
    with torch.cuda.device(device), torch.no_grad():
        bench = _Bench(model, device, largest)
        network = fitting_operator_arguments(bench.fitting)
        bounds, choices = _measure(bench, list(network.weights), largest)
    table = (ROUTE_TABLE_MAGIC, *fingerprint, len(choices), *bounds, *choices)
    _ROUTE_TABLES[(fingerprint, key)] = table
    log.info(
        "Measured the fitting routes of hidden width %d on %s in %.1f s "
        "(%d samples, %d of them checked on trains of one route and %d of "
        "these overturned, %d replays): %d intervals, bounds %s, choices %s",
        int(network.weights[0].shape[1]),
        torch.cuda.get_device_name(device),
        time.perf_counter() - start,
        bench.samples,
        bench.checked,
        bench.overturned,
        bench.replays,
        len(choices),
        bounds,
        choices,
    )


register_autotuner(
    "dpa4c_fitting_routes",
    tune_fitting_routes,
    min_level=0,
    lower_kinds=("dpa4c_canonical",),
)
