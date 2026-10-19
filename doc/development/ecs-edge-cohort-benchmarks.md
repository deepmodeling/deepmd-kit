# Experimental SO(2) edge-cohort inference backward

**Update:** the cold-process numbers below remain historical measurements, but
their earlier fixed-3%-noise interpretation is retired. See the
[controlled timing follow-up](ecs-edge-cohort-controlled-timing.md) for aligned
autotune choices, counterbalanced same-context normal-API measurements,
process-level uncertainty and negative experiments.

This opt-in experiment targets first-order input and alpha gradients of a frozen
DPA4-Neo model. It is not a training optimization and does not change the default
inference path. Enable `DP_SO2_EDGE_COHORT_INFER=1` **before importing the model**.
Use `DP_TRITON_INFER=2`, with TF32 and AMP disabled. Other model components retain
their original precision; the checkpoint is not globally cast to FP32.

## Scope and implementation

The checked layout is `F=2`, `Cf=32`, `lmax=3`, `ROW=320`, three mixing layers
and two gated layers, contiguous FP32 activations, and `apply_alpha=True`.
Unsupported layouts retain the existing dispatcher. Weight gradients, higher
derivatives and extra saved-output gradients retain the existing training path.
The switch defaults to zero.

The existing backward traverses a full edge set for each layer. The large-edge
experiment instead traverses the whole reverse layer chain of one cohort before
advancing to the next cohort. Scratch buffers scale with cohort size (8192),
while saved forward activations and the returned gradient still scale with the
full edge count. This is not a claim that all full-edge storage disappears.

For `E <= 16384`, a single-cohort `torch.library.triton_op` exposes `wrap_triton`
launches to Inductor and uses the internally owned returned gradient as the
residual buffer. This avoids the opaque Python multi-cohort launch overhead on
small systems. It does **not** fuse the reverse chain into one mega-kernel.
16384 is a tested dispatch boundary, not a proven optimum for every workload.

Only internally owned residual elements are overwritten. Caller tensors remain
read-only, and GEMM matrix input `gz` never aliases GEMM output: independent
column tiles may still be reading `gz`. Saved and output strides account for
full-graph extent even when a cohort has a partial tail. Alpha backward remains
separate to preserve the forward-output lifetime contract. The single-cohort
residual reuse overlaps the internal inplace optimization discussed in #6058;
their percentages must not be added or presented as independent improvements.

The five GEMM schedule parameters, gate/point/recompute schedules, weights and
arithmetic are unchanged. Four native `wrap_triton` launch sites additionally
pass `saved_edges=0` / `output_edges=0` explicitly so optional Python defaults do
not become `None` in compiled native calls. Static and dynamic native tests
cover the disabled-switch path.

## Measurement protocol

The baseline is unmodified commit
`668b60cc2a9559f79abb9c33acd959833376344f`. The checkpoint is
`DPA4-Neo-OMat24-v20260627.pt`, SHA-256
`8d8b2b02b9575cb9e0af420cc1e8b1d9ffcbc04752c2203d29f48ccd0cd5f07d`.
It must be provided separately; it is not bundled in this PR.

Environment: RTX 5090, driver 595.91.07, PyTorch 2.11.0+cu128, CUDA runtime 12.8,
Triton 3.6.0. The main table uses GPU 0 only. It runs the normal ASE API for
energy, forces, virial and stress of a 64-atom diamond cell (`nrep=2`, approximately
10114 edges), seed 0, position perturbations with standard deviation 0.02 Angstrom.
Timing includes neighbor construction and host work. CUDA Events bracket the
whole API call on the default stream, including host launch gaps: they are not
the sum of kernel durations.

Three independent processes per arm run in the order BASE/ECS/ECS/BASE/BASE/ECS.
Each process uses new Inductor and Triton caches, eight warmup frames and thirty
measured frames. System/driver caches are not purged. First-call compilation
time is recorded separately and excluded from steady timings. P50 is the median
of the three process medians; P90 pools ninety samples per arm. A GPU 0 NVML
monitor checked for foreign compute processes every 250 ms; none was detected.
Clocks were recorded, not locked, and no global driver settings were changed.
The monitor covers compilation and idle time as well as execution; its clock
range is not a claim of constant clocks during timing.

**Primary memory includes the first call and compilation.** Counters are not
reset between loading the model, first execution, warmup and the thirty timed
calls. Report the worst high-water mark across the three processes for both
allocated and reserved memory. A second replay resets only peak counters to
measure steady allocated memory; allocator caches are not emptied. Reserved
memory can therefore retain the earlier cold peak. These are full-model CUDA
allocator metrics, not SO(2)-exclusive storage or total driver/context memory.
A compilation scratch allocation still consumes real capacity and is not
subtracted from the primary metric. Sampled device memory is an auxiliary
measurement and can miss narrow peaks.

Numeric results and all per-call timings are in
[the JSON evidence](ecs-edge-cohort-benchmark-results.json).

## Results: cold-process Neo N64

MB below is decimal (`1 MB = 10^6 bytes`). First and complete-window peaks
were identical in all six processes. Each arm's three allocated peak readings
were also identical. Memory columns report the worst process, not its average.

| Full-model metric | Unmodified baseline | Opt-in candidate | Change |
| --- | ---: | ---: | ---: |
| **First/complete allocated peak, MB** | **498.199552** | **480.156672** | **-3.6216%** |
| First/complete reserved peak, MB | 528.482304 | 507.510784 | -3.9683% |
| Steady allocated peak, MB, secondary | 397.536256 | 379.493376 | -4.5387% |
| Steady reserved peak, MB, cache retained | 528.482304 | 507.510784 | -3.9683% |
| Wall P50, ms, median of process medians | 12.157611 | 12.313776 | +1.2845% latency |
| Wall P90, ms, pooled 90 samples | 12.414686 | 12.432651 | +0.1447% latency |
| Wall min / max, ms | 11.702988 / 12.855804 | 12.185729 / 12.588670 | distribution, not a speedup |
| CUDA Event median, ms, pooled | 12.128464 | 12.314896 | whole API, not kernel sum |
| First call including JIT, seconds, range | 124.931-125.718 | 124.219-126.629 | excluded from steady P50 |

Baseline process P50s: 12.157611 / 11.771478 / 12.372728 ms.
Candidate process P50s: 12.398105 / 12.254008 / 12.313776 ms.
The single-batch latency difference alone is not a significance test; there is
**no claimed stable speedup**. The result is a reproducible memory reduction with no repeat
of the original approximately 7.8% small-system regression, not proof of zero
time cost. First-call compilation latency itself is approximately two minutes
in this fresh-cache protocol and is not hidden.

| Accuracy comparison, three frames x energy/forces/virial/stress | Pairs | Worst error/tolerance ratio | Result |
| --- | ---: | ---: | --- |
| Baseline self, independent processes | 3 | 0.282621 | pass |
| Candidate self, independent processes | 3 | 0.184105 | pass |
| All baseline-to-candidate process pairs | 9 | 0.256570 | pass |
| Six processes versus fresh pure-PyTorch output | 6 | 0.352841 | pass |

Within-process worst ratios are 0.256139 (baseline), 0.354324 (candidate), and
0.211513 (pure PyTorch). These characterize observed noise, not permission to
relax the absolute/relative acceptance thresholds. All compared arrays are
finite. During the full GPU 0 process windows, the highest sampled temperature
was 46 C; clock samples ranged from idle 172 MHz to 2902 MHz. No throttling
cause is inferred from those whole-window statistics.

## Correctness and limitations

Use `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)` elementwise, without
relaxing thresholds on failure. Relative error uses
`max(abs(reference), 1e-8)` in the denominator. The evidence includes max/mean
absolute error, max relative error, finite checks and the maximum error/tolerance
ratio for each frame and physical output.

The cold matrix is checked against a fresh pure-PyTorch reference with Triton
disabled, and also checks baseline-to-baseline, candidate-to-candidate and all
nine baseline-to-candidate process pairs. The pure-PyTorch reference runs on
GPU 1 in one process with three self-repeats; this is not a three-process Torch
noise matrix, and its timings are not mixed with GPU 0 performance.
Full-model floating-point reductions are not promised to be bitwise identical.

The focused CUDA test file has 60 passing tests, including zero/tail edges,
legal near-zero alpha, caller-input immutability, native static/dynamic compile,
and dispatch across 16384/16385 edges. Local experimental checks of input/alpha
gradients cover 24 edge-count/distribution cases with bitwise equality. Training
and higher-order tests check dispatch guards, not numerical correctness of a
complete training run.

Earlier opaque two-cohort N64 runs regressed by 6.85-7.81% despite reducing
memory. The compiled single-cohort branch addresses that small-system penalty;
it must not inherit the earlier 9.46% cold-peak reduction claim. Differences
are now evaluated with process-level uncertainty, not a fixed 3% cutoff. Earlier diagnostic
workers produced different baseline timings; their numbers are not mixed into
this same-entry-point comparison, and the cross-day difference is not uniquely
attributed to a hardware cause.

The latest cold-process matrix is Neo N64 only. Earlier larger-system or other
model runs with precision failures are not reclassified as passing. This draft
does not claim multi-model acceptance, standalone `.pt2`/AOTInductor/C++ deployment,
LAMMPS performance, new NCU measurements, or full training validation. The
large-edge runtime launcher remains experimental.

## Reproduce

Install a CUDA-enabled DeePMD build and its runtime dependencies, plus `ase`,
`pytest` and Triton. Use the same environment and checkpoint for both checkouts.
Create a clean baseline worktree at the commit above, and a worktree containing
this PR. Set the following paths to absolute local paths:

```bash
PY=/path/to/environment/bin/python
PR_ROOT=/path/to/pr-checkout
BASE_ROOT=/path/to/baseline-checkout
MODEL=/path/to/DPA4-Neo-OMat24-v20260627.pt
RESULTS=/path/to/new-results
BENCH=$PR_ROOT/source/tests/pt_expt/kernels/benchmark_so2_edge_cohort.py

# Each --out must be new. Default caches are inside that output directory.
$PY "$BENCH" --gpu 0 --mode torch --source-root "$BASE_ROOT" \
  --model "$MODEL" --out "$RESULTS/torch"
$PY "$BENCH" --gpu 0 --mode base --source-root "$BASE_ROOT" \
  --model "$MODEL" --out "$RESULTS/base-0" \
  --reference "$RESULTS/torch/outputs.npz"
$PY "$BENCH" --gpu 0 --mode ecs --source-root "$PR_ROOT" \
  --model "$MODEL" --out "$RESULTS/ecs-0" \
  --reference "$RESULTS/torch/outputs.npz"
```

Run three processes per arm in the order specified above, with unique output
directories; do not choose the fastest process. GPU 0 must be idle at launch.
The benchmark refuses a nonempty explicit `--cache-dir` unless
`--allow-warm-cache` is given; warm-cache results are a different experiment.
`--nrep 4` tests 512 atoms and `--nrep 8` tests 4096, but the current N64 result
does not establish their acceptance. A failed precision check exits nonzero.

The CLI writes `result.json`, `manifest.json`, `outputs.npz`, first-call memory
and compiler caches. It records global memory including validation, in addition
to the primary normal-inference window and separate steady replay, and records
source hashes. The public CLI has also been run end-to-end on Torch/base/ECS
with fresh caches; its GPU 1 timings are kept separate from the main matrix.

For generated-input CUDA tests, use a correctly installed PR checkout:

```bash
DP_TRITON_INFER=2 DP_TF32_INFER=0 DP_AMP_INFER=0 \
  $PY -m pytest source/tests/pt_expt/kernels/test_so2_edge_cohort.py -q
```

No weights or generated compiler caches are tracked in the repository.
