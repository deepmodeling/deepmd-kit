# ECS controlled timing follow-up

This follow-up investigates the previously reported N64 latency increase of
1.2845%. The kernel implementation is unchanged. It replaces the earlier
"less than 3% means noise" interpretation with counterbalanced measurements
and process-level uncertainty. No stable speedup or exact zero-cost guarantee
is claimed.

## Main finding

Four independent processes each held both implementations in **one CUDA
context**, using normal ASE energy/force/virial/stress calls. Two Python hash
seeds (0, 1) were each tested with baseline-first and ECS-first compilation.
Each process ran 24 symmetric ABBA blocks, 50 calls per round, eight warmups
before each round: 9600 measured calls per arm, 96 blocks in total.

| Metric | Baseline | ECS |
| --- | ---: | ---: |
| Pooled normal-API P50, ms | 11.683581 | 11.708142 |
| Pooled normal-API P90, ms | 11.911459 | 11.932993 |
| Direct ratio of pooled P50s | reference | +0.2102% latency |
| Mean process-level paired effect | reference | **+0.0795% latency** |
| 95% Student-t interval over four independent process effects | — | **[-0.3866%, +0.5479%]** |

The paired estimate averages log ratios of matched block medians within each
process, then gives each process equal weight. It is not the ratio of pooled
medians; both are reported rather than choosing one favorable number. Frames
are correlated within processes and are not 19200 independent experiments.

The measured interval does not support a persistent +1.2845% penalty under
these controlled conditions. It also does not establish exact zero overhead,
or equivalence on other models, sizes, compilers or hardware. Four processes
remain a finite sample. There is no fixed 3% significance rule.

| Process | First compiled model | Hash seed | Paired ECS latency change |
| --- | --- | ---: | ---: |
| 0 | Baseline | 0 | +0.2999% |
| 1 | ECS | 0 | -0.1536% |
| 2 | Baseline | 1 | +0.3656% |
| 3 | ECS | 1 | -0.1924% |

Compilation/initialization order is an observed confound, not a uniquely
identified low-level cause. No claim is made that a particular cache bank,
Python hash lookup or GPU clock policy explains the entire historical result.

## Why the old result was not an isolated ECS comparison

The earlier independent cold runs also independently autotuned other Inductor
kernels. Compared with the first baseline process, the first candidate had
**41 different launch choices among 109 matching cached programs**. Other
candidate runs had 44/49 changed choices; other baseline runs themselves had
48/47 among 110. These are actual XBLOCK/warp/stage changes, not differences
in recorded autotune duration. This does not quantify how much of the 1.28%
each change caused, but it prevents assigning the whole difference to ECS.

The five SO(2) GEMM schedule values remained `(64,64,32,4,2)`. A source option
such as `max_autotune=False` does not mean every generated pointwise/reduction
kernel has no runtime choice. In the controlled cache, all 110 anchor choices
were preserved. Shared semantic programs used identical actual configs, and
835 autotune/PTX/cubin file hashes remained unchanged during each main timed
window. Nothing was retuned halfway through the comparison.

## Controls, diagnostics and rejected explanations

These protocols have different scopes; do not pool their absolute times or
call their confidence intervals interchangeable.

| Protocol | Independent groups | Paired change | Process t 95% interval / limitation |
| --- | ---: | ---: | --- |
| Two resident workers, GPU0, 12 ABBA blocks/group | 6 | +0.0206% | [-1.4295%, +1.4920%]; initialization-order bias remained |
| One worker at a time, GPU0, matched seeds, 400 formal calls/process | 3 pairs | -0.5510% | [-4.1574%, +3.1910%]; too few groups for a tight conclusion |
| Identical baseline versus baseline clone, GPU1 | 2 | +0.3511% | pair effects +0.8666% / -0.1618%; calibration only |
| Candidate source with ECS disabled versus original baseline, GPU1 | 2 | +0.5092% | pair effects +1.6908% / -0.6587%; no stable compatibility-overhead attribution |
| Same-context compiled-graph replay, GPU1 | 1 | +0.2261% | conditional block interval [+0.0706%, +0.3842%], not population/API acceptance |

The last diagnostic intentionally excludes ASE, neighbor preprocessing and the
AOT runtime shell. It provides a small-overhead observation, not a substitute
for complete API performance. This model's energy inference path is
`make_fx -> aot_module_simplified -> compile_fx_inner`, **bypassing Dynamo**;
it would be incorrect to attribute the observed cost to Dynamo inference guards.

Both GPUs were checked for foreign work on the measured card. Only one worker
executed at a time; known resident workers in the resident diagnostic were
allowed to keep idle contexts. GPU0 workers used CPU8, monitor CPU16; GPU1 used
CPU40, monitor CPU48, on their respective local NUMA nodes. NVML sampled actual
timing intervals at 50 ms. SM clocks remained dynamic, not locked. Memory
clocks, temperature, power and available clock-reason data are retained.
Whole-process idle/compilation readings are not called timing-window clocks.
No driver settings, global GPU clocks, or other users' processes were changed.

As an explicitly secondary, post-stratified check, 90/96 primary blocks meet
all of: mean sampled SM-clock difference <=0.5%, memory clock 13801 MHz in both
arms, mean temperature difference <=2 C. Their paired estimate is +0.0861%,
process t 95% interval [-0.3873%, +0.5617%], consistent with the full result.
This is a hardware-matching criterion, **not** a latency-noise cutoff. All 96
blocks remain in the primary analysis. Selection changes composition and
does not prove constant per-kernel clocks. In the full primary sampling,
baseline SM mean was 2776.84 MHz and candidate 2815.98 MHz; ten baseline
samples reported memory idle-clock 810 MHz. These readings are not hidden.

In the resident experiment, average sampled SM clocks for first/second
initialized workers were approximately 2776.89/2775.65 MHz. That near equality
does not support blaming the whole initialization-order bias on downclocking.
Dynamic per-round frequency variation still prevents claiming constant clocks.
Outliers were retained; some individual API calls exceeded 30 ms.

## Same-context setup and validation

The manual harness first compiles one normal calculator. It then copies the
exact three reviewed kernel definitions and registered backward into the
experiment process, clears the **module-level** SeZM compiled cache, and
compiles the other calculator. Each retains its separate model-local cache.
This is necessary: the shared cache key does not independently prove that the
second experimental mode produced a new graph.

The harness asserts distinct compiled wrappers and the expected ECS marker,
keeps the first model's cache unchanged, and checks all three reference frames.
Runtime transplantation is **benchmark setup only**; the production proposal
uses ordinary checked-in functions and does not generate or rewrite kernel
source. No checkpoint or production file is modified by the harness.

To audit this method against the real PR integration, the generated candidate
has the same multiset of 191 embedded function bodies and the same entire
`call` AST after symbolic-size renaming. Normalized call SHA-256:
`2eb42115a595ebef74c6ffd358fcb1a3cb9ef2f39461cc8b91cd9de89b71ebb7`.
This counts source definitions, not runtime launches. The baseline remains the
unmodified commit `668b60cc2a9559f79abb9c33acd959833376344f`.

Every same-context process passes three-frame energy/force/virial/stress
comparisons against the pure-PyTorch reference and its paired baseline, using
`atol=1e-5, rtol=1e-4`; worst error/tolerance ratio across these runs is 0.376674.
All compared outputs are finite. The reference is one fresh pure-PyTorch
process with three self-repeats, not a new multi-process Torch matrix.
The 60 focused CUDA tests were rerun: 60 passed, 14 deprecation warnings.

## Bounded specialization attempt: not adopted

A full single cohort has `count == E`, `start == 0`. Four runtime
`saved_edges=E` / `output_edges=E` operands were experimentally replaced with
the natural-stride sentinel zero. No floating arithmetic, launch count or tile
was changed. Each of two processes checked 21 generated cases:
E=0/1/37/129/8193/10114/16384, random/zero/tiny legal alpha; input gradients
were bitwise equal, finite, and caller-owned inputs unchanged.

| Small-stack input-backward only | Original P50 ms | Natural stride P50 ms | Paired change |
| --- | ---: | ---: | ---: |
| Process 0 | 0.308923 | 0.309134 | +0.0653% |
| Process 1 | 0.310015 | 0.310216 | +0.0357% |

Conditional block intervals include zero. The selected binary sets differ,
so it is not established that the compiler erased every difference; nevertheless
there is no measured benefit. This attempt is **not adopted** and is not an
end-to-end speedup. Alpha backward itself was unchanged.

## Memory and the extent of Megakernels use

The unchanged implementation retains the earlier independent cold-process
capacity result: **first/complete allocated peak 498.199552 -> 480.156672 MB
(-3.6216%)**, reserved 528.482304 -> 507.510784 MB (-3.9683%). Steady allocated
397.536256 -> 379.493376 MB (-4.5387%) is secondary. JIT/first execution is not
removed from the primary peak. Dual-model resident capacity is not substituted
for this single-model measurement.

Current ECS borrows bounded-workset/dependency scheduling ideas; it does not
integrate the HazyResearch runtime or retain the reverse chain wholly on chip.
The new representative traces show **657 GPU launches/frame in both arms,
including 32 SO(2) launches**: two stacks, each 8 forward and 8 backward kernels.
There is no mega-kernel launch-count reduction in this proposal.

The reference [Megakernels implementation](https://github.com/HazyResearch/Megakernels/blob/main/include/megakernel.cuh)
uses device loader/consumer/storer/controller roles and synchronization. Those
mechanisms are not present here. Earlier single-CTA persistent PBS experiments
in this project regressed at E=158002 (baseline backward about 5.047 ms, best of
12 PBS configurations about 8.780 ms); that route was not promoted. Fusion can
lose column parallelism, increase registers/spill and synchronization costs.
Fewer buffers or launches alone do not establish faster execution.

## Reproduce and inspect

Use the same checkpoint/environment as the original protocol. Provide absolute
paths for `BASE_ROOT`, `PR_ROOT`, `MODEL`, `TORCH_REFERENCE` and fresh result dirs.
CPU numbers below are specific to the reported dual-socket server; select valid
GPU-local cores on another host.

```bash
PY=/path/to/cuda-environment/bin/python
TOOLS=$PR_ROOT/source/tests/pt_expt/kernels

# Initialize a shared cache using normal BASE and ECS benchmark calls first.
# A nonempty explicit cache requires --allow-warm-cache in the original CLI.
$PY "$TOOLS/benchmark_so2_edge_cohort.py" --mode base --source-root "$BASE_ROOT" \
  --gpu 0 --model "$MODEL" --reference "$TORCH_REFERENCE" \
  --cache-dir "$CACHE" --out "$PREP_BASE"
$PY "$TOOLS/benchmark_so2_edge_cohort.py" --mode ecs --source-root "$PR_ROOT" \
  --gpu 0 --model "$MODEL" --reference "$TORCH_REFERENCE" \
  --cache-dir "$CACHE" --allow-warm-cache --out "$PREP_ECS"

# Main same-context normal API; repeat all four seed/order combinations.
PYTHONHASHSEED=0 $PY "$TOOLS/benchmark_so2_edge_cohort_same_context.py" \
  --baseline-root "$BASE_ROOT" --candidate-root "$PR_ROOT" \
  --model "$MODEL" --reference "$TORCH_REFERENCE" --cache "$CACHE" \
  --gpu 0 --cpu 8 --monitor-cpu 16 --first BASE --blocks 24 --out "$NEW_OUT"
# Other combinations: seed0/ECS-first, seed1/BASE-first, seed1/ECS-first.
$PY "$TOOLS/analyze_so2_edge_cohort_same_context.py" \
  "$OUT0" "$OUT1" "$OUT2" "$OUT3" --out "$SUMMARY"
$PY "$TOOLS/analyze_so2_edge_cohort_clocks.py" \
  "$OUT0" "$OUT1" "$OUT2" "$OUT3" --out "$CLOCK_SENSITIVITY"

# Secondary serialized control (never two resident contexts on the card).
$PY "$TOOLS/benchmark_so2_edge_cohort_serial.py" \
  --baseline-root "$BASE_ROOT" --candidate-root "$PR_ROOT" \
  --model "$MODEL" --reference "$TORCH_REFERENCE" --cache "$CACHE" \
  --gpu 0 --cpu 8 --monitor-cpu 16 --seeds 0 1 2 --rounds 8 --samples 50 \
  --conditioning-samples 200 --out "$SERIAL_OUT"
$PY "$TOOLS/analyze_so2_edge_cohort_serial.py" "$SERIAL_OUT"
```

`benchmark_so2_edge_cohort_paired.py` and its analyzer reproduce the secondary
resident-worker ABBA experiment (`--pairs 6 --blocks 12 --profile`). It is not
the primary estimator because of the observed resident-process bias.
Original standalone CLI behavior remains unchanged when no control hook is
provided; the shared measurement body is AST-identical to the measured copy.

[Summary JSON](ecs-edge-cohort-controlled-results.json) includes complete
configuration, uncertainty, cache-choice changes and limitations.
[Compressed raw JSON](ecs-edge-cohort-controlled-raw.json.gz) includes per-call
wall/Event samples, conditional experiments, correctness and representative
profile summaries. It excludes private manifests, model weights and compiler
caches. Its SHA-256 is recorded in the summary; decompress with `python -m gzip
-d` or read with Python's `gzip.open`.

Unresolved gates remain: wider model/size matrices, large-edge make_fx/AOT
dispatch and deployment, PT2/C++ acceptance, and the exact low-level cause of
initialization bias. Passing a Dynamo boundary unit test is not an AOT runtime
bucket-dispatch acceptance test. Default-off draft status remains unchanged.
