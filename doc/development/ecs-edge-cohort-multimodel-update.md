# ECS multi-model and N4096 follow-up (2026-10-19)

This is a measurement update for the default-off Draft PR, not a claim of multi-model production acceptance. The production implementation is unchanged from PR head `9e2e4c5de1b90cf2d6bc8b2706bc103a06778dae`; BASE is pinned to `668b60cc2a9559f79abb9c33acd959833376344f`, not subsequent upstream master.

## Read the results by execution path

- **PR candidate:** unmodified current PR code. In the measured Neo make_fx/AOT deployment graphs, the small-edge single-cohort branch is specialized even for N4096. Those runs are **not** evidence of an 8192-edge workspace bound at large E.
- **ECS_RUNTIME diagnostic:** separate processes set `SINGLE_COHORT_MAX_EDGES=0` before compilation, without editing production files. Runtime instrumentation confirms `(E, capacity, batches)` of `(80898,8192,10)`, `(158002,8192,20)` and `(647170,8192,80)`. At N4096 the final batch contains two edges. This override is not a fix submitted in the PR.
- **Nano, Mini, Pro:** the Neo-specific eligibility guard falls back to the native implementation. These are compatibility/repeatability observations, **not evidence of ECS acceleration or of generalized ECS support**.

The associated [snapshot](ecs-edge-cohort-multimodel-summary.md), [full accuracy table](ecs-edge-cohort-multimodel-accuracy.md), and [machine-readable summary](ecs-edge-cohort-multimodel-results.json) retain incomplete runs and OOM explicitly. Pro measurements are still in progress at this snapshot; no pending cell is a pass.

## Protocol and scientific gate

RTX 5090; PyTorch 2.11.0+cu128, Triton 3.6.0; strict level-2 FP32 mixing, TF32/AMP disabled. Normal ASE energy/forces/virial/stress evaluation, diamond carbon at 3.567 Angstrom, seed0 and three 0.02-Angstrom-perturbed frames. Neo/Nano/Mini use N512/1000/4096; Pro uses N64/512/4096. For each model/size/mode: three independent processes, three frames, three repeats per frame. Completed groups therefore retain 27 evaluations per mode. Every comparison uses the same frame; inputs and model identity are checked. All modes for a model/size use the same physical GPU.

TORCH means the normal compiled path with `DP_TRITON_INFER=0`, not eager execution or an exact reference. BASE uses level2; PR candidate additionally enables `DP_SO2_EDGE_COHORT_INFER=1`. Each model/mode has its own cache reused by subsequent repeats; this is not a latency study with matched autotune selections. The short user-requested pause does not become a timing sample.

Every output must be finite and satisfy `abs(candidate-reference) <= 1e-5 + 1e-4*abs(reference)`. Max-relative uses `max(abs(reference),1e-8)`. Summary ratios are worst elementwise error/tolerance; <=1 is within the criterion. Full tables include energy, force, virial and stress separately and distinguish within-process repeats from across-process repeats. Ranges are min/max of pairwise max-absolute error, not confidence intervals. BASE or TORCH self-failure does **not** turn a candidate failure into a pass. All-pairs comparisons are descriptive, not independent statistical replicates.

The N4096 Neo compiled TORCH path OOMed in all three processes. BASE self-repeat worst ratio is 2.854, PR self-repeat 2.854, PR-vs-BASE 3.239; actual runtime batching-vs-BASE is 3.527. Thus **N4096 is not accepted under the fixed criterion**. Existing fluctuations complicate attribution, but do not establish that the candidate is harmless. No threshold was relaxed.

Energy and virial are in eV, force in eV/Angstrom, stress in eV/Angstrom^3; identical numerical tolerances across units are not equivalent physical tolerances. This is implementation-equivalence testing on perturbed carbon crystals, not DFT predictive-accuracy or long-MD validation.

## N4096 historical performance, not a current-PR speed claim

The following is explicitly the **2026-10-18 runtime prototype**, each arm one GPU0 process, 8 warmups and 30 synchronized wall-clock samples through the normal full API, plus separate memory replay. First/JIT execution is excluded from steady timing and included in the full memory window. Decimal GB=1e9 bytes. It is not the controlled N64 latency study and not a new performance measurement of the current PR.

| Strategy | Frame P50 ms | Time reduction | Steady allocated GB | Steady reduction | Full-window peak including first GB | Full-peak reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical BASE | 156.495125 | — | 23.970400 | — | 24.061593 | — |
| Historical runtime ECS prototype | 152.996582 | 2.2356% | 22.256657 | 7.1494% | 22.256657 | 7.5013% |

BASE/candidate P90: 156.614295/153.179559 ms. Candidate first-call-only allocated peak is 22.255608 GB; the full window is slightly higher at 22.256657 GB and is the primary capacity value above. The historical third-frame virial check failed (error/tolerance 1.394), and TORCH OOMed. These are **unaccepted historical observations**, not a stable speedup claim or a substitute for new precision/latency acceptance. The newer multi-process results further prohibit treating this historical row as accepted.

[Historical per-call timing, memory and accuracy JSON](ecs-edge-cohort-n4096-historical-results.json) contains both original result objects and the aggregate row. It is deliberately separate from the current accuracy snapshot.

## Model identity

- Neo: `DPA4-Neo-OMat24-v20260627.pt`, SHA256 `8d8b2b02b9575cb9e0af420cc1e8b1d9ffcbc04752c2203d29f48ccd0cd5f07d`.
- Nano/Mini: official OMat24 v20260805 checkpoints; exact SHA256 and loaded parameter counts are retained in the machine-readable snapshot.
- Pro: **DPA-4.0.1-Pro-MPtrj v2026.06.05**, not OMat24-Pro. [Official registry](https://github.com/janosh/matbench-discovery/blob/main/models/deepmd/dpa-4.0.1-pro-mptrj.yml), [official record](https://doi.org/10.6084/m9.figshare.32647224). Downloaded mirror revision `452f673645030b1daeda91c75cf6e1b1c346ee5f` of `JinukMoon/oh-my-mlip-mirror-dpa4`; 273318000 bytes, official MD5 `2174bb3095b197489f978d31393c1e06`, SHA256 `446a6e893101fc52190cfccabb9d2328863d651aca36981fd8d3401c71a61221`, both verified.
- Pro's legacy checkpoint fitting type `ener` is unsupported by the current reader. All three modes apply the same **in-memory metadata-only** alias `ener -> dpa4_ener`, referring to the same historical SeZMEnergyFittingNet class; tensor contents and original file are unchanged. Strict CPU state loading has zero missing/unexpected keys. Therefore Pro BASE is the native computation with disclosed reader compatibility, not proof that the unmodified default reader loads this checkpoint directly.

## Reproduction and limitations of published evidence

The public benchmark already provides normal API runs for the three primary modes. For example, from the candidate checkout, with a prebuilt compatible CUDA installation, an idle selected GPU and separate output/cache directories:

```bash
python source/tests/pt_expt/kernels/benchmark_so2_edge_cohort.py \
  --source-root /absolute/path/to/base-checkout --model /absolute/path/to/neo.pt \
  --mode base --gpu 0 --nrep 8 --warmup 8 --samples 30 \
  --out /absolute/path/to/new-base-output --cache-dir /absolute/path/to/new-base-cache
python source/tests/pt_expt/kernels/benchmark_so2_edge_cohort.py \
  --source-root /absolute/path/to/candidate-checkout --model /absolute/path/to/neo.pt \
  --mode ecs --gpu 0 --nrep 8 --warmup 8 --samples 30 \
  --out /absolute/path/to/new-ecs-output --cache-dir /absolute/path/to/new-ecs-cache
```

`nrep=2,4,5,8` corresponds to diamond N64,512,1000,4096. Use `--mode torch` for the compiled TORCH reference. Repeat independently with fresh outputs; do not compare across cards or source revisions. These commands exercise the current PR's deployment route and **do not force runtime batching**, and are not an exact replay of the new three-frame/repeat all-pairs matrix. Generated graph route inspection is mandatory.

This supplement publishes complete aggregate metrics/route counts and the historical per-call result objects, not model weights, compiler caches, full matrix output-array archives or its independent orchestration harness. Therefore it does not claim that the original full matrix can be audited end-to-end from GitHub alone yet. Existing focused unit tests do not replace full-model numerical acceptance or PT2/C++ deployment validation. Keep this PR Draft and default-off; dynamic dispatch correction, unresolved accuracy failures and portable full-matrix reproduction remain gates.
