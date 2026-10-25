# Manual ECS validation artifact (2026-10-25 measurements; 2026-10-26 publication)

## What is being compared

This artifact records **135 process outcomes: 120 completed, 15 OOM**, not 135 numerical passes. Each complete arm has three fresh processes, three identical perturbed frames repeated three times per process. The three columns are compiled pure TORCH, native master, and the isolated candidate port. Sources are pinned; raw results are not mixed across trees.

The master/candidate port uses `2ba617bc3db67adee2b8ba00c71e9fa1c2d2bb8c`, which includes merged #6058 residual-inplace. The existing personal PR still has the original668 base; **this evidence update does not silently rebase that PR or replace its production source**. `port-to-2ba617bc.patch` reproduces the tested three-file port on the pinned master. Generated source hashes are checked against the measurement manifests.

## Files

- `REPORT.md`: explicit three-column strict and empirical-band PASS/FAIL, per-output errors, P50/P90, CUDA Event time, first/full/steady allocated peaks, changes and route limits.
- `ACCURACY_TABLE.md`: separate within/across-process and implementation comparisons, strict failures and ranges.
- `summary.json`: complete aggregate metrics including mean_abs/max_relative, finite status, route counts, checkpoint identity and per-process peaks.
- `runs.json`: all135 original worker result/manifest records, including per-call timing samples and exceptions. OOM is retained.
- `provenance.json`: pinned sources, per-run NPZ hashes, source-consistency checks and publication-time environment snapshot.
- `validation.json`: prior60-test focused CUDA result and the published worker's separate complete Neo96/GPU1 smoke (3warmups/10samples,finite). The GPU0 attempt correctly stopped on a busy GPU. Smoke is not pooled into the135-run matrix and is not a new cross-reference accuracy acceptance.
- `worker.py`, `run_matrix.py`, `analyze.py`, `report.py`: standalone manual tools, not added to routine GPU CI. `pro_cpu_load_check.json` is the historical metadata-compatibility proof for the verified Pro checkpoint, not a new numerical reference.

The original input/output NPZ archives total approximately47MiB and remain in the local experiment archive; they are **not committed here**. JSON timing/config/error summaries are inspectable, and the published harness permits fresh reproduction, but the published subset does not itself permit recomputing every original output comparison. NPZ hashes in provenance support later transfer/integrity checking; a hash is not a substitute for the arrays.

## Checkpoints and scope

Neo: DPA4-Neo-OMat24-v20260627.pt, SHA256 `8d8b2b02b9575cb9e0af420cc1e8b1d9ffcbc04752c2203d29f48ccd0cd5f07d`. Nano/Mini: official OMat24 v20260805; SHA256 and loaded counts in manifests. Pro: DPA-4.0.1-Pro-MPtrj v2026.06.05, **not OMat24-Pro**; SHA256 `446a6e893101fc52190cfccabb9d2328863d651aca36981fd8d3401c71a61221`, official MD5 `2174bb3095b197489f978d31393c1e06`,273318000 bytes. [Official Pro record](https://doi.org/10.6084/m9.figshare.32647224).

Pro's historical fitting type alias is mapped in memory only (`ener -> dpa4_ener`), equally for all arms. Copy the included proof as `cpu_load_check.json` beside the verified Pro checkpoint before using the worker. This proof records strict CPU load on the historical reader; it is not evidence of passing inference on every current environment. The worker requires the exact verified Pro SHA; it does not patch arbitrary checkpoint files. No checkpoint tensors are changed or uploaded.

The candidate guard supports the fixed Neo layout. Nano/Mini/Pro execute **native fallback**, not optimized ECS. Their timing variation cannot be credited to ECS. Neo's measured AOT graphs specialize the single-cohort path, including4096; this is not evidence of actual bounded8192-edge runtime batches. No deployment fix is included in this evidence publication.

## Measurement protocol and limits

- RTX5090 x2, PyTorch2.11.0+cu128, Triton3.6.0, CUDA runtime12.8. Driver595.91.07 observed on the same server at publication. The supplied environment snapshot is **post-run**, not a clock/temperature trace of the timed window. This matrix did not collect continuous GPU telemetry, unlike the earlier controlled N64 study.
- Each worker checks for existing GPU compute clients before initialization. Same physical GPU for all arms/repeats of a model/system/size: Neo/Pro GPU0, Nano/Mini GPU1. Two GPUs may run distinct cases simultaneously; CPU/package contention is not excluded. No global clock/driver changes, no killed foreign process.
- FP32 mixing, TF32/AMP off, no FP16x3. Preserve original non-mixing model precision. Normal ASE energy/forces/virial/stress API includes neighbor construction.
- Diamond C3.567Angstrom; bccCu2.87Angstrom. Seed0, sigma0.02Angstrom, three exact same frames checked across arms. Diamond N96 repeats(2,2,3), N512(4,4,4), N4096(8,8,8). Cu96(4,4,3),512(8,8,4),4096(16,16,8).
- Accuracy9calls includes first compilation but those calls are not latency samples. Then8 warmups and30 synchronized wall/Event samples; separate30call steady-memory replay. Full peak takes the maximum including first execution and subsequent phases; steady reset is never substituted for first capacity. Allocated/reserved bytes retained; decimalGB=1e9B.
- Explicit cache per model/system/mode is reused in later independent repeats. The first worker was cold; later repeats benefit from that cache. All generic autotune choices are **not** frozen across arms. Therefore these independent-process quantiles do not establish a stable isolated sub-percent ECS cost/benefit. Earlier controlled N64 evidence remains a different experiment.
- Median of three process P50/P90 values, worst process capacity. Timing/accuracy/profile are distinct; no profiled timings are included. No DFT/long-MD accuracy, H20,10k-atoms, PT2/C++ or new training acceptance claim.

## Numerical criteria

Strict elementwise criterion: finite and `abs(got-ref)<=1e-5+1e-4*abs(ref)`. Max-relative denominator `max(abs(ref),1e-8)`. Energy/virial in eV, force eV/Angstrom, stress eV/Angstrom^3; same numeric tolerance across units does not imply same physical accuracy.

Meeting-band diagnostic: separately per output/model/system/size, worst same-frame cross-implementation max_abs divided by pure-TORCH's own worst repeat max_abs (within and across processes). <=3 is marked band PASS; it is an empirical sampled repeat band, not a formal error bound. Strict failures remain failures; zero self-band requires zero deviation. Missing/OOM reference is N/A, not PASS. A maximum over more repeat samples may increase; this protocol does not allow enlarging the sample set only to inflate tolerance.

## Reproduce on explicit source trees

Use a compatible CUDA/native DeePMD installation. Set source roots explicitly; the worker checks imported package provenance and avoids silently using an unrelated editable checkout. Obtain checkpoints separately under their licenses. Start with idle GPUs and new output/cache directories.

```bash
git clone https://github.com/deepmodeling/deepmd-kit.git base
git -C base checkout --detach 2ba617bc3db67adee2b8ba00c71e9fa1c2d2bb8c
git clone --shared base candidate
git -C candidate apply /absolute/path/to/this-artifact/port-to-2ba617bc.patch
```

Create a local model-map JSON with Neo/Nano/Mini/Pro mapped to absolute checkpoint paths (not committed weights). For Pro, place its checked compatibility proof beside the checkpoint as described above. Then, using the appropriate Python environment:

```bash
python run_matrix.py --base-root /absolute/base --candidate-root /absolute/candidate \
  --model-map /absolute/models.json --out-root /absolute/new-results \
  --gpu 0 --models Neo Pro
python run_matrix.py --base-root /absolute/base --candidate-root /absolute/candidate \
  --model-map /absolute/models.json --out-root /absolute/new-results \
  --gpu 1 --models Nano Mini
python report.py --root /absolute/new-results
```

The two commands can use different GPUs concurrently. Do not run two workers on one GPU. Default is all three sizes and three independent processes; `--sizes 96` permits an initial smoke, not full acceptance. Existing outcomes are not overwritten; explicit `--skip-existing` only resumes, not a new rerun. The public runner reproduces the calculation/protocol with explicit paths; orchestration/path handling is adapted from the local runner. Validate it on the target environment rather than claiming a new135-process run from `--help` success.

## Conclusion and open gates

Current-master Neo4096: P50 **156.512587->156.594862ms**, full allocated peak **22.313645->22.806378GB** (increase2.2082%). Neo96: **17.992521->18.035012ms**, full peak **0.646372->0.658145GB** (increase1.8213%). Small timing differences lack matched-autotune/process uncertainty; the observed memory increases are retained. No new positive ECS performance claim is made.

Remaining: fix real dynamic deployment dispatch; resolve scientific acceptance/OOM reference limitations; replace the fine-grained development switch with an agreed existing-level/internal policy and tuning integration where appropriate; retain optional-Triton CPU safety; add proportionate actual engagement/guard/alias tests; obtain real upstream CI approval. Keep personal PR Draft/default-off. This update is evidence for human review, **not an upstream submission or ready-to-merge declaration**.
