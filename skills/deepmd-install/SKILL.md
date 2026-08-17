---
name: deepmd-install
description: Install or rebuild DeePMD-kit with conda, pip, dp1s, offline packages, Docker, or a source checkout. Use for CPU, NVIDIA CUDA, or ROCm environments; PyTorch, TensorFlow, JAX, or Paddle backends; backend-enabled C/C++ interfaces; and DeePMD-enabled LAMMPS, including Kokkos pair styles for DPA4 and DPA4C.
---

# Install DeePMD-kit

Install the smallest runtime that satisfies the user's goal. Probe first, keep
all decisions in a validated plan, execute one self-contained gate at a time,
and verify the requested public interface rather than package presence alone.

## Supported workflows

| Path                                            | Scope                                                                                          |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| conda, pip                                      | Stable Python installs for PyTorch, TensorFlow, JAX, and Paddle; optional packaged host LAMMPS |
| `dp1s`, offline package, Docker                 | Official release artifacts selected by the user or matching documentation                      |
| source Python                                   | PyTorch, TensorFlow, JAX, and Paddle with backend-specific dependencies and verification       |
| source C/C++                                    | Backend-enabled C/C++ libraries installed into a dedicated prefix                              |
| source LAMMPS host                              | Built-in DeePMD module with exact pair-style verification                                      |
| source LAMMPS Kokkos CUDA                       | PyTorch graph artifacts: `deepmd/kk` for DPA4/SeZM and `dpa4spin/kk` for DPA4C                 |
| ROCm source, Windows source, LAMMPS plugin mode | Follow the version-matched documentation in the selected checkout                              |

Backend-neutral C/C++ libraries with `ALLOW_NO_BACKEND=ON` are outside the
automated plan. Follow `doc/install/install-from-source.md` in the selected
checkout for that layout.

## Hard rules

1. Resolve the absolute directory containing this `SKILL.md` as `SKILL_ROOT`.
   Invoke every bundled script through that absolute path.
1. Run the read-only probe before choosing versions, paths, or accelerators.
1. Ask only for a missing value that changes the selected path. Do not ask
   source, compiler, CUDA, or LAMMPS questions for an unrelated easy install.
1. Record all decisions in `install-plan.json` and validate it before an
   install, checkout, compile, download, or environment modification.
1. Use the plan's absolute Python executable. Never substitute a bare
   `python`, `pip`, or an assumed conda activation.
1. Make every shell call self-contained. Do not rely on an `export`, `cd`, or
   `conda activate` from a previous agent tool call.
1. Render commands with concrete plan values. Stop if a plan placeholder,
   empty required value, unexpected path, or shell variable not assigned
   earlier in the same command block remains. Keep each plan string in the
   quoted argument position shown by the selected reference.
1. Never run `git reset --hard`, `git clean`, recursively remove an install
   prefix, or edit shell rc files as part of this workflow. Use a new build
   directory or versioned install prefix instead.
1. Use documentation from the selected checkout for version-sensitive source
   options. Do not apply `latest` documentation blindly to an older ref.
1. Confirm before piping a remote script to a shell. Use fail-on-HTTP-error
   downloads and verify a checksum whenever the plan contains one.
1. Check GPU occupancy and bind the confirmed physical device explicitly
   before a GPU smoke test.
1. On failure, stop the current gate and read
   [`references/failure-modes.md`](references/failure-modes.md). Do not restart
   the entire install or wipe caches.

## Workflow

### 0. Probe

Run with an available Python 3.10+ interpreter:

```bash
"<absolute-python>" "<absolute-skill-root>/scripts/probe_env.py" --json
```

Use the report to identify the OS, libc, Python environments, compilers,
toolkits, driver, GPU compute capabilities and visibility mask, disk/RAM, and
existing DeePMD-kit/PyTorch installs. The probe does not select versions.

### 1. Select one path

- Prefer an easy method for a stable release without local source changes,
  custom C/C++ libraries, or Kokkos device pair styles.
- Use source Python for a branch/fork, unreleased feature, local toolkit build,
  or customized OPs.
- Add source C/C++ only when a C/C++ client, source-built LAMMPS, or i-PI needs
  it.
- Add source LAMMPS only after the C/C++ gate passes.

### 2. Create and validate the plan

Read [`references/plan-schema.md`](references/plan-schema.md), write the plan to
a stable scratch path outside the source/build trees, and validate it:

```bash
"<absolute-python>" "<absolute-skill-root>/scripts/validate_plan.py" \
    "<absolute-plan-path>"
```

Present a concise summary only when the request or probe leaves a meaningful
choice unresolved. If the user already specified the method, backend, version,
and target, proceed without asking them to reconfirm their own request.

### 3. Execute the selected references

| Selection                           | Read                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| conda, pip, `dp1s`, offline, Docker | [`references/easy-install.md`](references/easy-install.md)   |
| source Python                       | [`references/source-python.md`](references/source-python.md) |
| source C/C++                        | [`references/source-cpp.md`](references/source-cpp.md)       |
| source LAMMPS                       | [`references/source-lammps.md`](references/source-lammps.md) |
| failed gate                         | [`references/failure-modes.md`](references/failure-modes.md) |

Read only the references required by the plan. Within each gate, render one
command block with absolute values so that directory and environment state
cannot leak across calls.

### 4. Enforce the gates

| Gate        | Required evidence                                                                                                                 |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Plan        | `validate_plan.py` exits zero and prints a validation summary                                                                     |
| Environment | The selected package manager and absolute interpreter target the planned environment                                              |
| Python      | `verify_python.py` passes for the selected backend and accelerator; source builds also match the expected build variant           |
| C/C++       | Expected libraries and headers exist, dynamic dependencies resolve, and the build cache records the requested accelerator/backend |
| LAMMPS      | `verify_lammps.py` finds the exact pair styles required by the model family and no unresolved dynamic dependency                  |
| Smoke       | The selected short example finishes on the explicitly bound device and emits its documented success signal                        |

For PyTorch source builds, install the intended PyTorch before DeePMD-kit and
use `--no-build-isolation`. Set both `DP_ENABLE_PYTORCH` and
`DP_ENABLE_TENSORFLOW` explicitly in the same command that invokes pip.

## LAMMPS model contract

| Model/runtime       | Host pair style | Kokkos CUDA pair style                                      | Artifact requirement                                      |
| ------------------- | --------------- | ----------------------------------------------------------- | --------------------------------------------------------- |
| conventional models | `deepmd`        | `deepmd/kk` when the model supports device edge/graph input | Compatible frozen model; `/kk` requires edge/graph `.pt2` |
| DPA4 / SeZM         | `deepmd`        | `deepmd/kk`                                                 | Target-specific graph `.pt2`                              |
| DPA4C native spin   | `dpa4spin`      | `dpa4spin/kk`                                               | `atom_style spin` and compact canonical graph `.pt2`      |

Do not accept `deepmd/kk` as proof that DPA4C is available. Do not substitute
LAMMPS `PKG_GPU` for Kokkos.

## Completion report

Report:

- plan path and resolved source commit, when applicable;
- environment activation or absolute Python path;
- installed DeePMD-kit/backend versions and accelerator visibility;
- C/C++ prefix and LAMMPS binary, when applicable;
- each gate command and observed result;
- any validation gate that was not run and the concrete reason;
- no claim of success beyond the gates that actually passed.
