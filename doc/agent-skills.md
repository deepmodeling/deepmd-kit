# Agent Skills

DeePMD-kit provides official [Agent Skills](https://agentskills.io/what-are-skills) that help AI agents run
DeePMD-kit workflows in a reproducible way. These skills capture
project-specific operating knowledge—such as installation, training inputs, model
selection, deployment, LAMMPS integration, and Python inference patterns—so an
agent can turn a high-level request into concrete files, commands, and
validation steps.

The DeePMD-kit skills were initially developed in the
[Computational Chemistry Agent Skills](https://github.com/jinzhezenggroup/computational-chemistry-agent-skills)
project as part of the work described below. They are now maintained directly
in the DeePMD-kit repository under `skills/`.

## List of skills

- `deepmd-install`: Install DeePMD-kit for users. The skill probes the machine,
  records the selected environment and build in a validated plan, then follows
  either an easy path (conda, pip, Docker, offline, `dp1s`) or a source build of
  the Python package, C/C++ interface, and LAMMPS. Kokkos builds use one
  `Kokkos_ARCH_*` flag per binary. DPA4/SeZM uses `deepmd/kk`; DPA4C uses
  `dpa4spin/kk` with a compact canonical graph artifact. Detailed recipes live
  under `skills/deepmd-install/references/` and are loaded only for the selected
  path or a matching failure mode.
- `deepmd-train`: Choose a DeePMD-kit model family, then train from scratch.
  The skill uses progressive disclosure: the top-level workflow handles common
  training steps and model selection, while model-specific configuration lives
  under `skills/deepmd-train/models/` and is read only after a model is chosen.
  Current references include DPA3 and se_e2_a.
- `deepmd-finetune-dpa3`: Fine-tune DPA3 models from self-trained checkpoints,
  multi-task pretrained models, or built-in models downloaded by `dp pretrained download`.
- `deepmd-python-inference`: Run Python and CLI inference with trained or
  frozen DeePMD-kit models, including energy, force, virial, descriptor, and
  model-deviation workflows.
- `lammps-deepmd`: Prepare, explain, and run LAMMPS simulations with DeePMD-kit
  potentials, including common NVE, NVT, and NPT setups.

## Related reference

The DeePMD-kit skills were originally developed as part of the following
paper:

- Mingwei Ding, Chen Huang, Yibo Hu, Yifan Li, Zitian Lu, Xingtai Yu, Duo
  Zhang, Wenxi Zhai, Tong Zhu, Qiangqiang Gu, and Jinzhe Zeng. [Automating
  Computational Chemistry Workflows via OpenClaw and Domain-Specific
  Skills](https://doi.org/10.1021/acs.jctc.6c00622). *Journal
  of Chemical Theory and Computation*, 2026.

## Install skills

To have an agent install DeePMD-kit itself, send
[Install with an AI agent](install/install-with-agent.md) and ask it to load
`deepmd-install`.

### If you are a user

The easiest way is to send this page to your agent and ask it to install the
skills for you. Users usually do not need to perform manual installation.

### If you are an agent

If you already have a DeePMD-kit checkout, run this command from the repository
root:

```bash
npx -y skills add ./skills --skill '*' -y
```

If you do not have a checkout, the same skills can also be installed directly
from GitHub:

```bash
npx -y skills add https://github.com/deepmodeling/deepmd-kit/tree/master/skills \
    --skill '*' -y
```

If direct GitHub access fails, clone the official Gitee mirror and install from
that checkout:

```bash
git clone --depth 1 \
    https://gitee.com/deepmodeling/deepmd-kit.git \
    deepmd-kit-skill-source
npx -y skills add ./deepmd-kit-skill-source/skills --skill '*' -y
```

The examples require Node.js/npm so that `npx` is available. The Skills CLI
installs every official skill for the detected agent. To target one product,
add its agent name, for example `--agent cursor` or `--agent claude-code`. The
GitHub command lets the CLI fetch the repository; installing from an existing
checkout avoids that download. Refresh or restart the session afterward so the
installed skills are reloaded.

## Minimal verification

Ask the agent to perform a small task that exercises the installed skill
without launching an expensive calculation. For example:

- “Use the `deepmd-python-inference` skill to write a minimal Python snippet
  for loading a frozen DeePMD-kit model and evaluating one frame.”
- “Use the `deepmd-train` skill to choose between DPA3 and se_e2_a for a small
  water dataset and draft a training input, but do not start training.”
- “Use the `lammps-deepmd` skill to prepare an NVT LAMMPS input file for a
  DeePMD-kit model, and explain each command.”
- “Use the `deepmd-install` skill to plan a PyTorch CUDA source install, but
  ask me for the CUDA toolkit path, PyTorch wheel, and install prefix first.
  Do not start compiling.”
