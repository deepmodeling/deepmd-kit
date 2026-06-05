# Agent Skills

DeePMD-kit provides official [Agent Skills](https://agentskills.io/what-are-skills) that help AI agents run
DeePMD-kit workflows in a reproducible way. These skills capture
project-specific operating knowledge—such as training inputs, model
deployment, LAMMPS integration, and Python inference patterns—so an agent can
turn a high-level request into concrete files, commands, and validation steps.

The DeePMD-kit skills were initially developed in the
[Computational Chemistry Agent Skills](https://github.com/jinzhezenggroup/computational-chemistry-agent-skills)
project as part of the work described below. They are now maintained directly
in the DeePMD-kit repository under `skills/`.

## List of skills

- `deepmd-train-dpa3`: Train DeePMD-kit models with the DPA3 descriptor and the
  PyTorch backend, including input generation, neighbor-selection choices,
  training, freezing, and testing.
- `deepmd-finetune-dpa3`: Fine-tune DPA3 models from self-trained checkpoints,
  multi-task pretrained models, or built-in models downloaded by `dp pretrained download`.
- `deepmd-train-se-e2-a`: Train classical Deep Potential models with the
  `se_e2_a` descriptor, including preparation of training JSON files and
  post-training validation.
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
  Skills](https://doi.org/10.48550/arXiv.2603.25522). Accepted by *Journal
  of Chemical Theory and Computation*, 2026.

## Install skills

### If you are a user

The easiest way is to send this page to your agent and ask it to install the
skills for you. Users usually do not need to perform manual installation.

### If you are an agent

If you already have a DeePMD-kit checkout, run this command from the repository
root:

```bash
npx -y skills add ./skills -a openclaw -y
```

If you do not have a checkout, the same skills can also be installed directly
from GitHub:

```bash
npx -y skills add https://github.com/deepmodeling/deepmd-kit/tree/master/skills \
    -a openclaw -y
```

The examples above require Node.js/npm so that `npx` is available, and they
install the skills for OpenClaw. Replace `openclaw` with the target agent name
when installing for another agent. The GitHub command lets the skill CLI fetch
the repository for you. For large repositories or slow networks, this can take
longer than installing from an existing local checkout. Refresh or restart the
session afterward so the installed skills are reloaded.

## Minimal verification

Ask the agent to perform a small task that exercises the installed skill
without launching an expensive calculation. For example:

- “Use the `deepmd-python-inference` skill to write a minimal Python snippet
  for loading a frozen DeePMD-kit model and evaluating one frame.”
- “Use the `deepmd-train-dpa3` skill to draft a small DPA3 training input for a
  water dataset, but do not start training.”
- “Use the `lammps-deepmd` skill to prepare an NVT LAMMPS input file for a
  DeePMD-kit model, and explain each command.”
