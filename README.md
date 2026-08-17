[<picture><source media="(prefers-color-scheme: dark)" srcset="./doc/_static/logo-dark.svg"><source media="(prefers-color-scheme: light)" srcset="./doc/_static/logo.svg"><img alt="DeePMD-kit logo" src="./doc/_static/logo.svg"></picture>][logo-guide]

# DeePMD-kit

**Start from a pretrained Deep Potential model, fine-tune it for your system,
and deploy it at simulation scale.**

[![GitHub release](https://img.shields.io/github/v/release/deepmodeling/deepmd-kit)][releases]
[![offline packages](https://img.shields.io/github/downloads/deepmodeling/deepmd-kit/total?label=offline%20packages)][releases]
[![conda-forge](https://img.shields.io/conda/dn/conda-forge/deepmd-kit?color=red&label=conda-forge&logo=conda-forge)](https://anaconda.org/conda-forge/deepmd-kit)
[![pip install](https://img.shields.io/pypi/dm/deepmd-kit?label=pip%20install)](https://pypi.org/project/deepmd-kit/)
[![docker pull](https://img.shields.io/docker/pulls/deepmodeling/deepmd-kit)](https://hub.docker.com/r/deepmodeling/deepmd-kit)
[![Documentation Status](https://readthedocs.org/projects/deepmd/badge/)][documentation]
[![License](https://img.shields.io/badge/license-LGPL--3.0--or--later-00a98f)](./LICENSE)

[**Pretrained models**][pretrained] · [**Fine-tuning**][finetune] ·
[**Documentation**][documentation] · [**Quick start**][quick-start] ·
[**Model guide**][model-guide] · [**Tutorials**][tutorials] ·
[**Examples**](./examples) · [**Releases**][releases]

> [!IMPORTANT]
> **A pretrained model can be your starting point, not just your end result.**
> Download a built-in DPA checkpoint, fine-tune the full model, or use a
> [DPA-4 LoRA adapter][dpa4-lora] with PyTorch single-task training, then test,
> export, and deploy it through the same DeePMD-kit workflow.

DeePMD-kit turns quantum-mechanical reference data into fast, scalable
interatomic potentials. Use it across molecular and materials science—from
finite molecules and covalent systems to periodic solids and metals—and scale
from laptop fine-tuning to distributed training and MPI-parallel molecular
dynamics.

<p align="center">
  <img alt="DPA4 model family Pareto frontier for Matbench Discovery CPS and saturated inference throughput" src="./doc/_static/dpa4-cps-throughput.webp" width="1080">
</p>

<p align="center"><em>The DPA4 model family traces a Pareto frontier across Matbench Discovery CPS and saturated inference throughput.</em></p>

## ⚡ Why DeePMD-kit

|     | Advantage                           | What it unlocks                                                                                                                                                                                                          |
| --- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 🧬  | **Pretrained-first workflows**      | Download [pretrained DPA models][pretrained], fine-tune full models, use [DPA-4 LoRA adapters][dpa4-lora] with PyTorch single-task training, or adapt learned representations to downstream properties with [DPA-ADAPT]. |
| 🏗️  | **Training from scratch**           | Design a model for a new system or physical target, then train it with single-task, multi-task, and distributed workflows across supported backends.                                                                     |
| 🧠  | **Modern model portfolio**          | Start with efficient DeepPot-SE descriptors or move to [DPA][model-guide] for large atomistic models.                                                                                                                    |
| 🧲  | **More than energy and force**      | Model virials, Hessians, spin and magnetic forces, dipoles, polarizabilities, electronic density of states, atomic populations, and arbitrary intensive or extensive properties.                                         |
| 🔄  | **Backend flexibility**             | Train or run supported models with [TensorFlow, PyTorch, JAX, or Paddle][backends], with backend-aware model formats and conversion paths for compatible architectures.                                                  |
| 🚀  | **Performance from training to MD** | Use CPUs, CUDA GPUs, ROCm source builds, distributed training, model compression, compiled DPA-4 paths, AOTInductor `.pt2` export, and MPI-enabled simulation.                                                           |
| 🔌  | **Deploy where science happens**    | Use the CLI, Python, C, C++, or Node.js, then connect models to LAMMPS, i-PI, ASE, GROMACS, JAX MD, nvalchemi, OpenMM, Amber, CP2K, ABACUS, and more.                                                                    |
| 🧩  | **Open and extensible**             | Compose hybrid potentials, add analytical ZBL or long-range corrections, create custom models and operators, or connect external GNNs such as MACE and NequIP through plugins.                                           |

> [!TIP]
> On supported descriptors and workloads, [model compression][compression] can
> deliver more than **10× inference speedup** and reduce memory usage by as much
> as **20×**. Actual gains depend on the model, system, and hardware.

Backend and interface support varies by model and feature. The
[web documentation][documentation] marks compatibility and limitations on each
feature page.

## 🧭 Two starting points, one path to dynamics

```mermaid
flowchart LR
    A["Pretrained DPA model"] --> C["Fine-tune on target data"]
    B["Model configuration"] --> D["Train from scratch"]
    E["Target reference data"] --> C
    E --> D
    C --> F["Test, compress, export"]
    D --> F
    F --> G["Python and native APIs"]
    F --> H["Molecular dynamics"]
```

1. **Choose a starting point:** download a pretrained DPA checkpoint for
   adaptation, or configure a model to train from scratch.
1. **Prepare target data** in DeePMD's NumPy format or convert structures and
   trajectories with [dpdata][data].
1. **Fine-tune or train:** adapt the full pretrained model, use
   [DPA-4 LoRA adapters][dpa4-lora] with PyTorch single-task training, or
   optimize a new model with single-task, multi-task, and distributed training
   workflows.
1. **Validate and export** with [`dp test`][testing], [`dp freeze`][freeze],
   backend conversion, embedding extraction, and supported compression paths.
1. **Run simulation** through Python or native APIs, or load the model into a
   supported molecular-dynamics engine.

## 🚀 Start in minutes

DeePMD-kit requires Python 3.10 or later. The fastest installation path is:

```bash
curl -fsSL https://dp1s.deepmodeling.com | bash
dp --version
dp -h
```

The [installation guide][installation] covers pip, conda-forge, containers,
offline packages, GPU builds, LAMMPS, i-PI, and source installation.

### Fine-tune from a pretrained DPA model

Download a built-in checkpoint, inspect its branches, and fine-tune the branch
that matches your target system:

```bash
dp pretrained download DPA-3.2-5M
dp --pt show ~/.cache/deepmd/pretrained/models/DPA-3.2-5M.pt model-branch
dp --pt train input.json \
    --finetune ~/.cache/deepmd/pretrained/models/DPA-3.2-5M.pt \
    --model-branch <branch> \
    --use-pretrain-script
```

`DPA-3.2-5M` is a PyTorch multi-task checkpoint: run the trainer in PyTorch
mode with `dp --pt` and select the branch that matches your system with
`--model-branch` (list them with
`dp --pt show ~/.cache/deepmd/pretrained/models/DPA-3.2-5M.pt model-branch`). The
`--use-pretrain-script` option imports that branch's descriptor and fitting
configuration, so `input.json` does not need to reproduce the DPA-3.2
architecture.

The [fine-tuning guide][finetune] covers full-model adaptation. [DPA-4 LoRA
fine-tuning][dpa4-lora] is available for PyTorch single-task training.
[DPA-ADAPT] reuses pretrained DPA representations for downstream
property-prediction tasks.

Pretrained model names can also be resolved and cached automatically by
Python:

```python
from deepmd.infer import DeepPot

potential = DeepPot("DPA-3.2-5M")
```

### Train a model from scratch

Training from scratch remains a first-class workflow for new architectures,
fully custom systems, and physical targets without a suitable pretrained
checkpoint. Clone the examples and start with the compact water system:

```bash
git clone https://github.com/deepmodeling/deepmd-kit.git
cd deepmd-kit/examples/water/se_e2_a

# TensorFlow backend
dp train input.json

# Or PyTorch
dp --pt train input_torch.json
```

Ready-to-run inputs include:

- [DPA-3 water training](./examples/water/dpa3/input_torch.json)
- [DPA-4 water training](./examples/water/dpa4/input.json)
- [Multi-task training](./examples/water_multi_task/pytorch_example/input_torch.json)
- [DPA-ADAPT property prediction](./examples/dpa_adapt/README.md)

For a guided end-to-end example, open the [web quick-start notebook][quick-start].

## 🧠 Choose a model family

DeepPot-SE is a strong default: efficient, established, and broadly supported.
For large atomistic models, start with [DPA-4](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html).

Use the [model guide][model-guide] to compare model families, supported backends,
targets, data formats, precision, compression, and deployment constraints.

<p align="center">
  <img alt="DPA4 energy and force accuracy versus saturated throughput" src="./doc/_static/dpa4-performance.webp" width="1200">
</p>

<p align="center"><em>DPA4 provides a family of accuracy–throughput trade-offs for different deployment budgets.</em></p>

## 🔬 Go beyond conventional force fields

| Goal                                   | DeePMD-kit capabilities                                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Potential-energy surfaces**          | Energy, atomic forces, virials, Hessians, hybrid descriptors, pair tables, and linear model combinations           |
| **Magnetic systems**                   | Spin-aware descriptors, atomic and magnetic forces, and spin-capable molecular dynamics                            |
| **Electronic and response properties** | Dipoles, polarizabilities, density of states, atomic charge populations, and custom property heads                 |
| **Long- and short-range physics**      | DPLR electrostatics, DPRc range correction for QM/MM, and analytical ZBL bridging                                  |
| **Representation learning**            | Per-atom descriptors, fitting-network features, structural embeddings, clustering, and downstream auxiliary models |

Explore the complete set of [models and physical targets][model-guide] in the
web documentation.

## 🔌 Deploy into the scientific ecosystem

### Inference interfaces

- [Python][python-inference]
- [C and C++][native-inference]
- [Node.js][node-inference]
- [Model embedding export][embeddings]

### Simulation and workflow integrations

- [LAMMPS], [i-PI][ipi], [ASE],
  [JAX MD][jax-md], and [nvalchemi]
- Ecosystem integrations for OpenMM, Amber, CP2K, GROMACS, ABACUS, DP-GEN, and
  MLatom
- External MACE and NequIP models through the DeePMD-GNN plugin

See the [integration hub][integrations] for maintained interfaces, third-party
projects, supported scope, and installation guidance.

The native C and C++ interfaces load machine-learning backends as runtime
plugins. Applications can therefore open the backend required by a model
without directly linking every framework.

> [!NOTE]
> Working with an AI coding or scientific agent? Start with
> [Install with an AI agent][agent-install], or browse the
> [official Agent Skills][agent-skills] for model selection, training,
> fine-tuning, Python inference, and LAMMPS workflows.
>
> ```bash
> npx -y skills add https://github.com/deepmodeling/deepmd-kit/tree/master/skills \
>     --skill deepmd-install -y
> ```
>
> If direct GitHub access fails, use `gh-proxy.com` for a public, read-only
> clone, then install from the local checkout. Do not send credentials or
> private repository URLs through the proxy.
>
> ```bash
> git clone --depth 1 \
>     https://gh-proxy.com/https://github.com/deepmodeling/deepmd-kit.git \
>     deepmd-kit-skill-source
> npx -y skills add ./deepmd-kit-skill-source/skills \
>     --skill deepmd-install -y
> ```

## 📚 Documentation and community

- Read the [full web documentation][documentation].
- Follow hands-on material in the [DeepModeling tutorials][tutorials].
- Browse [examples](./examples) for training, inference, and integrations.
- Ask questions or report problems in [GitHub Issues](https://github.com/deepmodeling/deepmd-kit/issues).
- Join development through the [contributing guide](./CONTRIBUTING.md).

## Citation

If DeePMD-kit contributes to published work, cite the general software paper
that matches the version used and the method-specific papers listed in
[CITATIONS.bib](./CITATIONS.bib):

- Wang et al., “DeePMD-kit: A deep learning package for many-body potential
  energy representation and molecular dynamics,” *Computer Physics
  Communications* 228 (2018), 178–184 (describes the initial version).
  [![doi:10.1016/j.cpc.2018.03.016](https://img.shields.io/badge/DOI-10.1016%2Fj.cpc.2018.03.016-blue)](https://doi.org/10.1016/j.cpc.2018.03.016)
  [![Citations](https://citations.njzjz.win/10.1016/j.cpc.2018.03.016)](https://badge.dimensions.ai/details/doi/10.1016/j.cpc.2018.03.016)
- Zeng et al., “DeePMD-kit v2: A software package for Deep Potential models,”
  *The Journal of Chemical Physics* 159 (2023), 054801 (covers features until
  v2.2.3).
  [![doi:10.1063/5.0155600](https://img.shields.io/badge/DOI-10.1063%2F5.0155600-blue)](https://doi.org/10.1063/5.0155600)
  [![Citations](https://citations.njzjz.win/10.1063/5.0155600)](https://badge.dimensions.ai/details/doi/10.1063/5.0155600)
- Zeng et al., “DeePMD-kit v3: A Multiple-Backend Framework for Machine
  Learning Potentials,” *Journal of Chemical Theory and Computation* 21
  (2025), 4375–4385 (covers features until v3.0).
  [![doi:10.1021/acs.jctc.5c00340](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.5c00340-blue)](https://doi.org/10.1021/acs.jctc.5c00340)
  [![Citations](https://citations.njzjz.win/10.1021/acs.jctc.5c00340)](https://badge.dimensions.ai/details/doi/10.1021/acs.jctc.5c00340)

## License

DeePMD-kit is licensed under the
[GNU Lesser General Public License v3.0 or later](./LICENSE).

[agent-install]: https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install-with-agent.html
[agent-skills]: https://docs.deepmodeling.com/projects/deepmd/en/latest/agent-skills.html
[ase]: https://docs.deepmodeling.com/projects/deepmd/en/latest/third-party/ase.html
[backends]: https://docs.deepmodeling.com/projects/deepmd/en/latest/backend.html
[compression]: https://docs.deepmodeling.com/projects/deepmd/en/latest/freeze/compress.html
[data]: https://docs.deepmodeling.com/projects/deepmd/en/latest/data/dpdata.html
[documentation]: https://docs.deepmodeling.com/projects/deepmd/en/latest/
[dpa-adapt]: https://docs.deepmodeling.com/projects/deepmd/en/latest/dpa_adapt/overview.html
[dpa4-lora]: https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html#lora-fine-tuning
[embeddings]: https://docs.deepmodeling.com/projects/deepmd/en/latest/inference/embedding.html
[finetune]: https://docs.deepmodeling.com/projects/deepmd/en/latest/train/finetuning.html
[freeze]: https://docs.deepmodeling.com/projects/deepmd/en/latest/freeze/freeze.html
[installation]: https://docs.deepmodeling.com/projects/deepmd/en/latest/install/easy-install.html
[integrations]: https://docs.deepmodeling.com/projects/deepmd/en/latest/third-party/index.html
[ipi]: https://docs.deepmodeling.com/projects/deepmd/en/latest/third-party/ipi.html
[jax-md]: https://docs.deepmodeling.com/projects/deepmd/en/latest/third-party/jaxmd.html
[lammps]: https://docs.deepmodeling.com/projects/deepmd/en/latest/third-party/lammps-command.html
[logo-guide]: https://docs.deepmodeling.com/projects/deepmd/en/latest/logo.html
[model-guide]: https://docs.deepmodeling.com/projects/deepmd/en/latest/model/index.html
[native-inference]: https://docs.deepmodeling.com/projects/deepmd/en/latest/inference/cxx.html
[node-inference]: https://docs.deepmodeling.com/projects/deepmd/en/latest/inference/nodejs.html
[nvalchemi]: https://docs.deepmodeling.com/projects/deepmd/en/latest/third-party/nvalchemi.html
[pretrained]: https://docs.deepmodeling.com/projects/deepmd/en/latest/model/pretrained.html
[python-inference]: https://docs.deepmodeling.com/projects/deepmd/en/latest/inference/python.html
[quick-start]: https://docs.deepmodeling.com/projects/deepmd/en/latest/getting-started/quick_start.html
[releases]: https://github.com/deepmodeling/deepmd-kit/releases
[testing]: https://docs.deepmodeling.com/projects/deepmd/en/latest/test/test.html
[tutorials]: https://tutorials.deepmodeling.com/
