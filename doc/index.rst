===========
DeePMD-kit
===========

.. rst-class:: lead

   Start from a pretrained Deep Potential model, fine-tune it for your system,
   and deploy it at simulation scale.

.. important::

   **A pretrained model can be your starting point, not just your end result.**
   Download a built-in DPA checkpoint, fine-tune the full model, or use
   `DPA-4 LoRA`_ with PyTorch single-task training, then test, export, and deploy
   it through the same DeePMD-kit workflow.

DeePMD-kit turns quantum-mechanical reference data into fast, scalable
interatomic potentials. Use it across molecular and materials science—from
finite molecules and covalent systems to periodic solids and metals—and scale
from laptop fine-tuning to distributed training and MPI-parallel molecular
dynamics.

.. figure:: _static/dpa4-cps-throughput.webp
   :alt: DPA4 model family Pareto frontier for Matbench Discovery CPS and saturated inference throughput
   :width: 100%
   :align: center

   The DPA4 model family traces a Pareto frontier across Matbench Discovery CPS
   and saturated inference throughput.

Choose your path
================

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: 🧬 Start from pretrained
      :link: model/pretrained
      :link-type: doc
      :shadow: md

      Download a built-in DPA checkpoint or resolve its model name directly.

   .. grid-item-card:: 🎯 Fine-tune to your data
      :link: train/finetuning
      :link-type: doc
      :shadow: md

      Adapt a full pretrained model, or use DPA-4 LoRA with PyTorch single-task
      training.

   .. grid-item-card:: 🏗️ Train from scratch
      :link: train/index
      :link-type: doc
      :shadow: md

      Build a new potential with single-task, multi-task, or distributed
      training.

   .. grid-item-card:: 🧠 Choose a model
      :link: model/index
      :link-type: doc
      :shadow: md

      Compare DeepPot-SE, DPA-1, DPA-2, DPA-3, DPA-4, and specialized
      physics models.

   .. grid-item-card:: 🔌 Deploy and integrate
      :link: third-party/index
      :link-type: doc
      :shadow: md

      Move models into Python, native APIs, LAMMPS, i-PI, ASE, GROMACS, and
      the wider simulation ecosystem.

Why DeePMD-kit
==============

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Pretrained-first adaptation
      :shadow: sm

      Start from built-in DPA checkpoints, fine-tune full models, use
      `DPA-4 LoRA`_ with PyTorch single-task training, and reuse learned
      representations with DPA-ADAPT.

   .. grid-item-card:: Training from scratch
      :link: train/index
      :link-type: doc
      :shadow: sm

      Configure new architectures and physical targets, then train with
      single-task, multi-task, or distributed workflows.

   .. grid-item-card:: Modern potential architectures
      :shadow: sm

      Use efficient DeepPot-SE descriptors, attention-based DPA models, large
      atomic models, and SO(3)-equivariant DPA-4.

   .. grid-item-card:: Broad physical targets
      :shadow: sm

      Learn energies, forces, virials, Hessians, spin, dipoles,
      polarizabilities, density of states, atomic populations, and custom
      properties.

   .. grid-item-card:: Multi-backend framework
      :link: backend
      :link-type: doc
      :shadow: sm

      Work with TensorFlow, PyTorch, JAX, or Paddle and use backend-aware model
      formats, conversion, and runtime plugins.

   .. grid-item-card:: Performance at scale
      :shadow: sm

      Run on CPUs and GPUs, distribute training, compress supported models,
      export compiled ``.pt2`` artifacts, and drive MPI-parallel simulations.

   .. grid-item-card:: Open scientific ecosystem
      :link: third-party/index
      :link-type: doc
      :shadow: sm

      Connect to simulation engines, workflow tools, native applications, and
      external GNN models through documented interfaces and plugins.

.. tip::

   On supported descriptors and workloads,
   :doc:`model compression <freeze/compress>` can deliver more than
   **10× inference speedup** and reduce memory usage by as much as **20×**.
   Actual gains depend on the model, system, and hardware.

Two starting points, one path to dynamics
=========================================

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: 1 · Pretrained
      :link: model/pretrained
      :link-type: doc

      Select a built-in DPA model or a compatible checkpoint.

   .. grid-item-card:: 2 · Prepare
      :link: data/index
      :link-type: doc

      Convert target structures and labels into DeePMD data.

   .. grid-item-card:: 3 · Fine-tune
      :link: train/finetuning
      :link-type: doc

      Adapt the full model, or use DPA-4 LoRA with PyTorch single-task training.

   .. grid-item-card:: 3 · Train
      :link: train/index
      :link-type: doc

      Build a new model from scratch when adaptation is not the right fit.

   .. grid-item-card:: 4 · Validate
      :link: test/index
      :link-type: doc

      Test accuracy, inspect deviation, freeze, and compress.

   .. grid-item-card:: 5 · Simulate
      :link: inference/index
      :link-type: doc

      Run inference directly or deploy into molecular dynamics.

Fine-tuning and from-scratch training converge on the same validation, export,
and deployment toolchain. `DPA-4 LoRA`_ is currently limited to PyTorch
single-task fine-tuning.

Choose a model family
=====================

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Family
     - A strong starting point when you need
   * - :doc:`DeepPot-SE <model/train-se-e2-a>`
     - An efficient, established baseline with broad backend and deployment
       support.
   * - :doc:`DPA-1 <model/train-se-atten>`
     - Attention-based local representations and type embedding.
   * - :doc:`DPA-2 <model/dpa2>`
     - Multi-task pretraining, shared representations, and smooth conservative
       potentials.
   * - :doc:`DPA-3 <model/dpa3>`
     - Message passing over line-graph representations and broad chemical
       coverage.
   * - :doc:`DPA-4 <model/dpa4>`
     - SO(3)-equivariant learning, `DPA-4 LoRA`_ for PyTorch single-task
       fine-tuning, optional ZBL bridging, spin support, and compiled ``.pt2``
       deployment.

.. figure:: _static/dpa4-performance.webp
   :alt: DPA4 energy and force accuracy versus saturated throughput
   :width: 100%
   :align: center

   DPA4 provides a family of accuracy–throughput trade-offs for different
   deployment budgets.

More than conventional force fields
===================================

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: 🧲 Spin and magnetism
      :link: model/train-energy-spin
      :link-type: doc

      Train spin-aware potentials with atomic and magnetic force targets.

   .. grid-item-card:: ⚛️ Long- and short-range physics
      :link: model/dplr
      :link-type: doc

      Combine learned local interactions with DPLR electrostatics, DPRc range
      correction, pair tables, or analytical ZBL bridging.

   .. grid-item-card:: 📊 Properties and embeddings
      :link: inference/embedding
      :link-type: doc

      Predict electronic or structural properties and export learned
      representations for analysis or downstream models.

New and noteworthy
==================

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Pretrained DPA models
      :link: model/pretrained
      :link-type: doc
      :shadow: sm

      Resolve built-in model names directly or download checkpoints to a local
      cache.

   .. grid-item-card:: DPA-4
      :link: model/dpa4
      :link-type: doc
      :shadow: sm

      Equivariant message passing, PyTorch single-task LoRA, ZBL, spin,
      compiled inference, and LAMMPS deployment.

   .. grid-item-card:: DPA-ADAPT
      :link: dpa_adapt/index
      :link-type: doc
      :shadow: sm

      Adapt pretrained DPA representations to downstream atomistic property
      tasks.

   .. grid-item-card:: Official Agent Skills
      :link: agent-skills
      :link-type: doc
      :shadow: sm

      Give AI agents reproducible guidance for training, fine-tuning,
      inference, and LAMMPS workflows.

Documentation map
=================

* **Getting Started:** :doc:`Installation and first model
  <getting-started/index>`.
* **User Guide:** :doc:`Backends <backend>` · :doc:`Installation
  <install/index>` · :doc:`Data <data/index>` · :doc:`Models <model/index>` ·
  :doc:`Training <train/index>` · :doc:`Freeze <freeze/index>` · :doc:`Test
  <test/index>` · :doc:`Inference <inference/index>` · :doc:`DPA-ADAPT
  <dpa_adapt/index>` · :doc:`CLI <cli>` · :doc:`Integrations
  <third-party/index>` · :doc:`Agent Skills <agent-skills>` · :doc:`NVNMD
  <nvnmd/index>` · :doc:`Environment <env>` · :doc:`Troubleshooting
  <troubleshooting/index>`.
* **Tutorials and Publications:** `Tutorials
  <https://tutorials.deepmodeling.com/>`_ · `Publications
  <https://blogs.deepmodeling.com/papers/deepmd-kit/>`_.
* **Developer Guide:** :doc:`CMake <development/cmake>` · :doc:`TensorFlow
  models <development/create-a-model-tf>` · :doc:`PyTorch models
  <development/create-a-model-pt>` · :doc:`Type embedding
  <development/type-embedding>` · :doc:`Coding conventions
  <development/coding-conventions>` · :doc:`CI/CD <development/cicd>` ·
  :doc:`Python API <autoapi/deepmd/index>` · :doc:`Custom operators <api_op>` ·
  :doc:`C++ API <API_CC/api_cc>` · :doc:`C API <api_c/api_c>` · :doc:`Core API
  <api_core/api_core>`.
* **Project Details:** :doc:`License <license>` · :doc:`Authors and credits
  <credits>` · :doc:`Logo <logo>`.

Citation
========

If you use DeePMD-kit in published work, cite the general software publication
that matches the version used:

* Han Wang, Linfeng Zhang, Jiequn Han, and Weinan E. "DeePMD-kit: A deep
  learning package for many-body potential energy representation and molecular
  dynamics." *Computer Physics Communications* 228 (2018): 178–184.
  `DOI: 10.1016/j.cpc.2018.03.016
  <https://doi.org/10.1016/j.cpc.2018.03.016>`_.
* Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen,
  Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye,
  Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao,
  Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan,
  Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu,
  Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar
  Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu,
  Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, and Han
  Wang. "DeePMD-kit v2: A software package for Deep Potential models."
  *The Journal of Chemical Physics* 159 (2023): 054801.
  `DOI: 10.1063/5.0155600 <https://doi.org/10.1063/5.0155600>`_.
* Jinzhe Zeng, Duo Zhang, Anyang Peng, Xiangyu Zhang, Sensen He, Yan Wang,
  Xinzijian Liu, Hangrui Bi, Yifan Li, Chun Cai, Chengqian Zhang, Yiming Du,
  Jia-Xin Zhu, Pinghui Mo, Zhengtao Huang, Qiyu Zeng, Shaochen Shi, Xuejian
  Qin, Zhaoxi Yu, Chenxing Luo, Ye Ding, Yun-Pei Liu, Ruosong Shi, Zhenyu Wang,
  Sigbjørn Løland Bore, Junhan Chang, Zhe Deng, Zhaohan Ding, Siyuan Han,
  Wanrun Jiang, Guolin Ke, Zhaoqing Liu, Denghui Lu, Koki Muraoka, Hananeh
  Oliaei, Anurag Kumar Singh, Haohui Que, Weihong Xu, Zhangmancang Xu,
  Yong-Bin Zhuang, Jiayu Dai, Timothy J. Giese, Weile Jia, Ben Xu, Darrin M.
  York, Linfeng Zhang, and Han Wang. "DeePMD-kit v3: A Multiple-Backend
  Framework for Machine Learning Potentials." *Journal of Chemical Theory and
  Computation* 21 (2025): 4375–4385.
  `DOI: 10.1021/acs.jctc.5c00340
  <https://doi.org/10.1021/acs.jctc.5c00340>`_.

Follow the :doc:`citation guide <credits>` for the method-specific publications
required by the models and features used in your work.

.. note::

   DeePMD-kit is licensed under the :doc:`GNU LGPL-3.0-or-later <license>`.

.. _getting-started:

.. toctree::
   :maxdepth: 3
   :caption: Getting Started
   :hidden:

   getting-started/index

.. _advanced:

.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: User Guide
   :hidden:

   backend
   install/index
   data/index
   model/index
   train/index
   freeze/index
   test/index
   inference/index
   dpa_adapt/index
   cli
   third-party/index
   agent-skills
   nvnmd/index
   env
   troubleshooting/index

.. _tutorial:

.. toctree::
   :maxdepth: 2
   :caption: Tutorials and Publications
   :hidden:

   Tutorials <https://tutorials.deepmodeling.com/>
   Publications <https://blogs.deepmodeling.com/papers/deepmd-kit/>

.. _developer-guide:

.. toctree::
   :maxdepth: 5
   :caption: Developer Guide
   :hidden:

   development/cmake
   development/create-a-model-tf
   development/create-a-model-pt
   development/type-embedding
   development/coding-conventions
   development/cicd
   Python API <autoapi/deepmd/index>
   api_op
   API_CC/api_cc
   api_c/api_c
   api_core/api_core

.. _project-details:

.. toctree::
   :maxdepth: 3
   :caption: Project Details
   :hidden:

   license
   credits
   logo

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _DPA-4 LoRA: https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html#lora-fine-tuning

.. _feedback:
.. _affiliated packages:
