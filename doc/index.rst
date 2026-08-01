===========
DeePMD-kit
===========

.. rst-class:: lead

   From first-principles data to scalable molecular dynamics—through one open
   framework.

DeePMD-kit turns quantum-mechanical reference data into fast, scalable
interatomic potentials. It combines modern Deep Potential architectures,
multiple machine-learning backends, adaptation workflows, and
simulation-ready deployment in one open-source toolkit.

Use it across molecular and materials science—from finite molecules and
covalent systems to periodic solids and metals—and scale from laptop
experiments to distributed training and MPI-parallel molecular dynamics.

Choose your path
================

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: 🚀 Install and start
      :link: getting-started/index
      :link-type: doc
      :shadow: md

      Install DeePMD-kit, prepare a small dataset, and train your first model.

   .. grid-item-card:: 🧠 Choose a model
      :link: model/index
      :link-type: doc
      :shadow: md

      Compare DeepPot-SE, DPA-1, DPA-2, DPA-3, DPA-4/SeZM, and specialized
      physics models.

   .. grid-item-card:: 🧬 Train and adapt
      :link: train/index
      :link-type: doc
      :shadow: md

      Run single-task or multi-task training, fine-tuning, LoRA, and
      pretrained-model workflows.

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

   .. grid-item-card:: Modern potential architectures
      :shadow: sm

      Use efficient DeepPot-SE descriptors, attention-based DPA models, large
      atomic models, and SO(3)-equivariant DPA-4/SeZM.

   .. grid-item-card:: Broad physical targets
      :shadow: sm

      Learn energies, forces, virials, Hessians, spin, dipoles,
      polarizabilities, density of states, atomic populations, and custom
      properties.

   .. grid-item-card:: Foundation-model workflows
      :shadow: sm

      Download built-in DPA models, share representations across tasks,
      fine-tune full models or LoRA adapters, and use DPA-ADAPT for downstream
      prediction.

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

   On supported descriptors and workloads, :doc:`model compression
   <freeze/compress>` can deliver more than **10× inference speedup** and
   reduce memory usage by as much as **20×**. Actual gains depend on the model,
   system, and hardware.

From data to dynamics
=====================

.. grid:: 1 2 3 5
   :gutter: 2

   .. grid-item-card:: 1 · Prepare
      :link: data/index
      :link-type: doc

      Convert reference structures and labels into DeePMD data.

   .. grid-item-card:: 2 · Model
      :link: model/index
      :link-type: doc

      Select a descriptor, physical target, and backend.

   .. grid-item-card:: 3 · Train
      :link: train/index
      :link-type: doc

      Train from scratch or adapt a pretrained model.

   .. grid-item-card:: 4 · Validate
      :link: test/index
      :link-type: doc

      Test accuracy, inspect deviation, freeze, and compress.

   .. grid-item-card:: 5 · Simulate
      :link: inference/index
      :link-type: doc

      Run inference directly or deploy into molecular dynamics.

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
   * - :doc:`DPA-4 / SeZM <model/dpa4>`
     - SO(3)-equivariant learning, LoRA fine-tuning, optional ZBL bridging,
       spin support, and compiled ``.pt2`` deployment.

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

   .. grid-item-card:: DPA-4 / SeZM
      :link: model/dpa4
      :link-type: doc
      :shadow: sm

      Equivariant message passing, LoRA, ZBL, spin, compiled inference, and
      LAMMPS deployment.

   .. grid-item-card:: Pretrained DPA models
      :link: model/pretrained
      :link-type: doc
      :shadow: sm

      Resolve built-in model names directly or download checkpoints to a local
      cache.

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

.. important::

   DeePMD-kit is licensed under the :doc:`GNU LGPL-3.0-or-later <license>`.
   If you use DeePMD-kit in published work, follow the
   :doc:`citation guide <credits>` for the software version and methods used.

.. _getting-started:

.. toctree::
   :maxdepth: 3
   :caption: Getting Started
   :hidden:

   getting-started/index

.. _advanced:

.. toctree::
   :maxdepth: 3
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

.. _feedback:
.. _affiliated packages:
