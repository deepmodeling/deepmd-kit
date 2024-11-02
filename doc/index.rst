.. deepmd-kit documentation master file, created by
   sphinx-quickstart on Sat Nov 21 18:36:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========================
DeePMD-kit's documentation
==========================

DeePMD-kit is a package written in Python/C++, designed to minimize the effort required to build deep learning-based models of interatomic potential energy and force field and to perform molecular dynamics (MD). This brings new hopes to addressing the accuracy-versus-efficiency dilemma in molecular simulations. Applications of DeePMD-kit span from finite molecules to extended systems and from metallic systems to chemically bonded systems.

.. Important::

   The project DeePMD-kit is licensed under `GNU LGPLv3.0 <https://github.com/deepmodeling/deepmd-kit/blob/master/LICENSE>`_.
   If you use this code in any future publications, please cite the following publications for general purpose:

      - Han Wang, Linfeng Zhang, Jiequn Han, and Weinan E. "DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics." Computer Physics Communications 228 (2018): 178-184.
      - Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang. "DeePMD-kit v2: A software package for Deep Potential models." J. Chem. Phys., 159, 054801 (2023).

   In addition, please follow :ref:`this page <cite>` to cite the methods you used.

.. _getting-started:

.. toctree::
   :maxdepth: 3
   :caption: Getting Started

   getting-started/index

.. _advanced:

.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: Advanced

   backend
   install/index
   data/index
   model/index
   train/index
   freeze/index
   test/index
   inference/index
   cli
   third-party/index
   nvnmd/index
   env
   troubleshooting/index


.. _tutorial:

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   Tutorials <https://tutorials.deepmodeling.com/>
   Publications <https://blogs.deepmodeling.com/papers/deepmd-kit/>

.. _developer-guide:

.. toctree::
   :maxdepth: 5
   :caption: Developer Guide

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

   license
   credits
   logo


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _feedback:
.. _affiliated packages:
