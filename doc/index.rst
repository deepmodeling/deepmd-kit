.. deepmd-kit documentation master file, created by
   sphinx-quickstart on Sat Nov 21 18:36:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========================
DeePMD-kit's documentation
==========================

DeePMD-kit is a package written in Python/C++, designed to minimize the effort required to build deep learning based model of interatomic potential energy and force field and to perform molecular dynamics (MD). This brings new hopes to addressing the accuracy-versus-efficiency dilemma in molecular simulations. Applications of DeePMD-kit span from finite molecules to extended systems and from metallic systems to chemically bonded systems.

.. Important:: The project DeePMD-kit is licensed under `GNU LGPLv3.0 <https://github.com/deepmodeling/deepmd-kit/blob/master/LICENSE>`_. If you use this code in any future publications, please cite this using *Han Wang, Linfeng Zhang, Jiequn Han, and Weinan E. "DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics." Computer Physics Communications 228 (2018): 178-184.*

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

   install/index
   data/index
   model/index
   train/index
   freeze/index
   test/index
   inference/index
   third-party/index
   troubleshooting/index

.. _developer-guide:

.. toctree::
   :maxdepth: 5
   :caption: Developer Guide
   :glob:

   development/*
   api_py/api_py
   API_CC/api_cc


.. _project-details:

.. toctree::
   :maxdepth: 3
   :caption: Project Details

   license
   credits


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _feedback: 
.. _affiliated packages: 
