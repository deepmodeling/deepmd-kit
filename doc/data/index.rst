Data
====
In this section, we will introduce how to convert the DFT labeled data into the data format used by DeePMD-kit.

The DeePMD-kit organize data in :code:`systems`. Each :code:`system` is composed by a number of :code:`frames`. One may roughly view a :code:`frame` as a snap short on an MD trajectory, but it does not necessary come from an MD simulation. A :code:`frame` records the coordinates and types of atoms, cell vectors if the periodic boundary condition is assumed, energy, atomic forces and virial. It is noted that the :code:`frames` in one :code:`system` share the same number of atoms with the same type.

.. toctree::
   :maxdepth: 1

   data-conv
   dpdata
