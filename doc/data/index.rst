Data
====
In this section, we will introduce how to convert the DFT-labeled data into the data format used by DeePMD-kit.

The DeePMD-kit organizes data in :code:`systems`. Each :code:`system` is composed of a number of :code:`frames`. One may roughly view a :code:`frame` as a snapshot of an MD trajectory, but it does not necessarily come from an MD simulation. A :code:`frame` records the coordinates and types of atoms, cell vectors if the periodic boundary condition is assumed, energy, atomic forces and virials. It is noted that the :code:`frames` in one :code:`system` share the same number of atoms with the same type.

.. toctree::
   :maxdepth: 1

   system
   data-conv
   dpdata
