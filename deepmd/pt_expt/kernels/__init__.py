# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hand-written operator packages for graph-lower inference.

The kernels themselves live under ``source/op/pt`` and compile into
``libdeepmd_op_pt.so``. The modules here bind the resulting
``torch.ops.deepmd.*`` operators to the pt_expt graph lower: the schema
front end, the backward and meta (fake) implementations ``torch.export`` and
``make_fx`` require, the immutable compression artifacts, and the eligibility
predicates that decide whether an operator may serve a given model.

None of that binding layer is device specific -- the device lives in the
compiled kernel behind the dispatcher -- so an operator implemented for more
than one device is bound once, here:

:mod:`.dpa4c`
    DPA4C compressed descriptor: radial spline lookup, one packed moment
    reduction carrying both envelope masses, the invariant readout, and the
    analytical edge-vector backward. Includes the compact canonical lower
    that the LAMMPS deployment ABI consumes.
:mod:`.graph_fitting`
    Descriptor-agnostic fused energy fitting network on the flat node axis.
:mod:`.edge_force_virial`
    Descriptor-agnostic force / atom-virial / per-frame-virial assembly from
    the per-edge energy gradient.

Operators that exist for one device only stay in that device's package:
:mod:`.cuda` (DPA1 and DPA4), :mod:`.triton`, :mod:`.cute`, :mod:`.cutile`.

:mod:`.utils` resolves which of them may run.
"""
