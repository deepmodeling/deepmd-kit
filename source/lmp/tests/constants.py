# SPDX-License-Identifier: LGPL-3.0-or-later
# https://github.com/lammps/lammps/blob/1e1311cf401c5fc2614b5d6d0ff3230642b76597/src/update.cpp#L193
nktv2p = 1.6021765e6
nktv2p_real = 68568.415
metal2real = 23.060549

dist_metal2real = 1.0
ener_metal2real = 23.060549
force_metal2real = ener_metal2real / dist_metal2real
mass_metal2real = 1.0
charge_metal2real = 1.0

dist_metal2si = 1.0e-10
ener_metal2si = 1.3806504e-23 / 8.617343e-5
force_metal2si = ener_metal2si / dist_metal2si
mass_metal2si = 1e-3 / 6.02214e23
charge_metal2si = 1.6021765e-19
