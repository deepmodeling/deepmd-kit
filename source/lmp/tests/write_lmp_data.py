# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

comment_lmp_data = "# the first line must be comment"


def write_lmp_data(box, coord, type_list, file_name) -> None:
    natom = coord.shape[0]
    ntype = np.unique(type_list).shape[0]
    with open(file_name, "w") as f:
        f.write(comment_lmp_data + "\n")
        f.write("%d atoms\n" % (natom))
        f.write("%d atom types\n" % (ntype))
        f.write(f"{box[0]:.10e} {box[1]:.10e} xlo xhi\n")
        f.write(f"{box[2]:.10e} {box[3]:.10e} ylo yhi\n")
        f.write(f"{box[4]:.10e} {box[5]:.10e} zlo zhi\n")
        f.write(f"{box[6]:.10e} {box[7]:.10e} {box[8]:.10e} xy xz yz\n\nAtoms\n\n")
        for i in range(natom):
            f.write(
                "%d %d %.10e %.10e %.10e\n"
                % (i + 1, type_list[i], coord[i][0], coord[i][1], coord[i][2])
            )
        f.write("\n")


def write_lmp_data_full(
    box, coord, mol_list, type_list, charge, file_name, bond_list, mass_list
) -> None:
    natom = coord.shape[0]
    ntype = np.unique(type_list).shape[0]
    nbond_type = len(bond_list)
    nbond_list = np.zeros(nbond_type, dtype="int")
    for i in range(nbond_type):
        nbond_list[i] = len(bond_list[i])
    with open(file_name, "w") as f:
        f.write(comment_lmp_data + "\n")
        f.write("%d atoms\n" % (natom))
        f.write("%d atom types\n" % (ntype))
        f.write("%d bonds\n" % (nbond_list.sum()))
        f.write("%d bond types\n" % (nbond_type))
        f.write(f"{box[0]:.10e} {box[1]:.10e} xlo xhi\n")
        f.write(f"{box[2]:.10e} {box[3]:.10e} ylo yhi\n")
        f.write(f"{box[4]:.10e} {box[5]:.10e} zlo zhi\n")
        f.write(f"{box[6]:.10e} {box[7]:.10e} {box[8]:.10e} xy xz yz\n")
        f.write("\nMasses\n\n")
        for i in range(3):
            f.write(f"{i+1:d} {mass_list[i]:.10e}\n")
        f.write("\nAtoms\n\n")
        for i in range(natom):
            f.write(
                "%d %d %d %.10e %.10e %.10e %.10e\n"
                % (
                    i + 1,
                    mol_list[i],
                    type_list[i],
                    charge[i],
                    coord[i][0],
                    coord[i][1],
                    coord[i][2],
                )
            )
        f.write("\nBonds\n\n")
        bond_count = 0
        for i in range(nbond_type):
            for j in range(nbond_list[i]):
                bond_count += 1
                f.write(
                    "%d %d %d %d\n"
                    % (bond_count, i + 1, bond_list[i][j][0], bond_list[i][j][1])
                )
        f.write("\n")


def write_lmp_data_spin(box, coord, spin, type_list, file_name) -> None:
    natom = coord.shape[0]
    ntype = np.unique(type_list).shape[0]
    sp_norm = np.linalg.norm(spin, axis=1, keepdims=True)
    sp_unit = spin / np.where(sp_norm == 0, 1, sp_norm)
    sp_unit = np.where(sp_norm == 0, 1, sp_unit)
    with open(file_name, "w") as f:
        f.write(comment_lmp_data + "\n")
        f.write("%d atoms\n" % (natom))
        f.write("%d atom types\n" % (ntype))
        f.write(f"{box[0]:.10e} {box[1]:.10e} xlo xhi\n")
        f.write(f"{box[2]:.10e} {box[3]:.10e} ylo yhi\n")
        f.write(f"{box[4]:.10e} {box[5]:.10e} zlo zhi\n")
        f.write(f"{box[6]:.10e} {box[7]:.10e} {box[8]:.10e} xy xz yz\n\nAtoms\n\n")
        for i in range(natom):
            f.write(
                "%d %d %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n"
                % (
                    i + 1,
                    type_list[i],
                    coord[i][0],
                    coord[i][1],
                    coord[i][2],
                    sp_unit[i][0],
                    sp_unit[i][1],
                    sp_unit[i][2],
                    sp_norm[i][0],
                )
            )
        f.write("\n")
