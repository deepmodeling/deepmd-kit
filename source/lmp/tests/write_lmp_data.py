import numpy as np

comment_lmp_data = "# the first line must be comment"


def write_lmp_data(box, coord, type_list, file_name):
    natom = coord.shape[0]
    ntype = np.unique(type_list).shape[0]
    with open(file_name, "w") as f:
        f.write(comment_lmp_data + "\n")
        f.write("%d atoms\n" % (natom))
        f.write("%d atom types\n" % (ntype))
        f.write(f"{box[0]:.1f} {box[1]:.1f} xlo xhi\n")
        f.write(f"{box[2]:.1f} {box[3]:.1f} ylo yhi\n")
        f.write(f"{box[4]:.1f} {box[5]:.1f} zlo zhi\n")
        f.write(f"{box[6]:.1f} {box[7]:.1f} {box[8]:.1f} xy xz yz\n\nAtoms\n\n")
        for i in range(natom):
            f.write(
                "%d %d %.2f %.2f %.2f\n"
                % (i + 1, type_list[i], coord[i][0], coord[i][1], coord[i][2])
            )
        f.write("\n")

def write_lmp_data_full(box, coord, mol_list, type_list, charge, file_name, bond_list, mass_list):
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
        f.write(f"{box[0]:.1f} {box[1]:.1f} xlo xhi\n")
        f.write(f"{box[2]:.1f} {box[3]:.1f} ylo yhi\n")
        f.write(f"{box[4]:.1f} {box[5]:.1f} zlo zhi\n")
        f.write(f"{box[6]:.1f} {box[7]:.1f} {box[8]:.1f} xy xz yz\n")
        f.write("\nMasses\n\n")
        for i in range(3):
            f.write(f"{i+1:d} {mass_list[i]:.6f}\n")
        f.write("\nAtoms\n\n")
        for i in range(natom):
            f.write(
                "%d %d %d %d %.2f %.2f %.2f\n"
                % (i + 1, mol_list[i], type_list[i], charge[i], coord[i][0], coord[i][1], coord[i][2])
            )
        f.write("\nBonds\n\n")
        bond_count = 0
        for i in range(nbond_type):
            for j in range(nbond_list[i]):
                bond_count += 1
                f.write("%d %d %d %d\n"%(bond_count, i+1, bond_list[i][j][0], bond_list[i][j][1]))
        f.write("\n")
