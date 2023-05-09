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
