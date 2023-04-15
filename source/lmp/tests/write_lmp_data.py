import numpy as np
comment_lmp_data = "# the first line must be comment"

def write_lmp_data(box, coord, type_list, file_name):
    natom = coord.shape[0]
    ntype = np.unique(type_list).shape[0]
    with open(file_name, "w") as f:
        f.write(comment_lmp_data+"\n")
        f.write("%d atoms\n"%(natom))
        f.write("%d atom types\n"%(ntype))
        f.write("%.1f %.1f xlo xhi\n"%(box[0], box[1]))
        f.write("%.1f %.1f ylo yhi\n"%(box[2], box[3]))
        f.write("%.1f %.1f zlo zhi\n"%(box[4], box[5]))
        f.write("%.1f %.1f %.1f xy xz yz\n\nAtoms\n\n"%(box[6], box[7], box[8]))
        for i in range(natom):
            f.write("%d %d %.2f %.2f %.2f\n"%(i+1, type_list[i], coord[i][0], coord[i][1], coord[i][2]))
        f.write("\n")