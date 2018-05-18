#!/usr/bin/env python3

import numpy as np
import argparse, os
import os.path

def copy (in_dir,
          out_dir,
          ncopies = [1,1,1]) :
    has_energy = os.path.isfile (in_dir + "/energy.raw")
    has_force  = os.path.isfile (in_dir + "/force.raw")
    has_virial = os.path.isfile (in_dir + "/virial.raw")

    i_box       = np.loadtxt (in_dir + "/box.raw")
    i_coord     = np.loadtxt (in_dir + "/coord.raw")
    if has_energy :
        i_energy    = np.loadtxt (in_dir + "/energy.raw")
    if has_force : 
        i_force     = np.loadtxt (in_dir + "/force.raw")
    if has_virial: 
        i_virial    = np.loadtxt (in_dir + "/virial.raw")
    i_type      = np.loadtxt (in_dir + "/type.raw")

    nsys = ncopies[0] * ncopies[1] * ncopies[2]
    nframes = i_coord.shape[0]
    natoms = i_coord.shape[1] // 3

    if has_energy :
        o_energy = i_energy * nsys
    if has_virial :
        o_virial = i_virial * nsys    

    o_box = np.zeros(i_box.shape)
    for ii in range (3) :
        o_box[:, ii*3:ii*3+3] = i_box[:, ii*3:ii*3+3] * ncopies[ii]
        
    o_coord = i_coord
    if has_force :
        o_force = i_force
    i_type = np.reshape (i_type, [-1, natoms])
    o_type = i_type
    for ii in range (ncopies[0]) :
        for jj in range (ncopies[1]) :
            for kk in range (ncopies[2]) :
                if ii == 0 and jj == 0 and kk == 0 : 
                    continue
                citer = np.array ([ii, jj, kk])
                shift = np.zeros ([nframes, 3])
                for dd in range (3) :
                    shift += i_box[:, dd*3:dd*3+3] * citer[dd]
                ashift = np.tile (shift, natoms)
                o_coord = np.concatenate ((o_coord, i_coord + ashift), axis = 1)
                if has_force :
                    o_force = np.concatenate ((o_force, i_force), axis = 1)
                o_type = np.concatenate ((o_type, i_type), axis = 1)

    if not os.path.exists (out_dir) : 
        os.makedirs (out_dir)
        
    np.savetxt (out_dir + "/box.raw",           o_box)
    np.savetxt (out_dir + "/coord.raw",         o_coord)
    if has_energy :
        np.savetxt (out_dir + "/energy.raw",        o_energy)
    if has_force :
        np.savetxt (out_dir + "/force.raw",         o_force)
    if has_virial :
        np.savetxt (out_dir + "/virial.raw",        o_virial)
    np.savetxt (out_dir + "/type.raw",          o_type, fmt = '%d')
    np.savetxt (out_dir + "/ncopies.raw",       ncopies, fmt = "%d")
    
def _main () :
    parser = argparse.ArgumentParser (
        description = "parse copy raw args" )
    parser.add_argument ("INPUT", default = ".", 
                         help = "input dir of raw files")
    parser.add_argument ("OUTPUT", default = ".",
                         help = "output dir of copied raw files")
    parser.add_argument ("-n", "--ncopies", nargs = 3, default = [1,1,1], type = int,
                         help = "the number of copies")
    args = parser.parse_args()

    print ("# copy the system by %s copies" % args.ncopies)
    assert (np.all(np.array(args.ncopies, dtype = int) >= np.array([1,1,1], dtype=int))), \
        "number of copies should be larger than or equal to 1"
    copy (args.INPUT, args.OUTPUT, args.ncopies)

if __name__ == "__main__" :
    _main()
