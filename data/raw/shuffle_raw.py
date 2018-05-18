#!/usr/bin/env python3

import os
import numpy as np
import argparse

def _parse_args () : 
    parser = argparse.ArgumentParser (
        description = "parse shuffle args" )
    parser.add_argument ("INPUT", default = ".", 
                         help = "input dir of raw files")
    parser.add_argument ("OUTPUT", default = ".",
                         help = "output dir of shuffled raw files")
    parser.add_argument ('-r', '--raws', nargs = '+', default = [],
                         help = "raw files, if not set, then auto detect")
    return parser.parse_args()

def detect_raw (path) :
    raws = []
    names = ["box.raw", "coord.raw", "energy.raw", "force.raw", "virial.raw"]
    for ff in names :
        if os.path.isfile (path + "/" + ff) : raws.append (ff)
    return raws

def _main () :
    args = _parse_args ()
    raws = args.raws
    inpath = args.INPUT
    outpath = args.OUTPUT

    if not os.path.isdir (inpath):
        print ("# no input dir " + inpath + ", exit")
        return
        
    if not os.path.isdir (outpath) : 
        os.mkdir (outpath)

    if len(raws) == 0 :
        raws = detect_raw (inpath)

    if len(raws) == 0 :
        print ("# no file to shuffle, exit")
        return

    assert ("box.raw" in raws)
    tmp = np.loadtxt(os.path.join(inpath, "box.raw"))
    tmp = np.reshape(tmp, [-1, 9])
    nframe = tmp.shape[0]
    print(nframe)

    print ("# will shuffle raw files " + str(raws) + 
           " in dir " + inpath +
           " and output to dir " + outpath)

    tmp = np.loadtxt (inpath + "/" + raws[0])
    tmp = np.reshape(tmp, [nframe, -1])
    nframe = tmp.shape[0]
    idx = np.arange (nframe)
    np.random.shuffle(idx)
    
    for ii in raws : 
        data = np.loadtxt(inpath + "/" + ii)
        data = np.reshape(data, [nframe, -1])
        data = data [idx]
        np.savetxt (outpath + "/" + ii, data)

if __name__ == "__main__" :
    _main()
