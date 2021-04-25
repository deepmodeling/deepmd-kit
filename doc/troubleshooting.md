# Troubleshooting
In consequence of various differences of computers or systems, problems may occur. Some common circumstances are listed as follows. 
If other unexpected problems occur, you're welcome to contact us for help.

## Model compatability

When the version of DeePMD-kit used to training model is different from the that of DeePMD-kit running MDs, one has the problem of model compatability.

DeePMD-kit guarantees that the codes with the same major and minor revisions are compatible. That is to say v0.12.5 is compatible to v0.12.0, but is not compatible to v0.11.0 nor v1.0.0. 

## Installation: inadequate versions of gcc/g++
Sometimes you may use a gcc/g++ of version <4.9. If you have a gcc/g++ of version > 4.9, say, 7.2.0, you may choose to use it by doing 
```bash
export CC=/path/to/gcc-7.2.0/bin/gcc
export CXX=/path/to/gcc-7.2.0/bin/g++
```

If, for any reason, for example, you only have a gcc/g++ of version 4.8.5, you can still compile all the parts of TensorFlow and most of the parts of DeePMD-kit. i-Pi will be disabled automatically.

## Installation: build files left in DeePMD-kit
When you try to build a second time when installing DeePMD-kit, files produced before may contribute to failure. Thus, you may clear them by
```bash
cd build
rm -r *
```
and redo the `cmake` process.

## MD: cannot run LAMMPS after installing a new version of DeePMD-kit
This typically happens when you install a new version of DeePMD-kit and copy directly the generated `USER-DEEPMD` to a LAMMPS source code folder and re-install LAMMPS.

To solve this problem, it suffices to first remove `USER-DEEPMD` from LAMMPS source code by 
```bash
make no-user-deepmd
```
and then install the new `USER-DEEPMD`.

If this does not solve your problem, try to decompress the LAMMPS source tarball and install LAMMPS from scratch again, which typically should be very fast.
