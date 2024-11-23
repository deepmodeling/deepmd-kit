# Formats of a system

Two binary formats, NumPy and HDF5, are supported for training. The raw format is not directly supported, but a tool is provided to convert data from the raw format to the NumPy format.

## NumPy format

In a system with the NumPy format, the system properties are stored as text files ending with `.raw`, such as `type.raw` and `type_map.raw`, under the system directory. If one needs to train a non-periodic system, an empty `nopbc` file should be put under the system directory. Both input and labeled frame properties are saved as the [NumPy binary data (NPY) files](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#npy-format) ending with `.npy` in each of the `set.*` directories. Take an example, a system may contain the following files:

```
type.raw
type_map.raw
nopbc
set.000/coord.npy
set.000/energy.npy
set.000/force.npy
set.001/coord.npy
set.001/energy.npy
set.001/force.npy
```

We assume that the atom types do not change in all frames. It is provided by `type.raw`, which has one line with the types of atoms written one by one. The atom types should be integers. For example the `type.raw` of a system that has 2 atoms with 0 and 1:

```bash
$ cat type.raw
0 1
```

Sometimes one needs to map the integer types to atom names. The mapping can be given by the file `type_map.raw`. For example

```bash
$ cat type_map.raw
O H
```

The type `0` is named by `"O"` and the type `1` is named by `"H"`.

For training models with descriptor `se_atten`, a [new system format](../model/train-se-atten.md#data-format) is supported to put together the frame-sparse systems with the same atom number.

## HDF5 format

A system with the HDF5 format has the same structure as the NumPy format, but in an HDF5 file, a system is organized as an [HDF5 group](https://docs.h5py.org/en/stable/high/group.html). The file name of a NumPy file is the key in an HDF5 file, and the data is the value of the key. One needs to use `#` in a DP path to divide the path to the HDF5 file and the HDF5 path:

```
/path/to/data.hdf5#/H2O
```

Here, `/path/to/data.hdf5` is the file path and `/H2O` is the HDF5 path. All HDF5 paths should start with `/`. There should be some data in the `H2O` group, such as `/H2O/type.raw` and `/H2O/set.000/force.npy`.

An HDF5 file with a large number of systems has better performance than multiple NumPy files in a large cluster.

## Raw format and data conversion

A raw file is a plain text file with each information item written in one file and one frame written on one line. **It's not directly supported**, but we provide a tool to convert them.

In the raw format, the property of one frame is provided per line, ending with `.raw`. Take an example, the default files that provide box, coordinate, force, energy and virial are `box.raw`, `coord.raw`, `force.raw`, `energy.raw` and `virial.raw`, respectively. Here is an example of `force.raw`:

```bash
$ cat force.raw
-0.724  2.039 -0.951  0.841 -0.464  0.363
 6.737  1.554 -5.587 -2.803  0.062  2.222
-1.968 -0.163  1.020 -0.225 -0.789  0.343
```

This `force.raw` contains 3 frames with each frame having the forces of 2 atoms, thus it has 3 lines and 6 columns. Each line provides all the 3 force components of 2 atoms in 1 frame. The first three numbers are the 3 force components of the first atom, while the second three numbers are the 3 force components of the second atom. Other files are organized similarly. The number of lines of all raw files should be identical.

One can use the script `$deepmd_source_dir/data/raw/raw_to_set.sh` to convert the prepared raw files to the NumPy format. For example, if we have a raw file that contains 6000 frames,

```bash
$ ls
box.raw  coord.raw  energy.raw  force.raw  type.raw  virial.raw
$ $deepmd_source_dir/data/raw/raw_to_set.sh 2000
nframe is 6000
nline per set is 2000
will make 3 sets
making set 0 ...
making set 1 ...
making set 2 ...
$ ls
box.raw  coord.raw  energy.raw  force.raw  set.000  set.001  set.002  type.raw  virial.raw
```

It generates three sets `set.000`, `set.001` and `set.002`, with each set containing 2000 frames in the NumPy format.
