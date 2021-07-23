# Installation
## Inadequate versions of gcc/g++
Sometimes you may use a gcc/g++ of version <4.9. If you have a gcc/g++ of version > 4.9, say, 7.2.0, you may choose to use it by doing 
```bash
export CC=/path/to/gcc-7.2.0/bin/gcc
export CXX=/path/to/gcc-7.2.0/bin/g++
```

If, for any reason, for example, you only have a gcc/g++ of version 4.8.5, you can still compile all the parts of TensorFlow and most of the parts of DeePMD-kit. i-Pi will be disabled automatically.

## Build files left in DeePMD-kit
When you try to build a second time when installing DeePMD-kit, files produced before may contribute to failure. Thus, you may clear them by
```bash
cd build
rm -r *
```
and redo the `cmake` process.
