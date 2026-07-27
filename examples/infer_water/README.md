# infer_water based on C/C++ API

This is a project based on C/C++ API described in the documentation.

Build the project using

```sh
cmake -DCMAKE_PREFIX_PATH=$deepmd_root .
make
```

Building only compiles the executables; it does not create the model file.
The inference programs load a frozen model `graph.pb` from the current
directory, so generate it first by running the `convert_model` helper, which
converts the bundled test model `../../source/tests/infer/deeppot.pbtxt` into
`graph.pb`:

```sh
./convert_model
```

Run `convert_model` from this directory so the relative path to the bundled
model resolves. It requires `$deepmd_root` to be built with the TensorFlow
backend, because `graph.pb` is a TensorFlow frozen model.

Then run any of the inference examples:

```sh
./infer_water_cc
./infer_water_c
./infer_water_hpp
./infer_water_nlist
```
