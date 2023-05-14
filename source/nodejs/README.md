# DeePMD-kit.js

DeePMD-kit Node.js package to infer Deep Potential models.

## Installation

### Install form npm

[![Version](https://img.shields.io/npm/v/deepmd-kit.svg)](https://npmjs.com/package/deepmd-kit)
[![Downloads](https://img.shields.io/npm/dt/deepmd-kit.svg)](https://npmjs.com/package/deepmd-kit)

```sh
npm i deepmd-kit
# Or if you want to install globally
npm i -g deepmd-kit
```

### Build from source

Firstly, install [Node.js](https://nodejs.org/), [SWIG](https://www.swig.org) (v4.1.0 for Node.js v12-v18 support), and [node-gyp](https://github.com/nodejs/node-gyp) globally.

When using CMake to build DeePMD-kit, set argument `BUILD_NODEJS_IF=ON` and `NODEJS_INCLUDE_DIRS=/path/to/nodejs/include` (the path to the include directory of Node.js):

```sh
cmake -D BUILD_NODEJS_IF=ON \
      -D NODEJS_INCLUDE_DIRS=/path/to/nodejs/include \
      .. # and other arguments
make
make install
```

After installing DeePMD-kit, two files, `bind.gyp` and `deepmdJAVASCRIPT_wrap.cxx` will be generated in `source/nodejs`.

Go to this directory, and install the package globally:

```sh
cd $deepmd_source_dir/source/nodejs
npm i
npm link
```

## Simple usage

See [tests/test_deeppot.js](tests/test_deeppot.js) for an simple example.

```sh
cd tests
node test_deeppot.js
```
