# Install Node.js interface

DeePMD-kit has an inference interface for Node.js, the most common programming language in the world, via a wrapper of the header-only C++ interface created by SWIG.

## Install from npm

```sh
npm i deepmd-kit
# Or if you want to install globally
npm i -g deepmd-kit
```

## Build from source

Before building DeePMD-kit, install [Node.js](https://nodejs.org/), [SWIG](https://www.swig.org) (v4.1.0 for Node.js v12-v18 support), and [node-gyp](https://github.com/nodejs/node-gyp) globally.

When using CMake to [build DeePMD-kit from source](./install-from-source.md), set argument `BUILD_NODEJS_IF=ON` and `NODEJS_INCLUDE_DIRS=/path/to/nodejs/include` (the path to the include directory of Node.js):

```sh
cmake -D BUILD_NODEJS_IF=ON \
      -D NODEJS_INCLUDE_DIRS=/path/to/nodejs/include \
      .. # and other arguments
make
make install
```

After installing DeePMD-kit, two files, `bind.gyp` and `deepmdJAVASCRIPT_wrap.cxx` will be generated in `$deepmd_source_dir/source/nodejs`.

Go to this directory, and install the Node.js package globally:

```sh
cd $deepmd_source_dir/source/nodejs
npm i
npm link
```

The `deepmd-kit` package should be globally available in Node.js environments:

```js
const deepmd = require("deepmd-kit");
```
