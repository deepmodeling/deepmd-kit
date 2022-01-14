# Install tensorflow's C++ interface
The tensorflow's C++ interface will be compiled from the source code. Firstly one installs bazel. It is highly recommended that the bazel version 0.10.0 is used. A full instruction of bazel installation can be found [here](https://docs.bazel.build/versions/master/install.html).
```bash
cd /some/workspace
wget https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-dist.zip
mkdir bazel-0.10.0
cd bazel-0.10.0
unzip ../bazel-0.10.0-dist.zip
./compile.sh
export PATH=`pwd`/output:$PATH
```

Firstly get the source code of the tensorflow
```bash
cd /some/workspace
git clone https://github.com/tensorflow/tensorflow tensorflow -b v1.8.0 --depth=1
cd tensorflow
```

DeePMD-kit is compiled by cmake, so we need to compile and integrate tensorflow with cmake projects. The rest of this section basically follows [the instruction provided by Tuatini](http://tuatini.me/building-tensorflow-as-a-standalone-project/). Now execute
```bash
./configure
```
You will answer a list of questions that help configure the building of tensorflow. It is recommended to build for Python3. You may want to answer the question like this (please replace `$tensorflow_venv` by the virtual environment directory):
```bash
Please specify the location of python. [Default is $tensorflow_venv/bin/python]:
```
The library path for Python should be set accordingly.

Now build the shared library of tensorflow:
```bash
bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so
```
You may want to add options `--copt=-msse4.2`,  `--copt=-mavx`, `--copt=-mavx2` and `--copt=-mfma` to enable SSE4.2, AVX, AVX2 and FMA SIMD accelerations, respectively. It is noted that these options should be chosen according to the CPU architecture. If the RAM becomes an issue of your machine, you may limit the RAM usage by using `--local_resources 2048,.5,1.0`. 

Now I assume you want to install tensorflow in directory `$tensorflow_root`. Create the directory if it does not exists
```bash
mkdir -p $tensorflow_root
```
Before moving on, we need to compile the dependencies of tensorflow, including Protobuf, Eigen and nsync. Firstly, protobuf
```bash
mkdir /tmp/proto
tensorflow/contrib/makefile/download_dependencies.sh
cd tensorflow/contrib/makefile/downloads/protobuf/
./autogen.sh
./configure --prefix=/tmp/proto/
make
make install
```
Then Eigen
```bash
mkdir /tmp/eigen
cd ../eigen
mkdir build_dir
cd build_dir
cmake -DCMAKE_INSTALL_PREFIX=/tmp/eigen/ ../
make install
```
And nsync
```bash
mkdir /tmp/nsync
cd ../../nsync
mkdir build_dir
cd build_dir
cmake -DCMAKE_INSTALL_PREFIX=/tmp/nsync/ ../
make
make install
cd ../../../../../..
```
Now, copy the libraries to the tensorflow's installation directory:
```bash
mkdir $tensorflow_root/lib
cp bazel-bin/tensorflow/libtensorflow_cc.so $tensorflow_root/lib/
cp bazel-bin/tensorflow/libtensorflow_framework.so $tensorflow_root/lib/
cp /tmp/proto/lib/libprotobuf.a $tensorflow_root/lib/
cp /tmp/nsync/lib/libnsync.a $tensorflow_root/lib/
```
Then copy the headers
```bash
mkdir -p $tensorflow_root/include/tensorflow
cp -r bazel-genfiles/* $tensorflow_root/include/
cp -r tensorflow/cc $tensorflow_root/include/tensorflow
cp -r tensorflow/core $tensorflow_root/include/tensorflow
cp -r third_party $tensorflow_root/include
cp -r /tmp/proto/include/* $tensorflow_root/include
cp -r /tmp/eigen/include/eigen3/* $tensorflow_root/include
cp -r /tmp/nsync/include/*h $tensorflow_root/include
```
Now clean up the source files in the header directories:
```bash
cd $tensorflow_root/include
find . -name "*.cc" -type f -delete
```
The temporary installation directories for the dependencies can be removed:
```bash
rm -fr /tmp/proto /tmp/eigen /tmp/nsync
```
