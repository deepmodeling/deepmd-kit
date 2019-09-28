# Install tensorflow-gpu's C++ interface 
The tensorflow's C++ interface will be compiled from the source code. Firstly one installs bazel. It is highly recommended that the bazel version 0.24.1 is used. A full instruction of bazel installation can be found [here](https://docs.bazel.build/versions/master/install.html).
```bash
cd /some/workspace
wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-dist.zip
mkdir bazel-0.24.1
cd bazel-0.24.1
unzip ../bazel-0.24.1-dist.zip
./compile.sh
export PATH=`pwd`/output:$PATH
```

Firstly get the source code of the tensorflow
```bash
cd /some/workspace
git clone https://github.com/tensorflow/tensorflow tensorflow -b v1.14.0 --depth=1
cd tensorflow
```

DeePMD-kit is compiled by cmake, so we need to compile and integrate tensorflow with cmake projects. The rest of this section basically follows [the instruction provided by Tuatini](http://tuatini.me/building-tensorflow-as-a-standalone-project/). Now execute

You will answer a list of questions that help configure the building of tensorflow. It is recommended to build for Python3. You may want to answer the question like this (please replace `$tensorflow_venv` by the virtual environment directory):
```bash
./configure
Please specify the location of python. [Default is xxx]:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
AttributeError: module 'site' has no attribute 'getsitepackages'
Found possible Python library paths:
  /xxx/deepmd_gpu/tensorflow_venv/lib/python3.7/site-packages
Please input the desired Python library path to use.  Default is [xxx]

Do you wish to build TensorFlow with XLA JIT support? [Y/n]:
XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]:
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]:
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with TensorRT support? [y/N]:
No TensorRT support will be enabled for TensorFlow.

Found CUDA 10.1 in:
    /usr/local/cuda/lib64
    /usr/local/cuda/include
Found cuDNN 7 in:
    /usr/local/cuda/lib64
    /usr/local/cuda/include

Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 6.1,6.1]:

Do you want to use clang as CUDA compiler? [y/N]:
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:


Do you wish to build TensorFlow with MPI support? [y/N]:
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]:

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=gdr         	# Build with GDR support.
	--config=verbs       	# Build with libverbs support.
	--config=ngraph      	# Build with Intel nGraph support.
	--config=numa        	# Build with NUMA support.
	--config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
	--config=v2          	# Build TensorFlow 2.x instead of 1.x.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=noaws       	# Disable AWS S3 filesystem support.
	--config=nogcp       	# Disable GCP support.
	--config=nohdfs      	# Disable HDFS support.
	--config=noignite    	# Disable Apache Ignite support.
	--config=nokafka     	# Disable Apache Kafka support.
	--config=nonccl      	# Disable NVIDIA NCCL support.
Configuration finished
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
Before moving on, we need to compile the dependencies of tensorflow, including Protobuf, Eigen, nsync and absl. Firstly, protobuf
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
nsync
```bash
mkdir /tmp/nsync
cd ../../nsync
mkdir build_dir
cd build_dir
cmake -DCMAKE_INSTALL_PREFIX=/tmp/nsync/ ../
make
make install
```
And absl
```bash
cd ../../absl
bazel build
mkdir -p $tensorflow_root/include/
rsync -avzh --include '*/' --include '*.h' --exclude '*' absl $tensorflow_root/include/
touch $tensorflow_root/include/absl/numeric/int128_have_intrinsic.inc
cd ../../../../..
```
Now, copy the libraries to the tensorflow's installation directory:
```bash
mkdir $tensorflow_root/lib
cp -d bazel-bin/tensorflow/libtensorflow_cc.so* $tensorflow_root/lib/
cp -d bazel-bin/tensorflow/libtensorflow_framework.so* $tensorflow_root/lib/
cp $tensorflow_root/lib/libtensorflow_framework.so.1 $tensorflow_root/lib/libtensorflow_framework.so
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

# Troubleshooting
```bash
git: unknown command -C ...
```
This may be your git version issue, because low version of git does not support this command. Upgrading your git maybe helpful.

```bash
CMake Error: The following variables are used in this project, but they are set to NOTFOUND.
Please set them or make sure they are set and tested correctly in the CMake files:
FFTW_LIB (ADVANCED)
    linked by target "FFTW" in directory xxx
```
Currently, when building eigen package, you can delete the FFTW in the cmake file.

```bash
fatal error: absl/numeric/int128_have_intrinsic.inc: No such file or directory
```
Basicly, you could build an empty file named "int128_have_intrinsic.inc" at the same directory of "int128.h".


