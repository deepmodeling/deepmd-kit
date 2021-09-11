# Install TensorFlow's C++ interface 
The tensorflow's C++ interface will be compiled from the source code. Firstly one installs bazel. The bazel version 3.1.0 should be used. A full instruction of bazel installation can be found [here](https://docs.bazel.build/versions/master/install.html).
```bash
cd /some/workspace
wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh
chmod +x bazel-3.1.0-installer-linux-x86_64.sh
./bazel-3.1.0-installer-linux-x86_64.sh --prefix /some/workspace/bazel
export PATH=/some/workspace/bazel/bin:$PATH
```

Firstly get the source code of the tensorflow
```bash
git clone https://github.com/tensorflow/tensorflow tensorflow -b v2.3.0 --depth=1
cd tensorflow
./configure
```

You will answer a list of questions that help configure the building of tensorflow. You may want to answer the question like the following. If you do not want to add CUDA support, please answer no.

```
Please specify the location of python. [Default is xxx]:

Found possible Python library paths:
  xxx
Please input the desired Python library path to use.  Default is [xxx]

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]:
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]:
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with TensorRT support? [y/N]:
No TensorRT support will be enabled for TensorFlow.

Found CUDA 10.2 in:
    /usr/local/cuda/lib64
    /usr/local/cuda/include
Found cuDNN 7 in:
    /usr/local/cuda/lib64
    /usr/local/cuda/include

Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 7.5,7.5]:

Do you want to use clang as CUDA compiler? [y/N]:
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]:

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=ngraph      	# Build with Intel nGraph support.
	--config=numa        	# Build with NUMA support.
	--config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
	--config=v2          	# Build TensorFlow 2.x instead of 1.x.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=noaws       	# Disable AWS S3 filesystem support.
	--config=nogcp       	# Disable GCP support.
	--config=nohdfs      	# Disable HDFS support.
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
Now, copy the libraries to the tensorflow's installation directory:
```bash
mkdir -p $tensorflow_root/lib
cp -d bazel-bin/tensorflow/libtensorflow_cc.so* $tensorflow_root/lib/
cp -d bazel-bin/tensorflow/libtensorflow_framework.so* $tensorflow_root/lib/
cp -d $tensorflow_root/lib/libtensorflow_framework.so.2 $tensorflow_root/lib/libtensorflow_framework.so
```
Then copy the headers
```bash
mkdir -p $tensorflow_root/include/tensorflow
rsync -avzh --exclude '_virtual_includes/' --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-bin/ $tensorflow_root/include/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' tensorflow/cc $tensorflow_root/include/tensorflow/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' tensorflow/core $tensorflow_root/include/tensorflow/
rsync -avzh --include '*/' --include '*' --exclude '*.cc' third_party/ $tensorflow_root/include/third_party/
rsync -avzh --include '*/' --include '*' --exclude '*.txt' bazel-tensorflow/external/eigen_archive/Eigen/ $tensorflow_root/include/Eigen/
rsync -avzh --include '*/' --include '*' --exclude '*.txt' bazel-tensorflow/external/eigen_archive/unsupported/ $tensorflow_root/include/unsupported/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-tensorflow/external/com_google_protobuf/src/google/ $tensorflow_root/include/google/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-tensorflow/external/com_google_absl/absl/ $tensorflow_root/include/absl/
```

# Troubleshooting
```bash
git: unknown command -C ...
```
This may be your git version issue, because low version of git does not support this command. Upgrading your git maybe helpful.
