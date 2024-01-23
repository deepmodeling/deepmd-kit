set -e

SCRIPT_PATH=$(dirname $(realpath -s $0))

wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcpu.zip -O libtorch.zip
unzip libtorch.zip
mv libtorch ${SCRIPT_PATH}/../libtorch

docker run --rm -v ${SCRIPT_PATH}/../..:/root/deepmd-kit -w /root/deepmd-kit \
	tensorflow/build:${TENSORFLOW_BUILD_VERSION:-2.15}-python3.11 \
	/bin/sh -c "pip install \"tensorflow${TENSORFLOW_VERSION}\" cmake \
            && cd /root/deepmd-kit/source/install \
            && CC=/dt9/usr/bin/gcc \
               CXX=/dt9/usr/bin/g++ \
               CMAKE_PREFIX_PATH=/root/deepmd-kit/source/libtorch \
               /bin/sh package_c.sh"
