set -e

SCRIPT_PATH=$(dirname $(realpath -s $0))

docker run --rm -v ${SCRIPT_PATH}/../..:/root/deepmd-kit -w /root/deepmd-kit \
	tensorflow/build:${TENSORFLOW_BUILD_VERSION:-2.15}-python3.11 \
	/bin/sh -c "pip install \"tensorflow${TENSORFLOW_VERSION}\" cmake \
            && cd /root/deepmd-kit/source/install \
            && CC=/dt9/usr/bin/gcc \
               CXX=/dt9/usr/bin/g++ \
               /bin/sh package_c.sh"
