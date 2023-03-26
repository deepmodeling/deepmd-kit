set -e

SCRIPT_PATH=$(dirname $(realpath -s $0))

docker run --rm -v ${SCRIPT_PATH}/../..:/root/deepmd-kit -w /root/deepmd-kit \
	tensorflow/build:2.13-python3.11 \
	/bin/sh -c "pip install tensorflow cmake \
            && cd /root/deepmd-kit/source/install \
            && CC=/dt9/usr/bin/gcc \
               CXX=/dt9/usr/bin/g++ \
               /bin/sh package_c.sh"
