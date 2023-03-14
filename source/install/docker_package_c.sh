set -e

SCRIPT_PATH=$(dirname $(realpath -s $0))

docker run --rm -v ${SCRIPT_PATH}/../..:/root/deepmd-kit -w /root/deepmd-kit \
	ghcr.io/deepmodeling/libtensorflow_cc:2.9.2_cuda11.6_centos7_cmake \
	/bin/sh -c "source /opt/rh/devtoolset-10/enable \
            && cd /root/deepmd-kit/source/install \
            && /bin/sh package_c.sh"
