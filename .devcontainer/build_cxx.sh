#!/bin/bash
set -ev

NPROC=$(nproc --all)
SCRIPT_PATH=$(dirname $(realpath -s $0))

export CMAKE_PREFIX_PATH=${SCRIPT_PATH}/../libtorch
TENSORFLOW_ROOT=$(python -c 'import importlib,pathlib;print(pathlib.Path(importlib.util.find_spec("tensorflow").origin).parent)')

mkdir -p ${SCRIPT_PATH}/../buildcxx/
cd ${SCRIPT_PATH}/../buildcxx/
cmake -D ENABLE_TENSORFLOW=ON \
	-D ENABLE_PYTORCH=ON \
	-D ENABLE_PADDLE=ON \
	-D CMAKE_INSTALL_PREFIX=${SCRIPT_PATH}/../dp/ \
	-D LAMMPS_VERSION=stable_29Aug2024_update1 \
	-D CMAKE_BUILD_TYPE=Debug \
	-D BUILD_TESTING:BOOL=TRUE \
	-D TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
	${SCRIPT_PATH}/../source
cmake --build . -j${NPROC}
cmake --install .
