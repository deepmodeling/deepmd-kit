#!/bin/bash
set -ev

SCRIPT_PATH=$(dirname $(realpath -s $0))
cd ${SCRIPT_PATH}/..

wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip -O ~/libtorch.zip
unzip ~/libtorch.zip
