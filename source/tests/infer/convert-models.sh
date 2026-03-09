#!/bin/bash

set -ev

SCRIPT_PATH=$(dirname $(realpath -s $0))

dp convert-backend ${SCRIPT_PATH}/deeppot_sea.yaml ${SCRIPT_PATH}/deeppot_sea.savedmodel
dp convert-backend ${SCRIPT_PATH}/deeppot_dpa.yaml ${SCRIPT_PATH}/deeppot_dpa.savedmodel

# Generate .pth and .pt2 model files for C++ tests
python ${SCRIPT_PATH}/gen_dpa1.py
python ${SCRIPT_PATH}/gen_dpa2.py
python ${SCRIPT_PATH}/gen_dpa3.py
python ${SCRIPT_PATH}/gen_fparam_aparam.py
