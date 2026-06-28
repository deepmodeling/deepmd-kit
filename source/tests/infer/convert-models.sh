#!/bin/bash

set -ev

SCRIPT_PATH=$(dirname $(realpath -s $0))

# The Python output backend for .savedmodel is TF2. The C++ API loads
# SavedModel artifacts through the TensorFlow C API loader historically named
# DeepPotJAX.
dp convert-backend ${SCRIPT_PATH}/deeppot_sea.yaml ${SCRIPT_PATH}/deeppot_sea.savedmodel
dp convert-backend ${SCRIPT_PATH}/deeppot_dpa.yaml ${SCRIPT_PATH}/deeppot_dpa.savedmodel
