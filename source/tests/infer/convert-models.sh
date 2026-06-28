#!/bin/bash

set -ev

SCRIPT_PATH=$(dirname $(realpath -s $0))

# .savedmodel is the JAX/JAX2TF output suffix. .savedmodeltf is the TF2 output
# suffix. The C++ API loads both SavedModel artifacts through the TensorFlow C
# API loader historically named DeepPotJAX.
dp convert-backend ${SCRIPT_PATH}/deeppot_sea.yaml ${SCRIPT_PATH}/deeppot_sea.savedmodel
dp convert-backend ${SCRIPT_PATH}/deeppot_dpa.yaml ${SCRIPT_PATH}/deeppot_dpa.savedmodel
dp convert-backend ${SCRIPT_PATH}/deeppot_sea.yaml ${SCRIPT_PATH}/deeppot_sea.savedmodeltf
dp convert-backend ${SCRIPT_PATH}/deeppot_dpa.yaml ${SCRIPT_PATH}/deeppot_dpa.savedmodeltf
