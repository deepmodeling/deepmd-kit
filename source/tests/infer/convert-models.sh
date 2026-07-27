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

# Model-level pair_exclude_types variant of deeppot_sea (identical weights, only
# the exclusion differs) for the DeepPotJAX InputNlist ingestion-seam test. The
# C++ seam must fold exclusion into the LAMMPS nlist before the exported
# call_lower_* runs (decision #18/A4). Both SavedModel flavours (jax
# .savedmodel + tf2 .savedmodeltf) are consumed by the C++ DeepPotJAX loader.
python ${SCRIPT_PATH}/inject_pair_exclude.py ${SCRIPT_PATH}/deeppot_sea.yaml ${SCRIPT_PATH}/deeppot_sea_pairexcl.yaml
dp convert-backend ${SCRIPT_PATH}/deeppot_sea_pairexcl.yaml ${SCRIPT_PATH}/deeppot_sea_pairexcl.savedmodel
dp convert-backend ${SCRIPT_PATH}/deeppot_sea_pairexcl.yaml ${SCRIPT_PATH}/deeppot_sea_pairexcl.savedmodeltf
