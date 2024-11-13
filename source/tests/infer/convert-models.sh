#!/bin/bash

set -ev

SCRIPT_PATH=$(dirname $(realpath -s $0))

dp convert-backend ${SCRIPT_PATH}/deeppot_sea.yaml ${SCRIPT_PATH}/deeppot_sea.savedmodel
dp convert-backend ${SCRIPT_PATH}/deeppot_dpa.yaml ${SCRIPT_PATH}/deeppot_dpa.savedmodel
