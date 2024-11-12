#!/bin/bash

set -ev

dp convert-backend deeppot_sea.yaml deeppot_sea.savedmodel
dp convert-backend deeppot_dpa.yaml deeppot_dpa.savedmodel
