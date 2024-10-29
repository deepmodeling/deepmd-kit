#!/bin/bash
set -ev

SCRIPT_PATH=$(dirname $(realpath -s $0))
cd ${SCRIPT_PATH}/..

uv sync --dev --python 3.12 --extra cpu --extra torch --extra jax --extra lmp --extra test --extra docs
pre-commit install
