#!/bin/bash
set -e

#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))

#------------------
# upload to codecov
cd ${SCRIPT_PATH}
bash <(curl -s https://codecov.io/bash) || echo "Codecov did not collect coverage reports"

