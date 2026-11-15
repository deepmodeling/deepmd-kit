#!/bin/bash
set -euo pipefail

# Keep every pytest process separate, including retry_on_sigterm.sh attempts.
# exec preserves the test command's exit status and signal handling.
if [[ $# -lt 2 ]]; then
	echo "Usage: $0 REPORT_NAME COMMAND [ARG ...]" >&2
	exit 2
fi
REPORT_NAME=$1
shift
SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
REPO_ROOT=$(realpath "${SCRIPT_PATH}/../..")
REPORT_ROOT=${REPO_ROOT}/source/build_tests/pytest-reports
mkdir -p "${REPORT_ROOT}"
REPORT_DIR=$(mktemp -d "${REPORT_ROOT}/run.XXXXXX")
exec "$@" --junitxml="${REPORT_DIR}/${REPORT_NAME}.xml"
