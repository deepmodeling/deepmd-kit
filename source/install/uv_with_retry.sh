#!/bin/bash
# This script is used to retry the uv command if the error "error decoding response body" is encountered.
# See also:
# https://github.com/astral-sh/uv/issues/2586
# https://github.com/astral-sh/uv/issues/3456
# https://github.com/astral-sh/uv/issues/3514
# https://github.com/astral-sh/uv/issues/4402
tmpstderr=$(mktemp)
max_retry=3
while true; do
	uv "$@" 2> >(tee -a "${tmpstderr}" >&2)
	exit_code=$?
	# exit if ok
	if [ $exit_code -eq 0 ]; then
		rm -f "${tmpstderr}"
		exit 0
	fi
	# check if "error decoding response body" is in the stderr
	if grep -q "error decoding response body" "${tmpstderr}"; then
		echo "Retrying uv in 1 s..."
		max_retry=$((max_retry - 1))
		if [ $max_retry -eq 0 ]; then
			echo "Max retry reached, exiting..."
			rm -f "${tmpstderr}"
			exit 1
		fi
		sleep 1
	else
		rm -f "${tmpstderr}"
		exit $exit_code
	fi
done
