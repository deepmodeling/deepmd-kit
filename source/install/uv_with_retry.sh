#!/bin/bash
# This script is used to retry the uv command if a transient network error is encountered.
# See also:
# https://github.com/astral-sh/uv/issues/2586
# https://github.com/astral-sh/uv/issues/3456
# https://github.com/astral-sh/uv/issues/3514
# https://github.com/astral-sh/uv/issues/4402
tmpstderr=$(mktemp) || exit 1
trap 'rm -f "${tmpstderr}"' EXIT
max_attempts=3
attempt=1
delay=5
while true; do
	# Keep stdout separate and wait for tee before inspecting this attempt's stderr.
	{
		uv "$@" 2>&1 >&3 | tee "${tmpstderr}" >&2
		exit_code=${PIPESTATUS[0]}
	} 3>&1
	if [ "$exit_code" -eq 0 ]; then
		exit 0
	fi
	# Retry transient network failures, including errors fetching --find-links.
	if ! grep -qE "error decoding response body|network timeout|I/O operation failed during extraction|Failed to download distribution|HTTP status (server|client) error \((5[0-9]{2}|429)([[:space:]]|\))" "${tmpstderr}"; then
		exit "$exit_code"
	fi
	if [ "$attempt" -ge "$max_attempts" ]; then
		echo "Max retry reached, exiting..."
		exit "$exit_code"
	fi
	echo "Retrying uv in $delay s..."
	sleep "$delay"
	attempt=$((attempt + 1))
	delay=$((delay * 2))
done
