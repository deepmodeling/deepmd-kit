#!/bin/bash

# Retry a command only when it was terminated by SIGTERM (128 + SIGTERM = 143).
#
# GitHub-hosted runners have intermittently terminated the LAMMPS pytest
# process without a Python, LAMMPS, or MPI error.  Retrying only exit code 143
# masks that external transient while preserving every actionable test failure.

set -u

if [ "$#" -lt 2 ]; then
	echo "Usage: $0 MAX_ATTEMPTS COMMAND [ARG ...]" >&2
	exit 2
fi

max_attempts=$1
shift

if ! [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
	echo "MAX_ATTEMPTS must be a positive integer." >&2
	exit 2
fi

attempt=1
while true; do
	echo "Running command (attempt ${attempt}/${max_attempts}): $*"
	"$@"
	status=$?

	if [ "$status" -eq 0 ]; then
		exit 0
	fi
	if [ "$status" -ne 143 ] || [ "$attempt" -ge "$max_attempts" ]; then
		exit "$status"
	fi

	echo "Command received SIGTERM (exit 143); retrying after 2 seconds." >&2
	sleep 2
	attempt=$((attempt + 1))
done
